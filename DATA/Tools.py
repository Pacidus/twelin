######################
# --    Imports   -- #
# -- Nothing much -- #
######################
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

#########################
# -- Global Varibles -- #
# --      AKA        -- #
# --    Aliases      -- #
# Statics variables ofc #
#########################

dsmpl = int(1e6)
didx = None
dkwa = dict()
darg = []
Uniq = np.unique

#################
# -- Classes -- #
#################


class upstats:
    """
    A class to compute stats through batches
    """

    def __init__(self):
        self.mean = 0.0
        self.var = 0.0
        self.min = 0.0
        self.max = 0.0
        self.N = 0.0
        self.__first__ = True

    def update(self, df):
        """
        update the statistics of the hole set through batches

        Parameters
        ----------
        df : pd.DataFrame
            the new batch to update the stats with
        """
        if not self.__first__:
            Na = self.N
            Nb = df.shape[0]
            Nab = Na + Nb
            d = df.mean() - self.mean
            M2a = Na * self.var
            M2b = Nb * df.var()
            self.mean += d * Nb / Nab
            self.var = (M2a + M2b + (d * d * Na * Nb / Nab)) / Nab
            self.N = Nab
            self.min = np.minimum(self.min, df.min())
            self.max = np.maximum(self.max, df.max())
        else:
            self.mean = df.mean()
            self.var = df.var()
            self.min = df.min()
            self.max = df.max()
            self.N = df.shape[0]
            self.__first__ = False


###################
# -- Functions -- #
###################


def aparquet(pf, func, args=darg, kwargs=dkwa, index=didx, sample=dsmpl):
    """
    apply a function on a parquet file by sample.

    Parameters
    ----------
    pf : pq.ParquetFile
        the parquet file we want to itterate through.
    func : python function
        funtion that take a panda dataframe in input and *args, **kwargs.
    args : list or tuple, optional
        arguments of func.
    kwargs : dict, optional
        keyword arguments of func
    index : list(str), optional
        index we want to keep from pf
    sample : int, optional
        size of the sample

    Returns
    -------
    list
        list of the outputs of func(dtf, *args, **kwargs)
    """
    batches = pf.iter_batches(sample, columns=index)
    out = []
    for batch in batches:
        data = batch.to_pandas()
        out.append(func(data, *args, **kwargs))
    return out


def parquet2csv(name, pf, index=didx, sample=dsmpl):
    """
    Convert a parquet file to a csv file

    Parameters
    ----------
    name : str
        the name of the file
    pf : pq.ParquetFile or str
        the parquet file or path
    index : list(str), optional
        index we want to keep from pf
    sample : int, optional
        size of the sample
    """
    if type(pf) is str:
        pf = pq.ParquetFile(pf)

    kwargs = {"comments": "", "delimiter": ","}
    if index is None:
        index = pf.schema_arrow.names
    np.savetxt(name, [], header=",".join(index), **kwargs)
    with open(name, "a") as csvfile:
        aparquet(
            pf,
            lambda dtf: np.savetxt(csvfile, dtf.values, **kwargs),
            index=index,
        )


def csv2parquet(pname, name, index=didx, sample=dsmpl):
    """
    Convert a csv file into a parquet file

    Parameters
    ----------
    pname : str
        the path of the parquet file
    name : str
        the name of the file
    index : list(str), optional
        index we want to keep from pf
    sample : int, optional
        size of the sample
    """
    with pd.read_csv(name, usecols=index, chunksize=sample) as reader:
        chunk = next(reader)
        table = pa.Table.from_pandas(chunk)
        pqwriter = pq.ParquetWriter(pname, table.schema)
        pqwriter.write_table(table)
        for chunk in reader:
            table = pa.Table.from_pandas(chunk)
            pqwriter.write_table(table)


def stats(pf, index=didx, sample=dsmpl):
    """
    Gather statistical data over the parquet file

    Parameters
    ----------
    pf : pq.ParquetFile or str
        the parquet file or path we want to itterate through.
    index : list(str), optional
        index we want to keep from pf
    sample : int, optional
        size of the sample

    Returns
    -------
    upstats:
        class with following public variables
            mean, variation, min, max, N
    """
    if type(pf) is str:
        pf = pq.ParquetFile(pf)
    vals = upstats()
    aparquet(pf, vals.update, index=index, sample=sample)
    return vals


def sparq(pf, goto, name="out", index=didx, sample=dsmpl):
    """
    Split a parquet file into many subparquet file

    Parameters
    ----------
    pf : pq.ParquetFile
        the parquetfile we want to split
    goto : python function
        function that return a list that assign any row to a label,
        if it returns -1 no file is created and the row is discarded.
    index : list(str), optional
        index we want to keep from pf
    sample : int, optional
        size of the sample
    """
    if index is not None:
        schema = pa.schema([i for i in pf.schema_arrow if i.name in index])
    else:
        schema = pf.schema_arrow
    outs = aparquet(pf, lambda df: Uniq(goto(df)), index=index, sample=sample)
    Outs = []
    for o in outs:
        for i in o:
            Outs.append(i)
    outs = Uniq(Outs)
    outs = outs[~(outs == -1)]
    pqfs = {i: pq.ParquetWriter(f"{name}_{i}.parquet", schema) for i in outs}

    def split(df):
        vec = goto(df)
        sns = np.unique(vec)
        sns = sns[~(sns == -1)]
        for sn in sns:
            pqfs[sn].write_table(
                pa.Table.from_pandas(df[vec == sn].reset_index(drop=True))
            )

    aparquet(pf, split, index=index, sample=sample)


def getrand(pf, num, index=didx, sample=dsmpl):
    """
    Get random values in the parquet file

    Parameters
    ----------
    pf : pq.ParquetFile
        the parquetfile we want to split
    num : int
        number of element we want to get
    index : list(str), optional
        index we want to keep from pf
    sample : int, optional
        size of the sample

    Returns
    -------
    pd.DataFrame : the values selected
    """
    Ntot = pf.metadata.num_rows
    kwargs = {"ignore_index": True}

    class Get:
        def __init__(self):
            self.vals = np.random.choice(Ntot, num, replace=(Ntot < num))

        def __call__(self, df):
            mask = self.vals < len(df.index)
            val = self.vals[mask]
            self.vals = self.vals[~mask] - len(df.index)
            return df.loc[val]

    return pd.concat(aparquet(pf, Get(), index=index, sample=sample), **kwargs)
