######################
# --    Imports   -- #
# -- Nothing much -- #
######################
import numpy as np
import pandas as pd
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


#################
# -- Classes -- #
#################


class upstats:
    def __init__(self):
        self.mean = 0
        self.var = 0
        self.min = 0
        self.max = 0
        self.N = 0

    def update(self, df):
        Nb = df.shape[0]
        Nab = self.N + Nb
        d = df.mean - self.mean()
        self.mean += d * Nb / Nab
        self.var += df.var() + d * d * (self.N * Nb) / Nab
        self.N = Nab
        self.min = np.minimum(self.min, df.min())
        self.max = np.maximum(self.max, df.max())


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
    args : list or tuple
        arguments of func.
    kwargs : dict
        keyword arguments of func
    index : list(str)
        index we want to keep from pf
    sample : int
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
    pf : pq.ParquetFile
        the parquet file we want to convert
    """
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


def stats(pf, index=didx, sample=dsmpl):
    """
    Gather statistical data over the parquet file

    Parameters
    ----------
    pf : pq.ParquetFile
        the parquet file we want to itterate through.
    index : list(str)
        index we want to keep from pf
    sample : int
        size of the sample

    Returns
    -------
    upstats:
        class with following public variables
            mean, variation, min, max, N
    """
    vals = upstats()
    aparquet(pf, vals.update, index=index, sample=sample)
    return vals
