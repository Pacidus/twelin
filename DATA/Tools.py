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


def parquet2csv(name, pf, sample=dsmpl):
    """
    Convert a parquet file to a csv file

    Parameters
    ----------
    name: str
        the name of the file
    pf: pq.ParquetFile
        the parquet file we want to convert
    """
    kwargs = {"comments": "", "delimiter": ","}
    head = ",".join(pf.schema_arrow.names)
    np.savetxt(name, [], header=head, **kwargs)
    with open(name, "a") as csvfile:
        aparquet(
            pf,
            lambda dtf, of, kwa: np.savetxt(csvfile, dtf.values, **kwargs),
            args=[csvfile, kwargs],
        )
