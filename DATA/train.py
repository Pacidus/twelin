import numpy as np
import pandas as pd
import DATA.Tools as ts
import pyarrow.parquet as pq


def getdta(files, sizes, sample):
    """
    get the data for the training

    Parameters
    ----------

    files : list(pq.Parquetfile)
        should be the same len as prop and it list the parquet file we want
        compose the training set from
    sizes : list(int)
        the given number of values we want to get from each files
    sample : int
        size of the samples
    Returns
    -------

    pd.DataFrame :
        Concat the data of the different files
    """
    dta = [ts.getrand(i, j, sample=sample) for i, j in zip(files, sizes)]
    return pd.concat(dta, ignore_index=True)


def train(
    model, xy, files, prop, trainsize, sample, Nepoch, Nsets, nfile=None
):
    """
    train the model with a dataset composed with some proportions of others

    Parameters
    ----------

    model : keras.Model
        the model we want to train
    xy : python function, pd.dataframe -> (x,y)
        this function transform a pandas dataframe to a tuple input output for
        the training
    files : list(pq.Parquetfile)
        should be the same len as prop and it list the parquet file we want
        compose the training set from
    prop : list(float), (or equivalent)
        the given proportion we should put from every file (it's normalized)
    trainsize: int
        size of the training set
    sample : int
        size of the sample wen we're doing file stuffs (look at Tools)
    Nepoch : int
        number of epoch for the fit function
    Nsets : int
        number of sets of training before stoping the training
    nfile : str, optinal
        the name of the csv file the data will be concatenate into
    """
    pfiles = [pq.ParquetFile(i) for i in files]
    ratios = np.array(prop) / np.sum(prop)
    sizes = 1 + np.round(trainsize * ratios).astype("int64")

    for i in range(Nsets):
        Tdata = getdta(pfiles, sizes, sample)
        x, y = xy(Tdata)
        hist = model.fit(x=x, y=y, epochs=Nepoch, shuffle=True)
        if nfile:
            with open(nfile, "a") as ofile:
                np.savetxt(ofile, hist.history["loss"])
