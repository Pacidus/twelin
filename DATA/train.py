import numpy as np
import Tools as ts
import tensorflow.keras as kr


def train(model, xy, files, prop, trainsize, sample, Nepoch, Nsets):
    """
    train the model with a dataset composed with some proportions of others

    Parameters
    ----------

    model: keras.Model
        the model we want to train
    xy: python function, pd.dataframe -> (x,y)
        this function transform a pandas dataframe to a tuple input output for
        the training
    files: list(pq.Parquetfile)
        should be the same len as prop and it list the parquet file we want
        compose the training set from
    prop: list(float), (or equivalent)
        the given proportion we should put from every file (it's normalized)
    trainsize: int
        size of the training set
    sample: int
        size of the sample wen we're doing file stuffs (look at Tools)
    Nepoch: int
        number of epoch for the fit function
    Nsets: int
        number of sets of training before stoping the training
    """

    ratios = np.array(prop) / np.sum(prop)
    sizes = 1 + np.round(sample * ratios)
