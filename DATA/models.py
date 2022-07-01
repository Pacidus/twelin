from tensorflow import keras as kr


#################
# -- Aliases -- #
#################

layers = kr.layers


################
# -- Models -- #
################
class DenseBlock(kr.layers.Layer):
    """
    Dense hidden layers block

    Parameters
    ----------

    hidden_units: list(int)
        the size of the different layers in the block
    activation: str, optional
        the name of the activation function
    """

    def __init__(self, hidden_units, activation="relu"):
        kwargs = {"activation": activation}
        super(DenseBlock, self).__init__()
        self.hidden_units = hidden_units
        self.activation = activation
        self.dense_layers = [layers.Dense(u, **kwargs) for u in hidden_units]

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        return {
            "hidden_units": self.hidden_units,
            "activation": self.activation,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Bicephale(kr.Model):
    r"""
    Bicephale model

    the geometry of the model is the folowing :

         +-------+
         | Input |
         +-------+
           /   \
          /     \
    +-------+ +-------+
    | Dense | | Dense |
    | Block | | Block |
    +-------+ +-------+
        |         |
    +-------+ +-------+
    |  Sig  | |  Lin  |
    +-------+ +-------+
        \         /
         \       /
      +-------------+
      | dot product |
      +-------------+

    Parameters
    ----------

    Densize: list(int) or (list(int), list(int))
        the size of the different layers in the DenseBlock if it's a tuple the
        first goes define the sig part (the classifier) and the second define
        the lin part (the estimator).

    activation: str or (str, str), optional
        the name of the activation function in the DenseBlock if it's a tuple
        the first goes define the sig part (the classifier) and the second
        define the lin part (the estimator).
    """

    def __init__(self, Densize, activation="relu"):
        super(Bicephale, self).__init__()

        if type(Densize) is tuple:
            self.Densize = Densize
        else:
            self.Densize = (Densize, Densize)

        if type(activation) is tuple:
            self.activation = activation
        else:
            self.activation = (activation, activation)

        self.hiden_sig = DenseBlock(self.Densize[0], self.activation[0])
        self.hiden_lin = DenseBlock(self.Densize[1], self.activation[1])
        self.sig = layers.Dense(1, activation="sigmoid")
        self.lin = layers.Dense(1, activation="linear")
        self.dot = layers.Dot(1)

    def call(self, inputs):
        self.inputs = inputs
        s = self.hidden_sig(inputs)
        l = self.hidden_lin(inputs)
        s = self.sig(s)
        l = self.lin(l)
        return self.dot([s, l])

    def get_cephales(self):
        return [self.input, self.sig], [self.input, self.lin]

    def get_config(self):
        return {"Densize": self.Densize, "activation": self.activation}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
