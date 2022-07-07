import tensorflow.keras as kr


#################
# -- Aliases -- #
#################

layers = kr.layers


################
# -- Models -- #
################
class DenseBlock:
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
        self.hidden_units = hidden_units
        self.activation = activation

    def __call__(self, inputs):
        kwargs = {"activation": self.activation}
        self.dense_layers = [
            layers.Dense(u, **kwargs) for u in self.hidden_units
        ]
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x


class Bicephale:
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
        if type(Densize) is tuple:
            self.Densize = Densize
        else:
            self.Densize = (Densize, Densize)
        if type(activation) is tuple:
            self.activation = activation
        else:
            self.activation = (activation, activation)
        self.hidden_sig = DenseBlock(self.Densize[0], self.activation[0])
        self.hidden_lin = DenseBlock(self.Densize[1], self.activation[1])

    def __call__(self, inputs):
        s = self.hidden_sig(inputs)
        l = self.hidden_lin(inputs)

        sig = layers.Dense(
            1, activation="sigmoid", name="cla_out", use_bias=False
        )
        lin = layers.Dense(1, activation="linear", name="est_out")
        dot = layers.Dot(1)

        s = sig(s)
        l = lin(l)
        return dot([s, l])


def cephalise(model):
    """
    Return the two sub models of the Bicephale model

    Paramaters
    ----------

    model : keras.Model
    should be the Bicephal model
    """
    cla = kr.Model(model.input, model.get_layer("cla_out").output)
    est = kr.Model(model.input, model.get_layer("est_out").output)
    return cla, est
