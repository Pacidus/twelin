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
