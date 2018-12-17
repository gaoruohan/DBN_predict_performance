"""
Multi-Layer Perceptron Class
"""
import numpy as np
import tensorflow as tf



class HiddenLayer(object):
    """Typical hidden layer of MLP"""
    def __init__(self, input, n_in, n_out, W=None, b=None,
                 activation=tf.nn.sigmoid):
        """
        input: tf.Tensor, shape [n_examples, n_in]
        n_in: int, the dimensionality of input
        n_out: int, number of hidden units
        W, b: tf.Tensor, weight and bias
        activation: tf.op, activation function
        """
        if W is None:
            bound_val = 4.0*np.sqrt(6.0/(n_in + n_out))
            W = tf.Variable(tf.random_uniform([n_in, n_out], minval=-bound_val, maxval=bound_val),
                            dtype=tf.float32, name="W")
        if b is None:
            b = tf.Variable(tf.zeros([n_out, ]), dtype=tf.float32, name="b")

        self.W = W
        self.b = b
        # the output
        sum_W= tf.matmul(input, self.W) + self.b
        self.output = activation(sum_W) if activation is not None else sum_W
        # params
        self.params = [self.W, self.b]


class MLP(object):
    """Multi-layer perceptron class"""
    def __init__(self, inpt, n_in, n_hidden, n_out):
        """
        inpt: tf.Tensor, shape [n_examples, n_in]
        n_in: int, the dimensionality of input
        n_hidden: int, number of hidden units
        n_out: int, number of output units
        """
        # hidden layer
        self.hiddenLayer = HiddenLayer(inpt, n_in=n_in, n_out=n_hidden)
        # output layer (logistic layer)
        self.outputLayer = LogisticRegression(self.hiddenLayer.output, n_in=n_hidden,
                                              n_out=n_out)
        # L1 norm
        self.L1 = tf.reduce_sum(tf.abs(self.hiddenLayer.W)) + \
                  tf.reduce_sum(tf.abs(self.outputLayer.W))
        # L2 norm
        self.L2 = tf.reduce_sum(tf.square(self.hiddenLayer.W)) + \
                  tf.reduce_sum(tf.square(self.outputLayer.W))
        # cross_entropy cost function
        self.cost = self.outputLayer.cost
        # accuracy function
        self.accuracy = self.outputLayer.accuarcy

        # params
        self.params = self.hiddenLayer.params + self.outputLayer.params
        # keep track of input
        self.input = inpt