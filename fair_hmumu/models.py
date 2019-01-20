import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from fair_hmumu.utils import Saveable


class Model(Saveable):

    def __init__(self, hps):

        super().__init__(str(hps['type']))
        self.hps = hps

    def __str__(self):
        return '{}: {}'.format(self.classname(), str(self.hps))


class Classifier(Model):

    def __init__(self, hps):

        super().__init__(hps)
        self.output = None
        self.proba = None
        self.tf_vars = None
        self.loss = None

    def make_forward_pass(self, input_layer):

        # build the graph in the scope
        with tf.variable_scope(self.name):

            # input layer
            layer = input_layer

            # hidden layers
            for _ in range(int(self.hps['depth'])):
                layer = layers.relu(layer, int(self.hps['n_units']))

            # output layer
            self.output = layers.linear(layer, self.hps['n_classes'])

            # and an extra layer for getting the predictions directly
            self.proba = tf.reshape(layers.softmax(self.output)[:, 1], shape=(-1, 1))

        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def make_loss(self, target):

        # build the graph
        one_hot = tf.one_hot(target, depth=self.hps['n_classes'])

        # loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot, logits=self.output)
        self.loss = tf.math.reduce_mean(loss)


class Adversary(Model):

    def __init__(self, hps):

        super().__init__(hps)
        self.loss = None
        self.tf_vars = None

    @classmethod
    def create(cls, hps):

        type_map = {None:DummyAdversary,
                    'GaussMixNLL':GaussMixNLLAdversary}

        if hps['type'] not in type_map:
            raise ValueError('Unknown Adversary type {}.'.format(hps['type']))

        return type_map[hps['type']](hps)

class DummyAdversary(Adversary):

    def make_loss(self, _, __):

        with tf.variable_scope(self.name):
            dummy_var = tf.Variable(0.1, name='dummy')
            self.loss = tf.math.abs(dummy_var) # i.e. goes to zero

        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

class GaussMixNLLAdversary(Adversary):

    def __init__(self, hps):

        super().__init__(hps)
        self.proba = None
        self.sensitive = None
        self.nll_pars = None
        self.nll = None

    def make_loss(self, proba, sensitive):

        # store the input placeholders
        self.proba = proba
        self.sensitive = sensitive

        # nll network
        self._make_nll()

        # nll loss
        self._make_loss()

    def _make_nll(self):

        # for convenience
        n_components = self.hps['n_components']

        with tf.variable_scope(self.name):

            # define the input layer
            layer = self.proba

            # define the output of a network (depends on number of components)
            for _ in range(self.hps['depth']):
                layer = layers.relu(layer, self.hps['n_units'])

            # output layer: (mu, sigma, amplitude) for each component
            output = layers.linear(layer, 3*n_components)

            # make sure sigmas are positive and pis are normalised
            mu = output[:, :n_components]
            sigma = tf.exp(output[:, n_components:2*n_components])
            pi = tf.nn.softmax(output[:, 2*n_components:])

            # interpret the output layers as nll parameters
            self.nll_pars = tf.concat([mu, sigma, pi], axis=1)

        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def _make_loss(self):

        # for convenience
        n_components = self.hps['n_components']

        # build the pdf (max likelihood principle)
        mu = self.nll_pars[:, :n_components]
        sigma = self.nll_pars[:, n_components:2*n_components]
        pi = self.nll_pars[:, 2*n_components:]

        pdf = 0
        for c in range(n_components):
            pdf += pi[:, c] * ((1. / np.sqrt(2. * np.pi)) / sigma[:, c] *
                               tf.math.exp(-(self.sensitive - mu[:, c]) ** 2 / (2. * sigma[:, c] ** 2)))

        # make the loss
        self.nll = - tf.math.log(pdf)
        self.loss = tf.reduce_mean(self.nll)



