import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.contrib import layers
from fair_hmumu.utils import Saveable


class Model(Saveable):

    def __init__(self, name, hps):

        super().__init__(name)
        self.hps = hps

    def __str__(self):
        return '{}: {}'.format(self.classname(), str(self.hps))


class Classifier(Model):

    def __init__(self, name, hps):

        super().__init__(name, hps)
        self.logits = None
        self.proba = None
        self.tf_vars = None
        self.loss = None

    @classmethod
    def create(cls, name, hps):

        type_map = {'DNN':DeterministicClassifier,
                    'DeterministicClassifier':DeterministicClassifier,
                    'ProbabilisticClassifier':ProbabilisticClassifier}

        if hps['type'] not in type_map:
            raise ValueError('Unknown Adversary type {}.'.format(hps['type']))

        classifier = type_map[hps['type']]

        return classifier(name, hps)

    def make_loss(self, target):

        # build the graph
        one_hot = tf.one_hot(tf.reshape(target, shape=[-1]), depth=self.hps['n_classes'])

        # loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot, logits=self.logits)
        self.loss = tf.math.reduce_mean(loss)


class DeterministicClassifier(Classifier):

    def make_forward_pass(self, input_layer):

        # build the graph in the scope
        with tf.variable_scope(self.name):

            # input layer
            layer = input_layer

            # hidden layers
            for _ in range(int(self.hps['depth'])):
                layer = layers.relu(layer, int(self.hps['n_units']))

            # output layer
            self.logits = layers.linear(layer, self.hps['n_classes'])

            # and an extra layer for getting the predictions directly
            self.proba = tf.reshape(layers.softmax(self.logits)[:, 1], shape=(-1, 1))

        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


class ProbabilisticClassifier(Classifier):

    def make_forward_pass(self, input_layer):

        # useful numbers
        depth = int(self.hps['depth'])
        n_units = int(self.hps['n_units'])
        n_components = int(self.hps['n_components'])

        # use the scope, Luke
        with tf.variable_scope(self.name):
            
            # input layer
            layer = input_layer

            # hidden layers
            for _ in range(depth):
                layer = layers.relu(layer, n_units)

            # N components gaussian mix parameters
            pdf_pars = layers.linear(layer, 3*n_components)

            pi = tf.nn.softmax(pdf_pars[:, :n_components])
            mu = pdf_pars[:, n_components:2*n_components]
            sigma = tf.exp(pdf_pars[:, 2*n_components:])

            # sample the output from the probability density function
            fractions = tfd.Categorical(probs=pi)
            normals = [tfd.Normal(loc=mu[:, i], scale=sigma[:, i]) for i in range(n_components)]
            mixture = tfd.Mixture(cat=fractions, components=normals)

            # prepare logits and probabilities
            sample = tf.reshape(mixture.sample(), shape=(-1, 1))
            self.proba = tf.math.sigmoid(sample)
            self.logits = tf.concat([1-self.proba, self.proba], axis=1)

        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


class Adversary(Model):

    def __init__(self, name, hps):

        super().__init__(name, hps)
        self.loss = None
        self.tf_vars = None

    @classmethod
    def create(cls, name, hps):

        type_map = {None:DummyAdversary,
                    'GaussMixNLL':GaussMixNLLAdversary,
                    'ExpoGaussNLL':ExpoGaussNLLAdversary,
                    'MINE':MINEAdversary}

        if hps['type'] not in type_map:
            raise ValueError('Unknown Adversary type {}.'.format(hps['type']))

        adversary = type_map[hps['type']]

        return adversary(name, hps)


class DummyAdversary(Adversary):

    def make_loss(self, _, __):

        with tf.variable_scope(self.name):
            dummy_var = tf.Variable(0.001, name='dummy')
            self.loss = tf.math.abs(dummy_var) # i.e. goes to zero

        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


class ExpoGaussNLLAdversary(Adversary):

    def __init__(self, name, hps):

        super().__init__(name, hps)
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

        with tf.variable_scope(self.name):

            # input layer
            layer = self.proba

            # define the output of a network (depends on number of components)
            for _ in range(self.hps['depth']):
                layer = layers.relu(layer, self.hps['n_units'])

            # output layer: (amplitude, rate) for expo, (mu, sigma, amplitude) for gauss
            output = layers.linear(layer, 5)

            # make sure amplitudes (pi) are normalised, and sigmas are positive
            pi = tf.nn.softmax(output[:, 0:2])
            rate = tf.reshape(tf.sigmoid(output[:, 2]), shape=(-1, 1))
            mu = tf.reshape(output[:, 3], shape=(-1, 1))
            sigma = tf.reshape(tf.exp(output[:, 4]), shape=(-1, 1))

            # interpret the output layers as nll parameters
            self.nll_pars = tf.concat([pi, rate, mu, sigma], axis=1)

        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def _make_loss(self):

        # get parameters of the nll
        pi_exp = self.nll_pars[:, 0]
        pi_gauss = self.nll_pars[:, 1]
        rate = self.nll_pars[:, 2]
        mu = self.nll_pars[:, 3]
        sigma = self.nll_pars[:, 4]

        # get the mass (x value)
        mass = tf.reshape(self.sensitive, shape=[-1]) / 100.

        # exponential part
        expo_part = pi_exp * rate * tf.math.exp(-rate * mass)

        # gaussian part
        normalisation = pi_gauss * (1. / np.sqrt(2. * np.pi)) / sigma
        exp = tf.math.exp(-(mass - mu) ** 2 / (2. * sigma ** 2))
        gauss_part = normalisation * exp
        
        # build the likelihood
        likelihood = expo_part + gauss_part

        # make the loss
        self.nll = - tf.math.log(likelihood)
        self.loss = tf.reduce_mean(self.nll)


class GaussMixNLLAdversary(Adversary):

    def __init__(self, name, hps):

        super().__init__(name, hps)
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

        likelihood = 0
        for c in range(n_components):

            # normalisation
            norm_vec = tf.reshape(pi[:, c] * (1. / np.sqrt(2. * np.pi)) / sigma[:, c], shape=(-1, 1))

            # exponential
            mu_vec = tf.reshape(mu[:, c], shape=(-1, 1))
            sigma_vec = tf.reshape(sigma[:, c], shape=(-1, 1))
            exp = tf.math.exp(-(self.sensitive - mu_vec) ** 2 / (2. * sigma_vec ** 2))

            # add to likelihood
            likelihood += norm_vec * exp

        # make the loss
        self.nll = - tf.math.log(likelihood)
        self.loss = tf.reduce_mean(self.nll)


class MINEAdversary(Adversary):

    def __init__(self, name, hps):

        super().__init__(name, hps)
        self.proba = None
        self.sensitive = None

    def make_loss(self, proba, sensitive):

        # store the input placeholders
        self.proba = tf.reshape(proba, shape=(-1, 1))
        self.sensitive = tf.reshape(sensitive, shape=(-1, 1))

        # aliases
        x_in = self.proba
        y_in = self.sensitive
        depth = self.hps['depth']
        n_units = self.hps['n_units']

        # use scope to keep track of vars
        with tf.variable_scope(self.name):
            
            # shuffle one of them
            y_shuffle = tf.random_shuffle(y_in)
            x_conc = tf.concat([x_in, x_in], axis=0)
            y_conc = tf.concat([y_in, y_shuffle], axis=0)

            # compute the forward pass
            layer_x = layers.linear(x_conc, n_units)
            layer_y = layers.linear(y_conc, n_units)
            layer = tf.nn.relu(layer_x + layer_y)

            for _ in range(depth):
                layer = layers.relu(layer, n_units)

            output = layers.linear(layer, 1)

            # split in T_xy and T_x_y
            N_batch = tf.shape(x_in)[0]
            T_xy = output[:N_batch]
            T_x_y = output[N_batch:]

            # compute the loss
            self.loss = - (tf.reduce_mean(T_xy, axis=0) - tf.math.log(tf.reduce_mean(tf.math.exp(T_x_y), axis=0)))

        # save variables
        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


