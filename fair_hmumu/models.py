import tensorflow as tf
import tensorflow.contrib.layers as layers
from fair_hmumu.utils import Saveable


class Model(Saveable):

    def __init__(self, hps):

        self.name = str(hps['type'])
        self.hps = hps

    def __str__(self):
        return '{}: {}'.format(self.classname(), str(self.hps))


class Classifier(Model):
    
    def make_forward_pass(self, input_layer):

        # build the graph in the scope
        with tf.variable_scope(self.name):

            # input layer
            lay = input_layer

            # hidden layers
            for layer in range(int(self.hps['depth'])):
                lay = layers.relu(lay, int(self.hps['n_units']))

            # output layer
            self.output = layers.linear(lay, self.hps['n_classes'])

            # and an extra layer for getting the predictions directly
            self.proba = tf.reshape(layers.softmax(self.output)[:,1], shape=(-1,1))

        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def make_loss(self, target):

        # build the graph
        one_hot = tf.one_hot(target, depth=self.hps['n_classes'])

        # loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot, logits=self.output)
        self.loss = tf.math.reduce_mean(loss)


class Adversary(Model):

    @classmethod
    def create(cls, hps):

        type_map = {None:DummyAdversary}

        if hps['type'] not in type_map:
            raise ValueError('Unknown Adversary type {}.'.format(hps['type']))

        return type_map[hps['type']](hps)

class DummyAdversary(Adversary):

    def make_loss(self, _, __):

        with tf.variable_scope(self.name):
            dummy_var = tf.Variable(0.1, name='dummy')
            self.loss = tf.math.abs(dummy_var) # i.e. goes to zero

        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)




