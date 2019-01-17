#import tensorflow as tf
#import tensorflow.contrib.layers as layers
from fair_hmumu.utils import Saveable


class Model(Saveable):

    def __init__(self, name, hps):

        self.name = name
        self.hps = hps

    def __str__(self):
        return '{}: {}'.format(self.classname(), str(self.hps))


class Classifier(Model):
    
    def forward(self, input_layer):

        pass 
        ## build the graph in the scope
        #with tf.variable_scope(self.name):

        #    # input layer
        #    lay = input_layer

        #    # hidden layers
        #    for layer in range(int(self.hps['depth'])):
        #        lay = layers.relu(lay, int(self.hps['n_units']))

        #    # output layer
        #    self.output = layers.linear(lay, self.hps['n_classes'])

        #these_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

        #return self.outputs, these_vars

    def loss(self, target):

        pass
        ## build the graph
        #one_hot = tf.one_hot(target, depth=self.hps['n_classes'])

        ## loss
        #loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot, logits=self.outputs)
        #self.loss = tf.math.reduce_mean(loss)

        #return self.loss



