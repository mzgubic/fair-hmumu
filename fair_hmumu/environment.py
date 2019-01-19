import tensorflow as tf
import fair_hmumu.defs as defs
from fair_hmumu.utils import Saveable

class TFEnvironment(Saveable):

    def __init__(self, clf, adv, opt_conf, config=tf.ConfigProto(intra_op_parallelism_threads = 32,
                                                                 inter_op_parallelism_threads = 32,
                                                                 allow_soft_placement = True,
                                                                 device_count = {'CPU': 32})):

        print('--- Starting TensorFlow session')

        # store classifier and adversary
        self.clf = clf
        self.adv = adv
        self.opt_conf = opt_conf

        # make a session
        self.sess = tf.Session(config=config)

    def build(self, batch):

        print('--- Building computational graph')

        # input placeholders
        self._in = {}
        for xyzw in defs.XYZW:
            tftype = tf.int32 if xyzw == 'Y' else tf.float32
            self._in[xyzw] = tf.placeholder(tftype, shape=(None, batch[xyzw].shape[1]), name='{}_in'.format(xyzw))

        # classifier output and loss
        _, _ = self.clf.forward(self._in['X'])
        _ = self.clf.loss(self._in['Y'])

        # adversary output and loss
        # TODO

        # optimisers
        self.opt_C = tf.train.AdamOptimizer(**self.opt_conf).minimize(self.clf.loss, var_list=self.clf.tf_vars)

    def initialise_variables(self):

        print('--- Initialising TensorFlow variables')

        self.sess.run(tf.global_variables_initializer())

    def pretrain_step(self, batch):

        feed_dict = {self._in[xyzw]:batch[xyzw] for xyzw in defs.XYZW}
        self.sess.run(self.opt_C, feed_dict=feed_dict)

    def train_step_clf(self, batch):

        feed_dict = {self._in[xyzw]:batch[xyzw] for xyzw in defs.XYZW}
        #self.sess.run(self.opt_C, feed_dict=feed_dict) # TODO: opt_C to opt_CA

    def train_step_adv(self, batch):

        feed_dict = {self._in[xyzw]:batch[xyzw] for xyzw in defs.XYZW}
        # TODO

    def clf_predict(self, data):

        feed_dict = {self._in[xyzw]:data[xyzw] for xyzw in defs.XYZW}
        return self.sess.run(self.clf.predict, feed_dict=feed_dict)
        






