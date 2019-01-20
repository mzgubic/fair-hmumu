import tensorflow as tf
from fair_hmumu import defs
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

        # build classifier output and loss
        self.clf.make_forward_pass(self._in['X'])
        self.clf.make_loss(self._in['Y'])

        # adversary loss
        self.adv.make_loss(self.clf.proba, self._in['Z'])

        # combined loss
        self.CA_loss = self.clf.loss - self.opt_conf['lambda'] * self.adv.loss

        # optimisers
        adam_hps = {key:self.opt_conf[key] for key in self.opt_conf if key not in ['lambda']}
        self.opt_C = tf.train.AdamOptimizer(**adam_hps).minimize(self.clf.loss, var_list=self.clf.tf_vars)
        self.opt_A = tf.train.AdamOptimizer(**adam_hps).minimize(self.adv.loss, var_list=self.adv.tf_vars)
        self.opt_CA = tf.train.AdamOptimizer(**adam_hps).minimize(self.CA_loss, var_list=self.clf.tf_vars)

    def initialise_variables(self):

        print('--- Initialising TensorFlow variables')

        self.sess.run(tf.global_variables_initializer())

    def pretrain_step(self, batch):

        feed_dict = {self._in[xyzw]:batch[xyzw] for xyzw in defs.XYZW}
        self.sess.run(self.opt_C, feed_dict=feed_dict)

    def train_step_clf(self, batch):

        feed_dict = {self._in[xyzw]:batch[xyzw] for xyzw in defs.XYZW}
        self.sess.run(self.opt_CA, feed_dict=feed_dict)

    def train_step_adv(self, batch):

        feed_dict = {self._in[xyzw]:batch[xyzw] for xyzw in defs.XYZW}
        self.sess.run(self.opt_A, feed_dict=feed_dict)

    def losses(self, data):

        feed_dict = {self._in[xyzw]:data[xyzw] for xyzw in defs.XYZW}
        return self.sess.run([self.clf.loss, self.adv.loss, self.CA_loss], feed_dict=feed_dict)

    def clf_predict(self, data):

        feed_dict = {self._in[xyzw]:data[xyzw] for xyzw in defs.XYZW}
        return self.sess.run(self.clf.proba, feed_dict=feed_dict)







