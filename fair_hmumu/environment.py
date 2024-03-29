import tensorflow as tf
from fair_hmumu import defs
from fair_hmumu import models
from fair_hmumu.utils import Saveable

class TFEnvironment(Saveable):

    def __init__(self, clf, adv, opt_conf, config=tf.ConfigProto(intra_op_parallelism_threads = 32,
                                                                 inter_op_parallelism_threads = 32,
                                                                 allow_soft_placement = True,
                                                                 device_count = {'CPU': 2})):

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

        # optimiser arguments
        opt = getattr(tf.train, self.opt_conf['type']) # choose the optimizer, for example tf.train.AdamOptimizer
        opt_args = list(opt.__init__.__code__.co_varnames)
        opt_hps = {key:self.opt_conf[key] for key in self.opt_conf if key in opt_args}

        # learning rate decay
        global_step = tf.Variable(0, name='global_step', trainable=False)
        try:
            lr = tf.train.exponential_decay(self.opt_conf['learning_rate'],
                                            global_step=global_step,
                                            decay_steps=self.opt_conf['decay_steps'],
                                            decay_rate=self.opt_conf['decay_rate'])
            opt_hps['learning_rate'] = lr
        except KeyError:
            pass

        # make optimisers
        self.opt_C = opt(**opt_hps).minimize(self.clf.loss, var_list=self.clf.tf_vars, global_step=global_step)
        self.opt_A = opt(**opt_hps).minimize(self.adv.loss, var_list=self.adv.tf_vars)
        self.opt_CA = opt(**opt_hps).minimize(self.CA_loss, var_list=self.clf.tf_vars, global_step=global_step)

    def initialise_variables(self):

        print('--- Initialising TensorFlow variables')

        self.sess.run(tf.global_variables_initializer())

    def pretrain_step(self, batch):

        feed_dict = {self._in[xyzw]:batch[xyzw] for xyzw in defs.XYZW}
        self.sess.run(self.opt_C, feed_dict=feed_dict)

    def train_step_clf(self, batch):

        # only use classifier loss of there is no adversary
        if isinstance(self.adv, models.DummyAdversary):
            opt = self.opt_C
        else:
            opt = self.opt_CA
        
        feed_dict = {self._in[xyzw]:batch[xyzw] for xyzw in defs.XYZW}
        self.sess.run(self.opt_CA, feed_dict=feed_dict)

    def train_step_adv(self, batch):

        # do not run if dummy adversary
        if isinstance(self.adv, models.DummyAdversary):
            return None

        feed_dict = {self._in[xyzw]:batch[xyzw] for xyzw in defs.XYZW}
        self.sess.run(self.opt_A, feed_dict=feed_dict)

    def losses(self, data):

        feed_dict = {self._in[xyzw]:data[xyzw] for xyzw in defs.XYZW}
        return self.sess.run([self.clf.loss, self.adv.loss, self.CA_loss], feed_dict=feed_dict)

    def clf_predict(self, data):

        feed_dict = {self._in[xyzw]:data[xyzw] for xyzw in defs.XYZW}
        return self.sess.run(self.clf.proba, feed_dict=feed_dict)

    def save_model(self, path):

        print('--- Saving the classifier as {}'.format(path))
        saver = tf.train.Saver(self.clf.tf_vars)
        saver.save(self.sess, path)

    def load_model(self, path):

        print('--- Restoring the classifier from {}'.format(path))
        saver = tf.train.Saver(self.clf.tf_vars)
        saver.restore(self.sess, path)







