import tensorflow as tf
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

        # set default graph and start session
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.sess = tf.Session(config=config)

    def build(self, batch):

        print('--- Building computational graph')

        # input placeholders
        self.X_in = tf.placeholder(tf.float32, shape=(None, batch['X'].shape[1]), name='X_in')
        self.Y_in = tf.placeholder(tf.int32,   shape=(None, batch['Y'].shape[1]), name='Y_in')
        self.Z_in = tf.placeholder(tf.float32, shape=(None, batch['Z'].shape[1]), name='Z_in')
        self.W_in = tf.placeholder(tf.float32, shape=(None, batch['W'].shape[1]), name='W_in')

        # classifier output and loss
        _, _ = self.clf.forward(self.X_in)
        _ = self.clf.loss(self.Y_in)

        # adversary output and loss
        # TODO

        # optimisers
        opt_C = tf.train.AdamOptimizer(**self.opt_conf).minimize(self.clf.loss, var_list=self.clf.tf_vars)

    def initialise_variables(self):
        print('--- Initialising TensorFlow variables')
        self.sess.run(tf.global_variables_initializer())

    def pretrain(self, s):

        pass

    def forward(self): #TODO
        
        pass

