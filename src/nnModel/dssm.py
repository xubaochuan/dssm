#coding=utf-8
import os
import sys

import numpy as np
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf8')

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('embedding_dim', 200, 'the dim of the fusion of multi word2vec model')
flags.DEFINE_integer('sentence_length', 20, 'sentence max word')
flags.DEFINE_boolean('use_gpu', False, 'use gpu or not')
flags.DEFINE_string('model_version', 'v1', 'model version')
flags.DEFINE_boolean('fine_tune', False, 'enable word embedding fine tune')
flags.DEFINE_integer('vocab_size', 2278, 'the size of the vocab')

class Model(object):

    def __init__(self, sample = 11, sample_nums = 200, lr = 0.1, lrdf = 0.95, l1 = 1000, l2 = 240, l3 = 400, cks = [2,3,4], ckn = [128, 64, 64]):
        self.SAMPLE = sample
        self.BS = self.SAMPLE * sample_nums
        self.LEARNING_RATE = lr
        self.LEARNING_RATE_DECAY_FACTOR = lrdf

        self.L1_N = l1
        self.L2_N = l2
        self.L3_N = l3
        self.OUT_N = 2

        self.conv_kernel_size = cks
        self.conv_kernel_number = ckn

        if FLAGS.use_gpu:
            device_name = "/gpu:0"
        else:
            device_name = "/cpu:0"
        with tf.device(device_name):
            with tf.name_scope('input'):
                self.query = tf.placeholder(tf.int32, shape=[None, FLAGS.sentence_length], name='QueryData')
                self.doc = tf.placeholder(tf.int32, shape=[None, FLAGS.sentence_length], name="DocData")
                self.label = tf.placeholder(tf.float32, shape=[None, 2], name='Label')

            with tf.name_scope('w2v'):
                if FLAGS.fine_tune:
                    self.words = tf.Variable(self.__load_w2v('../../model/vec.txt', FLAGS.embedding_dim), dtype=tf.float32, name='words')
                else:
                    self.words = tf.Variable(tf.random_uniform([FLAGS.vocab_size, FLAGS.embedding_dim]), dtype=tf.float32, name='words')

                self.query_words = tf.nn.embedding_lookup(self.words, self.query)
                self.doc_words = tf.nn.embedding_lookup(self.words, self.doc)

                self.query_words_out = tf.expand_dims(self.query_words, -1)
                self.doc_words_out = tf.expand_dims(self.doc_words, -1)

            with tf.name_scope('convolution_layer'):
                self.wc = {}
                self.bc = {}
                self.query_conv = {}
                self.query_pool = {}
                self.query_pool_list = []
                self.doc_conv = {}
                self.doc_pool = {}
                self.doc_pool_list = []
                for i, size in enumerate(self.conv_kernel_size):
                    # conv kernel size = i
                    self.wc[size] = tf.Variable(tf.random_normal([size, FLAGS.embedding_dim, 1, self.conv_kernel_number[i]]),
                                           'wc' + str(size))
                    self.bc[size] = tf.Variable(tf.random_normal([self.conv_kernel_number[i]]), 'bc' + str(size))

                    self. query_conv[size] = self.__conv2d('conv' + str(size), self.query_words_out, self.wc[size], self.bc[size])
                    self.query_pool[size] = self.__full_max_pool('pool' + str(size), self.query_conv[size], [0, 3, 2, 1])
                    self.query_pool_list.append(self.query_pool[size])

                    self.doc_conv[size] = self.__conv2d('conv' + str(size), self.doc_words_out, self.wc[size], self.bc[size])
                    self.doc_pool[size] = self.__full_max_pool('pool' + str(size), self.doc_conv[size], [0, 3, 2, 1])
                    self.doc_pool_list.append(self.doc_pool[size])

                self.query_pool_merge = tf.concat(self.query_pool_list, 3)
                self.query_conv_out = tf.nn.l2_normalize(tf.nn.relu(tf.reshape(self.query_pool_merge, [-1, sum(self.conv_kernel_number)])), 1)

                self.doc_pool_merge = tf.concat(self.doc_pool_list, 3)
                self.doc_conv_out = tf.nn.l2_normalize(tf.nn.relu(tf.reshape(self.doc_pool_merge, [-1, sum(self.conv_kernel_number)])),
                                                  1)

            with tf.name_scope('dense_layer_1'):
                self.l1_par_range = np.sqrt(6.0 / (sum(self.conv_kernel_number) + self.L1_N))
                self.wd1 = tf.Variable(tf.random_uniform([sum(self.conv_kernel_number), self.L1_N], -self.l1_par_range, self.l1_par_range))
                self.bd1 = tf.Variable(tf.random_uniform([self.L1_N], -self.l1_par_range, self.l1_par_range))

                self.query_l1 = tf.matmul(self.query_conv_out, self.wd1) + self.bd1
                self.doc_l1 = tf.matmul(self.doc_conv_out, self.wd1) + self.bd1

                self.query_l1_out = tf.nn.l2_normalize(tf.nn.relu(self.query_l1), 1)
                self.doc_l1_out = tf.nn.l2_normalize(tf.nn.relu(self.doc_l1), 1)

            with tf.name_scope('dense_layer_2'):
                self.l2_par_range = np.sqrt(6.0 / (self.L1_N + self.L2_N))
                self.wd2 = tf.Variable(tf.random_uniform([self.L1_N, self.L2_N], -self.l2_par_range, self.l2_par_range))
                self.bd2 = tf.Variable(tf.random_uniform([self.L2_N], -self.l2_par_range, self.l2_par_range))

                self.query_l2 = tf.matmul(self.query_l1_out, self.wd2) + self.bd2
                self.doc_l2 = tf.matmul(self.doc_l1_out, self.wd2) + self.bd2

                self.query_l2_out = tf.nn.l2_normalize(tf.nn.relu(self.query_l2), 1)
                self.doc_l2_out = tf.nn.l2_normalize(tf.nn.relu(self.doc_l2), 1)

            # with tf.name_scope('merge_query_doc'):
            #     self.pairwise = tf.concat([self.query_l2_out, self.doc_l2_out], axis=1)
            #
            # with tf.name_scope('hidden_layer'):
            #     self.hl1_par_range = np.sqrt(6.0 / (self.L2_N * 2 + self.L3_N))
            #     self.wh1 = tf.Variable(tf.random_uniform([self.L2_N * 2, self.L3_N], -self.hl1_par_range, self.hl1_par_range), 'wh1')
            #     self.bh1 = tf.Variable(tf.random_uniform([self.L3_N], -self.hl1_par_range, self.hl1_par_range), 'bh1')
            #
            #     self.hl = tf.matmul(self.pairwise, self.wh1) + self.bh1
            #     self.hl_out = tf.nn.l2_normalize(tf.nn.relu(self.hl), 1)
            #
            # with tf.name_scope('mlp_out'):
            #     self.out_par_range = np.sqrt(6.0 / (self.L3_N + self.OUT_N))
            #     self.wo1 = tf.Variable(tf.random_uniform([self.L3_N, self.OUT_N], -self.out_par_range, self.out_par_range), 'wo1')
            #     self.bo1 = tf.Variable(tf.random_uniform([self.OUT_N], -self.out_par_range, self.out_par_range), 'bo1')
            #
            #     self.out = tf.matmul(self.hl_out, self.wo1) + self.bo1

            with tf.name_scope('cosine_similarity'):
                # Cosine similarity
                # query_norm = tf.sqrt(tf.reduce_sum(tf.square(self.query_l2_out), 1, True))
                # doc_norm = tf.sqrt(tf.reduce_sum(tf.square(self.doc_l2_out), 1, True))
                #
                # prod = tf.reduce_sum(tf.multiply(self.query_l2_out, self.doc_l2_out), 1, True)
                # norm_prod = tf.multiply(query_norm, doc_norm) + 0.01
                # cos_sim = tf.truediv(prod, norm_prod)
                prod = tf.reduce_sum(tf.multiply(self.query_l2_out, self.doc_l2_out), 1, True)
                unprod = tf.abs(1 - prod)
                self.out = tf.concat([unprod, prod], axis=1)

            with tf.name_scope('loss'):
                self.pred_y = tf.argmax(tf.nn.softmax(self.out), 1)
                self.label_y = tf.argmax(self.label, 1)
                self.pred = tf.equal(self.pred_y, self.label_y)
                self.accuracy = tf.reduce_mean(tf.cast(self.pred, tf.float32))
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.label))

            with tf.name_scope('train'):
                self.learning_rate = tf.Variable(float(self.LEARNING_RATE), trainable=False)
                self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * self.LEARNING_RATE_DECAY_FACTOR)
                #        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            self.merged = tf.summary.merge_all()

    def __load_w2v(self, path, expectDim):
        fp = open(path, "r")
        line = fp.readline().strip()
        ss = line.split(" ")
        total = int(ss[0])
        dim = int(ss[1])
        assert (dim == expectDim)
        ws = []
        for t in range(total):
            line = fp.readline().rstrip()
            ss = line.split(" ")[1:]
            assert (len(ss) == dim)
            vals = []
            for i in range(0, dim):
                fv = float(ss[i])
                vals.append(fv)
            ws.append(vals)
        fp.close()
        assert total == len(ws)
        print "w2v size : " + str(total)
        return np.asarray(ws, dtype=np.float32)

    def __conv2d(self, name, input, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='VALID'), b), name=name)

    def __full_max_pool(self, name, input, perm):
        conv1 = tf.transpose(input, perm=perm)
        values = tf.nn.top_k(conv1, 1, name=name).values
        conv2 = tf.transpose(values, perm=perm)
        return conv2

    def __norm(self, name, input, lsize=4):
        return tf.nn.local_response_normalization(input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

    def load_model(self, saver, sess, model_path):
        if os.path.exists(model_path + '.index'):
            saver.restore(sess, model_path)
        else:
            sess.run(tf.global_variables_initializer())

    def save_model(self, saver, sess, model_path):
        saver.save(sess, model_path)
