#coding=utf-8
import sys

import tensorflow as tf
sys.path.append("..")
from dataProvider import data_provider
import nnModel.dssm as dssm

reload(sys)
sys.setdefaultencoding('utf8')

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir', '/tmp/dssm-v1', 'Summaries directory')
flags.DEFINE_string('model_path', '../../model/dssm.ckpt', 'model path')
flags.DEFINE_integer('max_epoch', 50000, 'max train steps')

train_query, train_doc, train_label = data_provider.load_train_dataset()
valid_query, valid_doc, valid_label = data_provider.load_valid_dataset()
test_query, test_doc, test_label = data_provider.load_test_dataset()

def biclass_rate(label, pred):
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    for i in range(label.shape[0]):
        if label[i] == 1:
            if pred[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred[i] == 0:
                tn += 1
            else:
                fp += 1
    p_precision = tp / (tp + fp + 0.0001)
    p_recall = tp / (tp + fn + 0.0001)
    n_precision = tn / (tn + fn + 0.0001)
    n_recall = tn / (tn + fp + 0.0001)

    return p_precision, p_recall, n_precision, n_recall

def get_train_batch_data(step, BS):
    global train_query, train_doc, train_label
    start = step * BS
    end = (step + 1) * BS
    return train_query[start:end, :], train_doc[start:end, :], train_label[start:end, :]

def train():
    BS = (10+1)*200

    dssmModel = dssm.Model()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        dssmModel.load_model(saver, sess, model_path = FLAGS.model_path)

        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        max_loss = float('INF')
        epoch_steps = train_query.shape[0] / BS
        previous_losses = []
        for epoch in range(FLAGS.max_epoch):
            for step in range(epoch_steps):
                query_batch, doc_batch, label_batch = get_train_batch_data(step, BS)
                sess.run(dssmModel.train_step, feed_dict={dssmModel.query: query_batch, dssmModel.doc: doc_batch, dssmModel.label: label_batch})
                mgd, ls, acc, pred_, label_, l_rate = sess.run([dssmModel.merged, dssmModel.loss, dssmModel.accuracy, dssmModel.pred_y, dssmModel.label_y, dssmModel.learning_rate], feed_dict={dssmModel.query: query_batch, dssmModel.doc: doc_batch, dssmModel.label: label_batch})
                p_p, p_r, n_p, n_r = biclass_rate(label_, pred_)
                train_writer.add_summary(mgd, epoch * epoch_steps + step)
                print('Train Epoch %d, Step %d, l_rate: %f, loss: %f, accuracy: %f, p_precision: %f, p_recall: %f, n_precision: %f, n_recall: %f' % (epoch + 1, step + 1, l_rate, ls, acc, p_p, p_r, n_p, n_r))
                sys.stdout.flush()

                if step % 5 == 0:
                    ls, acc, pred_, label_ = sess.run([dssmModel.loss, dssmModel.accuracy, dssmModel.pred_y, dssmModel.label_y], feed_dict={dssmModel.query: valid_query, dssmModel.doc: valid_doc, dssmModel.label: valid_label})
                    p_p, p_r, n_p, n_r = biclass_rate(label_, pred_)
                    if len(previous_losses) >= 5 and ls > max(previous_losses[-5:]):
                        sess.run(dssmModel.learning_rate_decay_op)
                        print("Learning rate decay Epoch %d, Step %d , learning rate %f" % (epoch + 1, step + 1, dssmModel.learning_rate.eval()))
                    previous_losses.append(ls)
                    print('Valid Epoch %d, loss: %f, accuracy: %f, p_precision: %f, p_recall: %f, n_precision: %f, n_recall: %f' % (epoch + 1, ls, acc, p_p, p_r, n_p, n_r))
                    sys.stdout.flush()
                    if ls < max_loss:
                        dssmModel.save_model(saver, sess, FLAGS.model_path)

            ls, acc, pred_, label_ = sess.run([dssmModel.loss, dssmModel.accuracy, dssmModel.pred_y, dssmModel.label_y], feed_dict={dssmModel.query: test_query, dssmModel.doc: test_doc, dssmModel.label: test_label})
            p_p, p_r, n_p, n_r = biclass_rate(label_, pred_)
            print "--------------------------------------------"
            print('Test Epoch %d, loss: %f, accuracy: %f, p_precision: %f, p_recall: %f, n_precision: %f, n_recall: %f' % (epoch + 1, ls, acc, p_p, p_r, n_p, n_r))
            print "--------------------------------------------"
            sys.stdout.flush()

if __name__=='__main__':
    train()
