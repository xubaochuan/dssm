#coding=utf-8
import tensorflow as tf

from src.dataProvider import data_provider
from src.nnModel import dssm

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', '../model/dssm.ckpt', 'model path')

def biclass_rate(self, label, pred):
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

def predict(query_path, doc_path):
    query_data, doc_data = data_provider.load_dataset(query_path, doc_path)
    dssmModel = dssm.Model()
    with tf.Session() as sess:
        dssmModel.load_model(sess, FLAGS.model_path)
        pred_= sess.run([dssmModel.pred_y], feed_dict={dssmModel.query: query_data, dssmModel.doc: doc_data})
        return pred_

def write_result(query_path, doc_path, pred, output_path):
    query = load_file_data(query_path)
    doc = load_file_data(doc_path)
    pred = pred[0].tolist()
    assert len(query) == len(doc)
    assert len(query) == len(pred)
    fw = open(output_path, 'w')
    for i in range(len(pred)):
        row = query[i] + '\t' + doc[i] + '\t' + str(pred[i])
        fw.write(row + '\n')
    fw.close()

def write_diff(query_path, doc_path, pred, label, output_path):
    query = load_file_data(query_path)
    doc = load_file_data(doc_path)
    pred = pred[0].tolist()
    assert len(query) == len(doc)
    assert len(query) == len(pred)
    assert len(pred) == len(label)
    fw = open(output_path, 'w')
    for i in range(len(pred)):
        if pred[i] != label[i]:
            row = query[i] + '\t' + doc[i] + '\t' + str(pred[i]) + '\t' + str(label[i])
            fw.write(row + '\n')
    fw.close()

def load_file_data(filepath):
    fr = open(filepath)
    content = []
    for row in fr.readlines():
        row = row.rstrip()
        if row == '':
            continue
        content.append(row)
    fr.close()
    return content

def load_label_data(filepath):
    fr = open(filepath)
    label = []
    for row in fr.readlines():
        row = row.rstrip()
        if row == '':
            continue
        array = row.split(' ')
        if array[0] == '1':
            label.append(0)
        else:
            label.append(1)
    fr.close()
    return label

if __name__=='__main__':
    query_path = '../dataset/test_query.txt'
    doc_path = '../dataset/test_doc.txt'
    label_path = '../dataset/test_label.txt'
    pred = predict(query_path, doc_path)
    label = load_label_data(label_path)
    write_result(query_path, doc_path, pred, '../output/test1.txt')
    write_diff(query_path, doc_path, pred, label, '../output/diff1.txt')