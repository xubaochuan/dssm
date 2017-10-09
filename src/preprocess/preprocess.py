#coding=utf-8
import random

train_query_path = '../../dataset/train_query.txt'
train_doc_path = '../../dataset/train_doc.txt'
train_label_path = '../../dataset/train_label.txt'
valid_query_path = '../../dataset/valid_query.txt'
valid_doc_path = '../../dataset/valid_doc.txt'
valid_label_path = '../../dataset/valid_label.txt'
test_query_path = '../../dataset/test_query.txt'
test_doc_path = '../../dataset/test_doc.txt'
test_label_path = '../../dataset/test_label.txt'

def replaceTag(s):
    s = s.replace(',', '')
    s = s.replace('.', '')
    return s.lower()

def getDataPairs(filepath):
    fr = open(filepath)
    first_line = fr.readline()
    stcA = []
    stcB = []
    all_lines = []
    for line in fr.readlines():
        all_lines.append(line)
    random.shuffle(all_lines)
    fr.close()
    for line in all_lines:
        arr = line.strip().split('\t')
        if len(arr) != 12:
            continue
        if float(arr[4]) < 3: #去掉关联低的文本对
            continue
        stcA.append(replaceTag(arr[1]))
        stcB.append(replaceTag(arr[2]))
    assert len(stcA) == len(stcB)
    return stcA, stcB

def generate_vocab_small(stcA, stcB):
    all_stc = stcA + stcB
    vocab_set = set()
    for stc in all_stc:
        arr = stc.strip().split(' ')
        for v in arr:
            if v not in vocab_set:
                vocab_set.add(v)
    fw = open('../../model/vocab.txt', 'w')
    print "vocab size: " + str(len(vocab_set))
    for v in vocab_set:
        fw.write(v + '\n')
    fw.close()

def generate_vocab():
    fr = open('../../model/vec.txt')
    vocab = {}
    first_line = fr.readline()
    f_arr = first_line.strip().split(' ')
    size = int(f_arr[0])
    dim = int(f_arr[1])
    index = 0
    fw = open('../../model/vocab.txt', 'w')
    for line in fr.readlines():
        arr = line.rstrip().split(' ')
        if len(arr) != dim+1:
            print index,arr
            break
        vocab[arr[0]] = index
        index += 1
        fw.write(arr[0] + '\n')
    assert size == len(vocab)
    print "vocab size: " + str(size)
    fr.close()
    fw.close()

def generate_w2v(stcA, stcB):
    all_stc = stcA + stcB
    fw = open('../../raw_data/w2v_corpus.txt', 'w')
    for stc in all_stc:
        fw.write(stc + '\n')
    fw.close()

def generate_dataset(stcA, stcB, n_neg=10, former_rate=0.8, latter_rate = 0.9):
    query = stcA
    doc = stcB

    assert len(query) == len(doc)

    l = len(query)

    query_data = []
    doc_data = []
    label = []

    for i in range(l):
        query_data.append(query[i])
        doc_data.append(doc[i])
        label.append([0,1])
        for _ in range(n_neg):
            idx = random.randint(0, l-1)
            query_data.append(query[i])
            doc_data.append(doc[idx])
            label.append([1,0])

    sample_number = len(query_data)

    former = int(sample_number*former_rate)
    latter = int(sample_number*latter_rate)

    train_query = query_data[:former]
    train_doc = doc_data[:former]
    train_label = label[:former]

    valid_query = query_data[former:latter]
    valid_doc = doc_data[former:latter]
    valid_label = label[former:latter]

    test_query = query_data[latter:]
    test_doc = doc_data[latter:]
    test_label = label[latter:]

    write_file(train_query, train_query_path)
    write_file(train_doc, train_doc_path)
    write_file(train_label, train_label_path, True)

    write_file(valid_query, valid_query_path)
    write_file(valid_doc, valid_doc_path)
    write_file(valid_label, valid_label_path, True)

    write_file(test_query, test_query_path)
    write_file(test_doc, test_doc_path)
    write_file(test_label, test_label_path, True)

def write_file(content, filepath, label=False):
    fw = open(filepath, 'w')
    if label:
        for row in content:
            fw.write(' '.join([str(i) for i in row]) + '\n')
    else:
        for row in content:
            fw.write(row + '\n')
    fw.close()

def main():
    filepath = '../../raw_data/SICK.txt'
    stcA, stcB = getDataPairs(filepath)
    generate_w2v(stcA, stcB)
    generate_vocab_small(stcA, stcB)
    generate_dataset(stcA, stcB, n_neg=2)

if __name__=='__main__':
    main()