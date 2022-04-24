#!/usr/bin/env python
# -*- coding:utf-8 -*-
# datetime:2022/4/22 9:52

import os
import pandas as pd
import numpy as np

np.random.seed(1337)

from sklearn.model_selection import KFold
from sklearn import svm
import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.models import Sequential
from keras.engine.topology import Layer

# 二十种不同的氨基酸
AminoAcids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
ChemicalFeatures = [("C", "M"), ("A", "G", "P"), ("I", "L", "V"), ("D", "E"),
                    ("H", "K", "R"), ("F", "W", "Y"), ("N", "Q"), ("S", "T")]
PosLabel = '1'  # 可切割标记为1
NegLabel = '-1'  # 不可切割标记为-1


class Attention(Layer):
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        init_val_v = (np.random.randn(input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_v = K.variable(init_val_v, name='att_v')
        init_val_W = (np.random.randn(input_shape[2], input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_W = K.variable(init_val_W, name='att_W')
        self.trainable_weights = [self.att_v, self.att_W]

    def call(self, x, mask=None):
        y = K.dot(x, self.att_W)
        weights = tf.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
        weights = K.softmax(weights)
        out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
        return K.sum(out, axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def __seq_num(amino_data):
    orthogonal_dict = dict()
    for i, amino in enumerate(AminoAcids):
        orthogonal_dict[amino] = i
    train_xdata = []
    train_ydata = []
    train = []
    for i in range(len(amino_data)):
        feature = []
        for x in amino_data["Amino"][i]:
            feature.append(orthogonal_dict[x])
        label = amino_data['Label'][i]
        if label == -1:
            label = 0
        train_xdata.append(feature)
        train_ydata.append(label)
    train.append(train_xdata)
    train.append(train_ydata)
    return train


def __extract_feature(tp, tn, embedding_size=-1, filters1=-1, filters2=-1, kernel_size1=-1, kernel_size2=-1):
    (x_p, y_p), (x_n, y_n) = tp, tn
    x = np.append(x_p, x_n, axis=0)
    inx = len(x_p)
    model = Sequential()
    model.add(layers.Embedding(20, embedding_size, input_length=8))
    model.add(layers.Conv1D(filters=filters1, kernel_size=kernel_size1, activation="tanh", padding="same"))
    model.add(layers.Conv1D(filters=filters2, kernel_size=kernel_size2, activation="tanh", padding="same"))
    model.add(Attention())
    model.compile('rmsprop', 'mse')
    output_array = model.predict(x)
    x_p, x_n = output_array[:inx], output_array[inx:]
    return [x_p, y_p], [x_n, y_n]


def __train_test_model(c1, beta, input_file, train_data, test_data, test_seq, cv=-1):
    print("AttentionSVM is running")
    try:
        if cv == -1:
            if not os.path.exists("{}/results".format(input_file)):
                os.makedirs("{}/results".format(input_file))
            f = open('{}/results/result.txt'.format(input_file), 'w+', encoding='utf-8')
        else:
            if not os.path.exists("{}/{}/results".format(input_file, str(cv))):
                os.makedirs("{}/{}/results".format(input_file, str(cv)))
            f = open('{}/{}/results/result.txt'.format(input_file, str(cv)), 'w+', encoding='utf-8')
    except Exception as e:
        print(e)
        exit()

    x_train, y_train = train_data
    x_test, y_test = test_data
    y_test = list(y_test)

    clf = svm.SVC(kernel='linear', probability=True, random_state=42, max_iter=2000,
                  class_weight={0: c1 / beta, 1: c1})
    clf.fit(x_train, y_train)
    f.write("Amino label 1\n")
    for inx, s in enumerate(clf.predict_proba(x_test)):
        f.write("{} {} {}\n".format(test_seq["Amino"][inx], 1 if y_test[inx] == 1 else -1, s[1]))
    f.close()
    print("The result of AttentionSVM method is saved")


def run(e, f1, f2, k1, k2, c1, beta, input_file, cv):
    e, f1, f2, k1, k2 = map(eval, [e, f1, f2, k1, k2])
    c1, beta, cv = map(eval, [c1, beta, cv])

    if c1 <= 0 or beta <= 0:
        print(" c1>=0 and beta>0")
        exit()

    remain1 = 9 - k1
    remain2 = remain1 + 1 - k2
    if remain2 < 2:
        print("The input depth of Attention layer must more than 2")
        exit()

    if cv == -1:
        try:
            frame_train_p = pd.read_table('{}/train/pos'.format(input_file), sep=',',
                                          names=['Amino', 'Label'])
            frame_train_n = pd.read_table('{}/train/neg'.format(input_file), sep=',',
                                          names=['Amino', 'Label'])
            frame_train = pd.concat([frame_train_p, frame_train_n], axis=0)
            frame_train.index = range(len(frame_train))

            frame_test_p = pd.read_table('{}/test/pos'.format(input_file), sep=',',
                                         names=['Amino', 'Label'])
            frame_test_n = pd.read_table('{}/test/neg'.format(input_file), sep=',',
                                         names=['Amino', 'Label'])
            frame_test = pd.concat([frame_test_p, frame_test_n], axis=0)
            frame_test.index = range(len(frame_test))

            train_seq, test_seq = __seq_num(frame_train), __seq_num(frame_test)

            train_data, test_data = __extract_feature(train_seq, test_seq, e, f1, f2, k1, k2)

            __train_test_model(c1, beta, input_file, train_data, test_data, frame_test)
        except Exception as e:
            print(e)
            exit()
    else:
        #  cv-fold cross-validation
        try:
            frame_train_p = pd.read_table('{}/train/pos'.format(input_file), sep=',',
                                          names=['Amino', 'Label'])
            frame_train_n = pd.read_table('{}/train/neg'.format(input_file), sep=',',
                                          names=['Amino', 'Label'])
            frame_train = pd.concat([frame_train_p, frame_train_n], axis=0)
            frame_train.index = range(len(frame_train))

            train_pos_seq, train_neg_seq = __seq_num(frame_train_p), __seq_num(frame_train_n)

            train_pos_data, train_neg_data = __extract_feature(train_pos_seq, train_neg_seq, e, f1, f2, k1, k2)

            kf = KFold(n_splits=cv, shuffle=False)

            xp = np.array(train_pos_data[0])
            yp = np.array(train_pos_data[1])
            xp_train_index = []
            xp_test_index = []
            for train_index, test_index in kf.split(xp):
                xp_train_index.append(train_index)
                xp_test_index.append(test_index)

            xn = np.array(train_neg_data[0])
            yn = np.array(train_neg_data[1])
            xn_train_index = []
            xn_test_index = []
            for train_index, test_index in kf.split(xn):
                xn_train_index.append(train_index)
                xn_test_index.append(test_index)

            for i in range(1, cv + 1):
                xp_train, yp_train, xp_test, yp_test = xp[xp_train_index[i - 1]], yp[xp_train_index[i - 1]], \
                                                       xp[xp_test_index[i - 1]], yp[xp_test_index[i - 1]]
                xn_train, yn_train, xn_test, yn_test = xn[xn_train_index[i - 1]], yn[xn_train_index[i - 1]], \
                                                       xn[xn_test_index[i - 1]], yn[xn_test_index[i - 1]]
                x_train = np.append(xp_train, xn_train, axis=0)
                y_train = np.append(yp_train, yn_train, axis=0)
                x_test = np.append(xp_test, xn_test, axis=0)
                y_test = np.append(yp_test, yn_test, axis=0)
                test_seq = pd.concat([frame_train_p.iloc[xp_test_index[i - 1]],
                                      frame_train_n.iloc[xn_test_index[i - 1]]], axis=0, ignore_index=True)
                __train_test_model(c1, beta, input_file, (x_train, y_train), (x_test, y_test), test_seq, i)
        except Exception as e:
            print(e)
            exit()
