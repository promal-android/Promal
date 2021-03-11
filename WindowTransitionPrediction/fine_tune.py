#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import io
import json
import numpy as np
import os

from keras import backend as K
from keras.engine.topology import Layer
from keras.models import load_model

from sklearn.metrics import classification_report


GPU_ID = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

parser = argparse.ArgumentParser(description='ProMal fine-tuning')
parser.add_argument('--data_dir', type=str, required=True,
    help='path to pre-processed training data for fine-tuning')
parser.add_argument('--model_dir', type=str, required=True,
    help='path to the pre-trained model')
parser.add_argument('--output_dir', type=str, required=True,
    help='path to directory to save the pre-trained model')
args = parser.parse_args()

class ntn_layer(Layer):
    def __init__(self, inp_size, out_size, **kwargs):
        super(ntn_layer, self).__init__(**kwargs)
        self.k = out_size
        self.d = inp_size

    def build(self, input_shape):
        self.W = self.add_weight(name='w', shape=(self.k, self.d, self.d),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.V = self.add_weight(name='v', shape=(self.d * 2, self.k),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.G = self.add_weight(name='g', shape=(self.d * 2, self.k),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b1 = self.add_weight(name='b', shape=(self.k,),
                                  initializer='zeros',
                                  trainable=True)
        self.b2 = self.add_weight(name='b', shape=(self.k,),
                                  initializer='zeros',
                                  trainable=True)

        super(ntn_layer, self).build(input_shape)

    def call(self, x, mask=None):
        e1, e2 = x[0], x[1]
        forward_product = K.dot(K.concatenate([e1, e2]), self.V)
        tensor_products = [K.reshape(K.sum(K.dot(e1, self.W[i]) * e2, axis=1), [-1, 1]) for i in range(self.k)]
        match_product = K.dot(K.concatenate([K.abs(e2 - e1), e2 * e1]), self.G)
        return K.concatenate([K.tanh(match_product + self.b1),
                              K.tanh(K.concatenate(tensor_products, axis=1) + forward_product + self.b2)])

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 2 * self.k)

    def get_config(self):
        config = super(ntn_layer, self).get_config()
        config['out_size'] = self.k
        config['inp_size'] = self.d
        return config

siamese_model = load_model(args.model_dir + "pretrained_model.h5", {'ntn_layer': ntn_layer})

for layer in siamese_model.layers[:24]:
    layer.trainable = False
for layer in siamese_model.layers[24:]:
    layer.trainable = True
siamese_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
cw = {0: 0.2, 1: 0.8}
activity_list = list(np.load(args.data_dir + "/activity_np.npy"))
widget_list = list(np.load(args.data_dir + "/widget_np.npy"))
all_positive_pairs = np.load(args.data_dir + "/positive_pairs.npy")
img_np = np.load(args.data_dir + "/image_np.npy")
text_np = np.load(args.data_dir + "/text_np.npy")
uitree_np = np.load(args.data_dir + "/uitree_np.npy")
other_np = np.load(args.data_dir + "/other_np.npy")
w_image_np = np.load(args.data_dir + "/w_image_np.npy")
w_text_np = np.load(args.data_dir + "/w_text_np_tr.npy")
w_uitree_np = np.load(args.data_dir + "/w_uitree_np_tr.npy")
w_other_np = np.load(args.data_dir + "/w_other_np_tr.npy")

def has_link(i, j):
    for eachpair in all_positive_pairs:
        if eachpair[0] == i and eachpair[1] == j:
            return True
    return False

def widget2activity(wid, test=False):
    widget = widget_list[wid]
    activity = widget.split(": ")[0]
    return activity_list.index(activity)

print("start fine tuning")

for epoch in range(10):
    positive_pairs = np.load(args.data_dir + "positive_pairs_tr" + str(epoch) + ".npy")
    positive_pairs_t = np.load(args.data_dir + "positive_pairs_t" + str(epoch) + ".npy")

    widget_id_tr = list(np.load(args.data_dir + "widget_id_tr" + str(epoch) + ".npy"))
    activity_id_tr = list(np.load(args.data_dir + "activity_id_tr" + str(epoch) + ".npy"))
    widget_id_t = list(np.load(args.data_dir + "widget_id_t" + str(epoch) + ".npy"))
    activity_id_t = list(np.load(args.data_dir + "activity_id_t" + str(epoch) + ".npy"))

    negative_size = len(positive_pairs) * 10
    negative_size_t = len(positive_pairs_t) * 10
    random_widgets = np.random.randint(len(widget_id_tr), size=negative_size)
    random_widgets = np.array([widget_id_tr[wid] for wid in random_widgets])
    random_activities = np.random.randint(len(activity_id_tr), size=negative_size)
    random_activities = np.array([activity_id_tr[aid] for aid in random_activities])
    random_widgets = random_widgets.reshape((negative_size, 1))
    random_activities = random_activities.reshape((negative_size, 1))
    negative_pairs = np.append(random_widgets, random_activities, axis=1)
    negative_pairs = np.array([p for p in negative_pairs if not has_link(p[0], p[1])])

    labels = np.concatenate((np.zeros(len(negative_pairs)), np.ones(len(positive_pairs))))
    pair = np.concatenate((negative_pairs, positive_pairs))
    del negative_pairs
    shuffled_idx = np.arange(len(labels))
    np.random.shuffle(shuffled_idx)
    shuffled_pair = pair[shuffled_idx]
    shuffled_labels = labels[shuffled_idx]
    del pair, labels

    source_activities = np.array([widget2activity(p, False) for p in shuffled_pair[:, 0]])
    siamese_model.fit({'image_input1': img_np[source_activities[:]],
        'text_input1': text_np[source_activities[:]],
        'uitree_input1': uitree_np[source_activities[:]],
        'uitree_other1': other_np[source_activities[:]],
        'image_input2': img_np[shuffled_pair[:, 1]],
        'text_input2': text_np[shuffled_pair[:, 1]],
        'uitree_input2': uitree_np[shuffled_pair[:, 1]],
        'uitree_other2': other_np[shuffled_pair[:, 1]],
        'image_input1_0': w_image_np[shuffled_pair[:, 0]],
        'tag_input1_0': w_uitree_np[shuffled_pair[:, 0]],
        'uitree_other1_0': w_other_np[shuffled_pair[:, 0]],
        'text_input1_0': w_text_np[shuffled_pair[:, 0]]}, shuffled_labels, class_weight=cw)

    random_widgets_t = np.random.randint(len(widget_id_t), size=negative_size_t)
    random_widgets_t = np.array([widget_id_t[wid] for wid in random_widgets_t])
    random_activities_t = np.random.randint(len(activity_id_t), size=negative_size_t)
    random_activities_t = np.array([activity_id_t[aid] for aid in random_activities_t])
    random_widgets_t = random_widgets_t.reshape((negative_size_t, 1))
    random_activities_t = random_activities_t.reshape((negative_size_t, 1))
    negative_pairs_t = np.append(random_widgets_t, random_activities_t, axis=1)
    negative_pairs_t = np.array([p for p in negative_pairs_t if not has_link(p[0], p[1])])

    labels_t = np.concatenate((np.zeros(len(negative_pairs_t)), np.ones(len(positive_pairs_t))))
    pair_t = np.concatenate((negative_pairs_t, positive_pairs_t))
    del negative_pairs_t
    shuffled_idx_t = np.arange(len(labels_t))
    np.random.shuffle(shuffled_idx_t)
    shuffled_pair_t = pair_t[shuffled_idx_t]
    shuffled_labels_t = labels_t[shuffled_idx_t]
    del pair_t, labels_t

    source_activities = np.array([widget2activity(p, True) for p in shuffled_pair_t[:, 0]])
    pred1 = siamese_model.predict({'image_input1': img_np[source_activities[:]],
        'text_input1': text_np[source_activities[:]],
        'uitree_input1': uitree_np[source_activities[:]],
        'uitree_other1': other_np[source_activities[:]],
        'image_input2': img_np[shuffled_pair_t[:, 1]],
        'text_input2': text_np[shuffled_pair_t[:, 1]],
        'uitree_input2': uitree_np[shuffled_pair_t[:, 1]],
        'uitree_other2': other_np[shuffled_pair_t[:, 1]],
        'image_input1_0': w_image_np[shuffled_pair_t[:, 0]],
        'tag_input1_0': w_uitree_np[shuffled_pair_t[:, 0]],
        'uitree_other1_0': w_other_np[shuffled_pair_t[:, 0]],
        'text_input1_0': w_text_np[shuffled_pair_t[:, 0]]}, batch_size=32)
    p = []
    for each in pred1:
        if each >= 0.5:
            p.append(1)
        else:
            p.append(0)
    p = np.array(p)
    print(classification_report(shuffled_labels_t, p, digits=4))
    del p, pred1
    del shuffled_pair, shuffled_labels

    siamese_model.save(args.output_dir + "finetuned_model_" + str(epoch) + ".h5")
    del positive_pairs, positive_pairs_t
