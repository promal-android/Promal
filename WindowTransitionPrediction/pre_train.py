#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import gensim
import io
import jieba
import json
import numpy as np
import os
from PIL import Image

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import concatenate
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Masking
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.models import Model

from sklearn.metrics import classification_report


GPU_ID = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

parser = argparse.ArgumentParser(description='ProMal pre-training')
parser.add_argument('--data_dir', type=str, required=True,
    help='path to pre-processed training data for pre-training')
parser.add_argument('--paladin_dir', type=str, required=True,
    help='path to paladin result of apps for pre-training')
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


# tag dic: map widget tags to one-hot embeddings
all_tags_dict = json.load(open(args.data_dir + "./all_tags_dict.json", 'r'))
threshold = 0.5

model = gensim.models.Word2Vec.load(args.data_dir + "word2vec_wx")
def gettextvector(text):
    word = np.zeros(256, 'float32')
    wordlist = jieba.cut(text)
    countt = 0
    for eachword in wordlist:
        countt += 1
        try:
            word0 = model[eachword]
            word = word + word0
        except:
            countt -= 1
    if countt != 0:
        word = word / countt
    return word


def dfs(root):
    if root.get("children") and len(root["children"]) > 0:
        a, b, c, d, e, f = dfs(root["children"][0])
        if root.get("viewText"):
            f = root["viewText"] + " " + f
        return a, b, c, d, e, f
    tag = root["viewTag"]
    tag = bytes.decode(tag.split('.')[-1].encode('utf-8'))
    return root["x"], root["y"], root["width"], root["height"], tag, root['viewText']


def crop_image(img, x, y, width, height):
    try:
        im = Image.open(img).convert('RGB')
        new_img = im.resize((90, 160), Image.BILINEAR)
        data = np.array(new_img)
        for i in range(160):
            for j in range(90):
                if i < x // 16 or i >= (x + width) // 16:
                    if j < y // 16 or j >= (y + height) // 16:
                        data[i][j][0] = data[i][j][1] = data[i][j][2] = 0
        return data
    except IOError:
        a = np.zeros([160, 90, 3], dtype='float32')
        return a


# Define Input layers (Screen snapshot, text, UI layout trees
image_input = Input(shape=(160, 90, 3), dtype='float32', name='image_input')
text_input = Input(shape=(256,), dtype = 'float32', name = 'text_input')
uitree_input = Input(shape=(256, ), dtype='float32', name='uitree_input')
uitree_other = Input(shape=(128, 4), dtype='float32', name='uitree_other')

# source page feature
image_input1 = Input(shape=(160, 90, 3), dtype='float32', name='image_input1')
text_input1 = Input(shape=(256,), dtype = 'float32', name = 'text_input1')
uitree_input1 = Input(shape=(256, ), dtype='float32', name='uitree_input1')
uitree_other1 = Input(shape=(128, 4), dtype='float32', name='uitree_other1')
# source widget feature
image_input1_0 = Input(shape=(160, 90, 3), dtype='float32', name='image_input1_0')
text_input1_0 = Input(shape=(256,), dtype = 'float32', name = 'text_input1_0')
tag_input1_0 = Input(shape=(1, ), dtype='float32', name='tag_input1_0')
uitree_other1_0 = Input(shape=(4,), dtype='float32', name='uitree_other1_0')
# target page feature
image_input2 = Input(shape=(160, 90, 3), dtype='float32', name='image_input2')
text_input2 = Input(shape=(256,), dtype = 'float32', name = 'text_input2')
uitree_input2 = Input(shape=(256, ), dtype='float32', name='uitree_input2')
uitree_other2 = Input(shape=(128, 4), dtype='float32', name='uitree_other2')

neighbor_input1 = Input(shape=(64, ), dtype='float32', name='neighbor_input1')
neighbor_input2 = Input(shape=(64, ), dtype='float32', name='neighbor_input2')

page_list = []
f0 = open(page_list + "page_list.txt", 'r')
for eachline in f0:
    page_list.append(eachline[:-1])
f0.close()

def same_app(i, j):
    name1 = page_list[i].split("/")[0]
    name2 = page_list[j].split("/")[0]
    return name1 == name2

dense = Dense(32, activation='tanh')

X1 = Conv2D(batch_input_shape=(160, 90, 3), filters=64, kernel_size=5,
            strides=1, kernel_initializer='he_normal', padding='same', use_bias=False)(image_input)
X1 = BatchNormalization()(X1)
X1 = Activation('relu')(X1)
X1 = BatchNormalization()(X1)
X1 = Activation('relu')(X1)
X1 = GlobalAveragePooling2D()(X1)
X1 = Dense(64)(X1)

X2 = Dense(256, activation='sigmoid', input_shape=(512,))(text_input)
X2 = Dense(48, activation='sigmoid')(X2)

X3 = Embedding(5000, 20, mask_zero=True, input_length=256)(uitree_input)  # !
X3 = LSTM(72, dropout=0.2, recurrent_dropout=0.2)(X3) #!!!

X4 = BatchNormalization(input_shape=(128, 4))(uitree_other)
X4 = Masking(mask_value=0.0)(X4)
X4 = LSTM(72, dropout=0.2, recurrent_dropout=0.2)(X4)

x = concatenate([X1, X2, X3, X4])
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
x = Dense(64, activation='sigmoid')(x)

base_model = Model(inputs=[image_input, text_input,  uitree_input, uitree_other], output=x)

X0 = Conv2D(batch_input_shape=(160, 90, 3), filters=64, kernel_size=5,
            strides=1, kernel_initializer='he_normal', padding='same', use_bias=False)(image_input1_0)
X0 = BatchNormalization()(X0)
X0 = Activation('relu')(X0)
X0 = BatchNormalization()(X0)
X0 = Activation('relu')(X0)
X0 = GlobalAveragePooling2D()(X0)
X0 = Dense(64)(X0)
Xt = Dense(1)(tag_input1_0)
Xu = Dense(4)(uitree_other1_0)
Xtext = Dense(64)(text_input1_0)
X0 = concatenate([X0, Xt, Xu, Xtext])

a = base_model([image_input1, text_input1, uitree_input1, uitree_other1])
b = base_model([image_input2, text_input2, uitree_input2, uitree_other2])
a = concatenate([X0, a])
a = Dense(64)(a)
b = Dense(64)(b)
a = dense(a)
a = Dropout(0.1)(a)
a = BatchNormalization()(a)
b = dense(b)
b = Dropout(0.1)(b)
b = BatchNormalization()(b)
X = ntn_layer(inp_size=32, out_size=16)([a, b])
X = Dropout(0.5)(X)
X = Dense(1, activation='sigmoid')(X)

model = Model(inputs=[image_input1, text_input1, uitree_input1, uitree_other1, 
    image_input2, text_input2, uitree_input2, uitree_other2, image_input1_0, 
    tag_input1_0, uitree_other1_0, text_input1_0], output=X)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
cw = {0: 0.25, 1: 0.75}

adj = np.load(args.data_dir + "adj.npy")

image_np = np.load(args.data_dir + "image_np.npy")
text_np = np.load(args.data_dir + "text_np.npy")
uitree_np = np.load(args.data_dir + "uitree_np_2020.npy")
other_np = np.load(args.data_dir + "other_np.npy")

positive_pairs = np.load(args.data_dir + "positive_pairs.npy")

# widget numpys
widget_image_np = np.load(args.data_dir + "widget_image.npy")
widget_other_np = np.load(args.data_dir + "widget_other.npy")
widget_tag_np = np.load(args.data_dir + "widget_tag.npy")
widget_text_np = np.load(args.data_dir + "widget_text.npy")

split_index_new = 4 * len(widget_tag_np) // 5
positive_pair_tr = np.array(positive_pairs[:8 * len(positive_pairs) // 10])
positive_pair_test = np.array(positive_pairs[8 * len(positive_pairs) // 10:])


n = len(positive_pairs)
del positive_pairs
intraapp_rate = 0.3
interapp_rate = 1 - intraapp_rate
negative_size = 4 * n * 8 // 10
neg_split_index = 8 * len(page_list) // 10

# training
for epoch in range(7):
    feature_array = base_model.predict({'image_input': image_np, 'text_input': text_np,
                                        'uitree_input': uitree_np,
                                        'uitree_other': other_np},
                                       batch_size=64, verbose=True)

    intra_negative_pair = np.random.randint(neg_split_index, size=int(negative_size * intraapp_rate) * 2)
    intra_negative_pair = intra_negative_pair.reshape(int(negative_size * intraapp_rate), 2)
    intra_negative_pair = np.array([p for p in intra_negative_pair if p[0] != p[1] and adj[p[0]][p[1]]!=1 and same_app(p[0], p[1])])
    inter_negative_pair = np.random.randint(neg_split_index, size=int(negative_size * interapp_rate) * 2)
    inter_negative_pair = inter_negative_pair.reshape(int(negative_size * interapp_rate), 2)
    inter_negative_pair = np.array([p for p in inter_negative_pair if p[0] != p[1] and adj[p[0]][p[1]]!=1 and not same_app(p[0], p[1])])

    temp_widget_image_np = []
    temp_widget_tag_np = []
    temp_widget_other_np = []
    temp_widget_text_np = []

    for eachpair in intra_negative_pair:
        filename1 = args.paladin_dir + page_list[eachpair[0]] + ".json"
        f = open(filename1, 'br')
        setting = json.load(f)
        root = setting["root"]
        del setting
        x, y, w, h, tag, text = dfs(root)
        temp_widget_image_np.append(crop_image(args.paladin_dir + page_list[eachpair[0]] + ".png", x, y, w, h))
        temp_widget_tag_np.append(tag.split(".")[-1])
        temp_widget_other_np.append([x, y, w, h])
        temp_widget_text_np.append(gettextvector(text))
        f.close()
    for eachpair in inter_negative_pair:
        filename1 = args.paladin_dir + page_list[eachpair[0]] + ".json"
        f = open(filename1, 'br')
        setting = json.load(f)
        root = setting["root"]
        del setting
        x, y, w, h, tag, text = dfs(root)
        temp_widget_image_np.append(crop_image(args.paladin_dir + page_list[eachpair[0]] + ".png", x, y, w, h))
        temp_widget_tag_np.append(tag.split(".")[-1])
        temp_widget_other_np.append([x, y, w, h])
        temp_widget_text_np.append(gettextvector(text))
        f.close()
    temp_widget_image_np.extend(widget_image_np[:split_index_new])
    temp_widget_tag_np.extend(widget_tag_np[:split_index_new])
    temp_widget_other_np.extend(widget_other_np[:split_index_new])
    temp_widget_text_np.extend(widget_text_np[:split_index_new])
    temp_widget_image_np = np.array(temp_widget_image_np)
    temp_widget_tag_np = np.array(temp_widget_tag_np)
    temp_widget_other_np = np.array(temp_widget_other_np)
    temp_widget_text_np = np.array(temp_widget_text_np)

    labels = np.concatenate((np.zeros(len(intra_negative_pair)),
                             np.zeros(len(inter_negative_pair)), 
                             np.ones(len(positive_pair_tr))))
    pair = np.concatenate((intra_negative_pair, inter_negative_pair, positive_pair_tr))
    del intra_negative_pair, inter_negative_pair
    shuffled_idx = np.arange(len(labels))
    np.random.shuffle(shuffled_idx)
    shuffled_pair = pair[shuffled_idx]
    shuffled_labels = labels[shuffled_idx]
    del pair, labels
    shuffled_widget_tag_np = temp_widget_tag_np[shuffled_idx]
    del temp_widget_tag_np
    shuffled_widget_other_np = temp_widget_other_np[shuffled_idx]
    del temp_widget_other_np
    shuffled_widget_image_np = temp_widget_image_np[shuffled_idx]
    del temp_widget_image_np

    tmp_shuffled_widget_tag_np = []
    for eachtag in shuffled_widget_tag_np:
        embedding = all_tags_dict.get(eachtag, None)
        if embedding is not None:
            tmp_shuffled_widget_tag_np.append([embedding])
        else:
            tmp_shuffled_widget_tag_np.append([all_tags_dict.get("unknown", None)])
    shuffled_widget_tag_np = np.array(tmp_shuffled_widget_tag_np)
    shuffled_tag = []
    for each in shuffled_widget_tag_np:
        num = each[0]
        shuffled_tag.append(num)
    shuffled_tag = np.array(shuffled_tag)
    shuffle_widget_text_np = temp_widget_text_np[shuffled_idx]
    del temp_widget_text_np, shuffled_idx

    print('start training')
    loss = 0
    acc = 0
    for i in range(shuffled_labels.shape[0] // 32 + 1):
        j = i * 32
        k = i * 32 + 32
        if k >= shuffled_labels.shape[0]:
            k = shuffled_labels.shape[0]
        if j == k:
            break
        e = model.train_on_batch({'image_input1': image_np[shuffled_pair[j:k, 0]],
            'text_input1': text_np[shuffled_pair[j:k, 0]],
            'uitree_input1': uitree_np[shuffled_pair[j:k, 0]],
            'uitree_other1': other_np[shuffled_pair[j:k, 0]],
            'image_input2': image_np[shuffled_pair[j:k, 1]],
            'text_input2': text_np[shuffled_pair[j:k, 1]],
            'uitree_input2': uitree_np[shuffled_pair[j:k, 1]],
            'uitree_other2': other_np[shuffled_pair[j:k, 1]],
            'image_input1_0': shuffled_widget_image_np[j:k], 
            'tag_input1_0': shuffled_tag[j:k],
            'uitree_other1_0': shuffled_widget_other_np[j:k], 
            'text_input1_0': shuffled_widget_text_np[j:k]}, shuffled_labels[j:k], class_weight=cw)
        loss += e[0]
        acc += e[1]
        if i % 20 == 0:
            print(i, loss / (i + 1), acc / (i + 1))
    loss /= (shuffled_labels.shape[0] // 32 + 1)
    acc /= (shuffled_labels.shape[0] // 32 + 1)
    print(loss, acc)
    del shuffled_widget_other_np, shuffled_widget_tag_np, shuffled_widget_image_np
    del shuffled_pair, shuffled_labels

    # test data
    negative_size_t = n // 5 * 3
    intra_negative_pair_t = np.random.randint(neg_split_index // 4, size=int(negative_size_t * intraapp_rate) * 2)
    intra_negative_pair_t = intra_negative_pair_t + neg_split_index
    intra_negative_pair_t = intra_negative_pair_t.reshape(int(negative_size_t * intraapp_rate), 2)
    intra_negative_pair_t = np.array(
        [p for p in intra_negative_pair_t if p[0] != p[1] and adj[p[0]][p[1]]!=1 and same_app(p[0], p[1])])
    inter_negative_pair_t = np.random.randint(neg_split_index // 4, size=int(negative_size_t * interapp_rate) * 2)
    inter_negative_pair_t = inter_negative_pair_t + neg_split_index
    inter_negative_pair_t = inter_negative_pair_t.reshape(int(negative_size_t * interapp_rate), 2)
    inter_negative_pair_t = np.array(
        [p for p in inter_negative_pair_t if p[0] != p[1] and adj[p[0]][p[1]]!=1 and not same_app(p[0], p[1])])

    temp_widget_image_np_t = []
    temp_widget_tag_np_t = []
    temp_widget_other_np_t = []
    temp_widget_text_np_t = []
    for eachpair in intra_negative_pair_t:
        filename1 = args.paladin_dir + page_list[eachpair[0]] + ".json"
        f = open(filename1, 'br')
        setting = json.load(f)
        root = setting["root"]
        x, y, w, h, tag, text = dfs(root)
        temp_widget_image_np_t.append(crop_image(args.paladin_dir + page_list[eachpair[0]] + ".png", x, y, w, h))
        temp_widget_tag_np_t.append(tag.split(".")[-1])
        temp_widget_other_np_t.append([x, y, w, h])
        temp_widget_text_np_t.append(gettextvector(text))
        f.close()
    for eachpair in inter_negative_pair_t:
        filename1 = args.paladin_dir + page_list[eachpair[0]] + ".json"
        f = open(filename1, 'br')
        setting = json.load(f)
        root = setting["root"]
        x, y, w, h, tag, text = dfs(root)
        temp_widget_image_np_t.append(crop_image(args.paladin_dir + page_list[eachpair[0]] + ".png", x, y, w, h))
        temp_widget_tag_np_t.append(tag.split(".")[-1])
        temp_widget_other_np_t.append([x, y, w, h])
        temp_widget_text_np_t.append(gettextvector(text))
        f.close()
    temp_widget_image_np_t.extend(widget_image_np[split_index_new:])
    temp_widget_tag_np_t.extend(widget_tag_np[split_index_new:])
    temp_widget_other_np_t.extend(widget_other_np[split_index_new:])
    temp_widget_text_np_t.extend(widget_text_np[split_index_new:])
    temp_widget_image_np_t = np.array(temp_widget_image_np_t)
    temp_widget_tag_np_t = np.array(temp_widget_tag_np_t)
    temp_widget_other_np_t = np.array(temp_widget_other_np_t)
    temp_widget_text_np_t = np.array(temp_widget_text_np_t)

    labels_t = np.concatenate(
        (np.zeros(len(intra_negative_pair_t)), np.zeros(len(inter_negative_pair_t)), np.ones(len(positive_pair_test))))
    pair_t = np.concatenate((intra_negative_pair_t, inter_negative_pair_t, positive_pair_test))
    shuffled_idx_t = np.arange(len(labels_t))
    np.random.shuffle(shuffled_idx_t)
    shuffled_pair_t = pair_t[shuffled_idx_t]
    shuffled_labels_t = labels_t[shuffled_idx_t]
    shuffled_widget_image_np_t = temp_widget_image_np_t[shuffled_idx_t]
    shuffled_widget_tag_np_t = temp_widget_tag_np_t[shuffled_idx_t]
    shuffled_widget_other_np_t = temp_widget_other_np_t[shuffled_idx_t]
    shuffled_widget_text_np_t = temp_widget_text_np_t[shuffled_idx_t]
    del intra_negative_pair_t, inter_negative_pair_t, pair_t, shuffled_idx_t, labels_t
    del temp_widget_tag_np_t, temp_widget_other_np_t
    del temp_widget_image_np_t, temp_widget_text_np_t
    tmp_shuffled_widget_tag_np_t = []
    for eachtag in shuffled_widget_tag_np_t:
        embedding = all_tags_dict.get(eachtag, None)
        if embedding is not None:
            tmp_shuffled_widget_tag_np_t.append([embedding])
        else:
            tmp_shuffled_widget_tag_np_t.append([all_tags_dict.get("unknown", None)])
    shuffled_widget_tag_np_t = np.array(tmp_shuffled_widget_tag_np_t)
    shuffled_tag_t = []
    for each in shuffled_widget_tag_np_t:
        num = each[0]
        shuffled_tag_t.append(num)
    shuffled_tag_t = np.array(shuffled_tag_t)

    pred = []
    for i in range(shuffled_labels_t.shape[0] // 64 + 1):
        j = i * 64
        k = i * 64 + 64
        if k >= shuffled_labels_t.shape[0]:
            k = shuffled_labels_t.shape[0]
        if j == k:
            break
        pred1 = model.predict_on_batch({'image_input1': image_np[shuffled_pair_t[j:k, 0]],
            'text_input1': text_np[shuffled_pair_t[j:k, 0]],
            'uitree_input1': uitree_np[shuffled_pair_t[j:k, 0]],
            'uitree_other1': other_np[shuffled_pair_t[j:k, 0]],
            'image_input2': image_np[shuffled_pair_t[j:k, 1]],
            'text_input2': text_np[shuffled_pair_t[j:k, 1]],
            'uitree_input2': uitree_np[shuffled_pair_t[j:k, 1]],
            'uitree_other2': other_np[shuffled_pair_t[j:k, 1]],
            'image_input1_0': shuffled_widget_image_np_t[j:k], 
            'tag_input1_0': shuffled_tag_t[j:k],
            'uitree_other1_0': shuffled_widget_other_np_t[j:k],
            'text_input1_0': shuffled_widget_text_np_t[j:k]})
        predlist = list(pred1)
        del pred1
        pred.extend(predlist)
    p = []
    for each in pred:
        if each >= threshold:
            p.append(1)
        else:
            p.append(0)
    p = np.array(p)
    print(classification_report(shuffled_labels_t, p, digits=4))
    del p, pred, feature_array, shuffled_pair_t, shuffled_labels_t

    model.save(args.output_dir + "pretrained_model.h5")
