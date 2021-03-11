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


GPU_ID = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

parser = argparse.ArgumentParser(description='predict the window transition relationship.')
parser.add_argument('--data_dir', type=str, required=True,
    help='path to pre-processed training data for pre-training')
parser.add_argument('--model_dir', type=str, default='./model/',
    help='path to the fine-tuned model')
parser.add_argument('--app', type=str, required=True, help='the app to predict')
parser.add_argument('--widget_name', type=str, default='null', 
    help='the widget id of the source widget')
parser.add_argument('--source_activity', type=str, required=True, help='source activity')
parser.add_argument('--target_activity', type=str, required=True, help='target activity')
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


siamese_model = load_model(args.model_dir + "finetuned_model.h5", {'ntn_layer': ntn_layer})

activity_list = list(np.load(args.data_dir + "activity_np.npy"))
widget_list = list(np.load(args.data_dir + "widget_np.npy"))

img_np = np.load(args.data_dir + "image_np.npy")
text_np = np.load(args.data_dir + "text_np.npy")
uitree_np = np.load(args.data_dir + "uitree_np.npy")
other_np = np.load(args.data_dir + "other_np.npy")
w_image_np = np.load(args.data_dir + "w_image_np.npy")
w_text_np = np.load(args.data_dir + "w_text_np.npy")
w_uitree_np = np.load(args.data_dir + "w_uitree_np.npy")
w_other_np = np.load(args.data_dir + "w_other_np.npy")

static_activity_list = list(np.load(args.data_dir + "rendering_activity_list.npy"))
static_widget_list = list(np.load(args.data_dir + "rendering_widget_list.npy"))

static_img_np = np.load(args.data_dir + "rendering_img_np.npy")
static_text_np = np.load(args.data_dir + "rendering_text_np.npy")
static_uitree_np = np.load(args.data_dir + "rendering_uitree_np.npy")
static_other_np = np.load(args.data_dir + "rendering_other_np.npy")
static_w_image_np = np.load(args.data_dir + "rendering_w_img_np.npy")
static_w_text_np = np.load(args.data_dir + "rendering_w_text_np.npy")
static_w_uitree_np = np.load(args.data_dir + "rendering_w_tag_np.npy")
static_w_other_np = np.load(args.data_dir + "rendering_w_other_np.npy")

jsonfilepath = args.data_dir + 'ActivityName2Name.json'
f = open(jsonfilepath, 'br')
activityName2Name = json.load(f)

def find_widget_in_static(widgetname, source_activity):
    return_list = []
    for eachwidget in static_widget_list:
        if eachwidget.split("/")[0] != source_activity.split("/")[0]:
            continue
        if eachwidget.split("/")[1] != source_activity.split("/")[1]:
            continue
        if eachwidget.split("/")[-1] == widgetname:
            return_list.append(static_widget_list.index(eachwidget))
    return return_list

def find_all_in_static(source_activity):
    return_list = []
    for eachwidget in static_widget_list:
        if eachwidget.split("/")[0] != source_activity.split("/")[0]:
            continue
        if eachwidget.split("/")[1] != source_activity.split("/")[1]:
            continue
        return_list.append(static_widget_list.index(eachwidget))
    return return_list


app = args.app
widgetname = args.widget_name
source_activity = args.source_activity
target_activity = args.target_activity
target_imgt, target_textt, target_uitreet, target_othert = [], [], [], []
source_img, source_text, source_uitree, source_other = [], [], [], []
w_img, w_text, w_uitree, w_other = [], [], [], []
if target_activity[0] == 'n':
    # find result in static data
    for eachname in activityName2Name[target_activity]:
        if eachname.split("/")[0] != app:
            continue
        target_index = static_activity_list.index(eachname)
        target_imgt.append(static_img_np[target_index])
        target_textt.append(static_text_np[target_index])
        target_uitreet.append(static_uitree_np[target_index])
        target_othert.append(static_other_np[target_index])
else:
    target_activity_name = app + "/" + target_activity
    target_index = activity_list.index(target_activity_name)
    target_imgt.append(img_np[target_index])
    target_textt.append(text_np[target_index])
    target_uitreet.append(uitree_np[target_index])
    target_othert.append(other_np[target_index])
    target_img, target_text, target_uitree, target_other = [], [], [], []

if source_activity[0] == 'n':     
    # find result in static data
    for eachname in activityName2Name[source_activity]:
        widlist = []
        if eachname.split("/")[0] != app:
            continue
        source_index = static_activity_list.index(eachname)
        if widgetname != "null":
            widlist.extend(find_widget_in_static(widgetname, eachname))
        else:
            #find all widgets of the source activity
            widlist.extend(find_all_in_static(eachname))  
        for i in range(len(widlist)):
            for j in range(len(target_imgt)):
                source_img.append(static_img_np[source_index])
                source_text.append(static_text_np[source_index])
                source_uitree.append(static_uitree_np[source_index])
                source_other.append(static_other_np[source_index])
                w_img.append(static_w_image_np[widlist[i]])
                w_text.append(static_w_text_np[widlist[i]])
                w_uitree.append(static_w_uitree_np[widlist[i]])
                w_other.append(static_w_other_np[widlist[i]])
            target_img.extend(target_imgt)
            target_text.extend(target_textt)
            target_uitree.extend(target_uitreet)
            target_other.extend(target_othert)
else:
    source_activity_name = app + "/" + source_activity
    source_index = activity_list.index(source_activity_name)
    widlist = []
    if widgetname != "null":
        idx = 0
        for eachwidget in widget_list:
            if eachwidget.split(": ")[0] != source_activity_name:
                continue
            if eachwidget.split(": ")[1] == "":
                continue
            if eachwidget.split(": ")[1].split("/")[1] == widgetname:
                widlist.append(idx)
            idx += 1
    else:
        #find all widgets of the source activity
        idx = 0
        for eachwidget in widget_list:
            if eachwidget.split(": ")[0] == source_activity_name:
                widlist.append(idx)
            idx += 1
    for i in range(len(widlist)):
        for j in range(len(target_imgt)):  
            source_img.append(img_np[source_index])
            source_text.append(text_np[source_index])
            source_uitree.append(uitree_np[source_index])
            source_other.append(other_np[source_index])
            w_img.append(w_image_np[widlist[i]])
            w_text.append(w_text_np[widlist[i]])
            w_uitree.append(w_uitree_np[widlist[i]])
            w_other.append(w_other_np[widlist[i]])
        target_img.extend(target_imgt)
        target_text.extend(target_textt)
        target_uitree.extend(target_uitreet)
        target_other.extend(target_othert)
length = len(source_img)
pred = siamese_model.predict({'image_input1': np.array(source_img).reshape(length, 160, 90, 3),
    'text_input1': np.array(source_text).reshape(length, 256),
    'uitree_input1': np.array(source_uitree).reshape(length, 256),
    'uitree_other1': np.array(source_other).reshape(length, 128, 4),
    'image_input2': np.array(target_img).reshape(length, 160, 90, 3),
    'text_input2': np.array(target_text).reshape(length, 256),
    'uitree_input2':np.array(target_uitree).reshape(length, 256),
    'uitree_other2': np.array(target_other).reshape(length, 128, 4),
    'image_input1_0': np.array(w_img).reshape(length, 160, 90, 3),
    'tag_input1_0': np.array(w_uitree).reshape(length, 1),
    'uitree_other1_0': np.array(w_other).reshape(length, 4),
    'text_input1_0': np.array(w_text).reshape(length, 256)})
pred_result = -1
for eachpred in pred:
    pred_result = max(pred_result, eachpred[0])
print("pred result", pred_result)
