#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import gensim
import io
import json
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from PIL import Image
import string


parser = argparse.ArgumentParser(description='Pre-processing static analysis data.')
parser.add_argument('--data_dir', type=str, required=True,
    help='path to pre-processed training data for pre-training')
parser.add_argument('--static_dir', type=str, required=True,
    help='path to rendering result of apps')
parser.add_argument('--output_dir', type=str, required=True,
    help='path to directory to save the pre-processed numpys')
args = parser.parse_args()

GPU_ID = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
model = gensim.models.Word2Vec.load(args.data_dir + '/word2vec_wx')
all_tags_dict = json.load(open(args.data_dir + "./all_tags_dict.json", 'r'))

def get_image(img):
    try:
        im = Image.open(img).convert('RGB')
        new_img = im.resize((90, 160), Image.BILINEAR)
        data = np.array(new_img)
        return data
    except IOError:
        a = np.zeros([160, 90, 3], dtype='float32')
        return a

def get_tags_rendering(widget):
    tag = widget['Class'].split('.')[-1]
    x = int(widget['Position'].split(',')[0][1:])
    y = int(widget['Position'].split(',')[1].split(')')[0])
    a = int(widget['Position'].split(',')[1].split('(')[1])
    b = int(widget['Position'].split(',')[-1][:-1])
    width = a - x
    height = b - y
    text = widget['Content']
    if text == "null":
        text = ""
    total = np.zeros(shape=(256, ), dtype='float32')
    for c in string.punctuation:
        text = text.replace(c, " ")
    totalnum = 1
    if text != "":
        textlist = text.split(" ")
        totalnum = len(textlist)
        for eachword in textlist:
            total += model[eachword]
    num = len(widget['Children'])
    if num == 0:
        if totalnum > 0:
            total = total / totalnum
        return tag, [[x, y, width, height]], total, totalnum
    mid = num // 2
    tagsequence = ""
    otherlist = []
    i = 0
    for each in widget['Children']:
        if i > mid:
            break
        m, n, o, tmpnum = get_tags_rendering(widget['Children'][each])
        tagsequence += (" " + m + " ")
        otherlist.extend(n)
        total += (o * num)
        totalnum += tmpnum
        i += 1
    tagsequence += (" " + tag + " ")
    other = [x, y, width, height]
    otherlist.append(other)
    for each in widget['Children']:
        if i <= mid:
            break
        m, n, o, tmpnum = get_tags_rendering(widget['Children'][each])
        tagsequence += (" " + m + " ")
        otherlist.extend(n)
        total += (o * num)
        totalnum += tmpnum
        i += 1
    if totalnum > 0:
        total = total / totalnum
    return tagsequence, otherlist, total, totalnum

def crop_image(img, x, y, width, height):
    try:
        im = Image.open(img).convert('RGB')
        data = np.array(im)
        for i in range(800):
            for j in range(480):
                if i < x  or i >= (x + width):
                    if j < y  or j >= (y + height):
                        data[i][j][0] = data[i][j][1] = data[i][j][2] = 0
        img = Image.fromarray(data).convert('RGB')
        img = img.resize((90, 160), Image.BILINEAR)
        data = np.array(img)
        return data
    except IOError:
        a = np.zeros([160, 90, 3], dtype='float32')
        return a

def save_widgets(widget, activity_name, imgpath):
    tag = widget['Class'].split('.')[-1]
    x = int(widget['Position'].split(',')[0][1:])
    y = int(widget['Position'].split(',')[1].split(')')[0])
    a = int(widget['Position'].split(',')[1].split('(')[1])
    b = int(widget['Position'].split(',')[-1][:-1])
    width = a - x
    height = b - y
    text = widget['Content']
    if text == "null":
        text = ""
    total = np.zeros(shape=(256,), dtype='float32')
    for c in string.punctuation:
        text = text.replace(c, " ")
    if text != "":
        textlist = text.split(" ")
        totalnum = len(textlist)
        for eachword in textlist:
            total += model[eachword]
        if totalnum > 0:
            total = total / len(textlist)
    tag = [tag]
    other = [[x, y, width, height]]
    text = [total]
    widget_name = [activity_name + "/" + widget['Position'] + "/" + widget['WidgetID']] 
    img = [crop_image(imgpath, x, y, width, height)]
    for child in widget['Children']:
        tname, timg, ttag, tother, ttext = save_widgets(widget['Children'][child], 
            activity_name, imgpath)
        tag.extend(ttag)
        other.extend(tother)
        text.extend(ttext)
        widget_name.extend(tname)
        img.extend(timg)
    return widget_name, img, tag, other, text

# get rendering numpys
jsonpath = args.static_dir + "/viewtree_output_wid/"
activity_list = []
img_npy = []
tag_npy = []
other_npy = []
text_npy = []
dict = {}
w_img, w_tag, w_other, w_text, widget_list = [], [], [], [], []
rendering_data_path = args.static_dir + "/ui_output/"
jsonlist = os.listdir(jsonpath)
for eachjson in jsonlist:
    print(eachjson)
    jsonfilepath = jsonpath + eachjson
    f = open(jsonfilepath, 'r', errors = 'ignore')
    setting = json.load(f)
    cnt = 0
    for eachpage in setting:
        activity_list.append(eachjson[:-9] + "/" + eachpage['Name'])
        for each_activity_name in eachpage['ActivityName']:
            activity_name = each_activity_name
            if activity_name not in dict:
                dict[activity_name] = []
            dict[activity_name].append(eachjson[:-9] + "/" + eachpage['Name'])
        imgpath = rendering_data_path + eachpage['Screenshot'][12:]
        img_npy.append(get_image(imgpath))
        tags, others, texts, tmpnum = get_tags_rendering(eachpage['1'])
        tag_npy.append(tags)
        other_npy.append(others)
        text_npy.append(texts)
        cnt += 1
        widget_name, img, tag, other, text = save_widgets(eachpage['1'], 
            eachjson[:-9] + "/" + eachpage['Name'], imgpath)

        widget_list.extend(widget_name)
        w_img.extend(img)
        w_tag.extend(tag)
        w_other.extend(other)
        w_text.extend(text)

file_name = args.output_dir + 'ActivityName2Name.json'
with open(file_name, 'w') as file_obj:
    json.dump(dict, file_obj)

tmp_np = []
for eachuitree in tag_npy:
    tags = []
    for eachtag in eachuitree:
        embedding = all_tags_dict.get(eachtag, None)
        if embedding is not None:
            tags.append(embedding)
        else:
            tags.append(all_tags_dict.get("unknown", None))
    tmp_np.append(tags)
uitree_np = np.array(tmp_np)
uitree_np = pad_sequences(uitree_np, maxlen=256, dtype='float32', padding='post', truncating='pre')
activity_list = np.array(activity_list)

other_np = pad_sequences(np.array(other_npy), maxlen=128, dtype='float32',
    padding='post', truncating='pre')

tmp_np = []
for eachtag in w_tag:
    embedding = all_tags_dict.get(eachtag, None)
    if embedding is not None:
        tmp_np.append(embedding)
    else:
        tmp_np.append(all_tags_dict.get("unknown", None))
w_tag = np.array(tmp_np)
widget_list = np.array(widget_list)

np.save(args.output_dir + "rendering_widget_list.npy", widget_list)
np.save(args.output_dir + "rendering_activity_list.npy", activity_list)

np.save(args.output_dir + "rendering_img_np.npy", np.array(img_npy))
np.save(args.output_dir + "rendering_text_np.npy", np.array(text_npy))
np.save(args.output_dir + "rendering_uitree_np.npy",  uitree_np)
np.save(args.output_dir + "rendering_other_np.npy", other_np)

np.save(args.output_dir + "rendering_w_img_np.npy", np.array(w_img))
np.save(args.output_dir + "rendering_w_text_np.npy", np.array(w_text))
np.save(args.output_dir + "rendering_w_tag_np.npy",  np.array(w_tag))
np.save(args.output_dir + "rendering_w_other_np.npy", np.array(w_other))
