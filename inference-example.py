#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import numpy as np
import cv2
# import tensorflow as tf

from tensorpack import TowerContext
from tensorpack.tfutils import get_model_loader
from tensorpack.dataflow.dataset import ILSVRCMeta
# tf.disable_v2_behavior() 
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()
import nets
from PIL import Image
import h5py
from torchvision import transforms 


"""
A small inference example for attackers to play with.
"""


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--depth', help='ResNet depth',
                    type=int, default=152, choices=[50, 101, 152])
parser.add_argument('--arch', help='Name of architectures defined in nets.py',
                    default='ResNetDenoise')
parser.add_argument('--load', help='path to checkpoint')
parser.add_argument('--input', help='path to input image')
parser.add_argument('--neuronwise', help='whether neuro wise or not')
parser.add_argument('--session', help='session name')
args = parser.parse_args()

model = getattr(nets, args.arch + 'Model')(args)

input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
image = input / 127.5 - 1.0
image = tf.transpose(image, [0, 3, 1, 2])
# if torch.cuda.is_available():
#   image = tf.transpose(image, [0, 3, 1, 2])
with TowerContext('', is_training=False):
    logits,act_dict = model.get_logits(image)

sess = tf.Session()
get_model_loader(args.load).init(sess)


f = h5py.File(args.input,'r')
natural_data = f['images/naturalistic'][:]
session_path=args.session.replace('_','/')
final_path=session_path[:-1]+'_'+session_path[-1:]
synth_data = f['images/synthetic/monkey_'+final_path][:]
#synth_data=f['images/synthetic/monkey_m/stretch/session_1'][:]
print(natural_data.shape)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
      
        yield iterable[ndx:min(ndx + n, l)]
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225])
])
natural_sample = np.array([np.array(preprocess((Image.fromarray(i)).convert('RGB'))) for i in natural_data]).transpose(0,2,3,1)
synth_sample = np.array([np.array(preprocess((Image.fromarray(i)).convert('RGB'))) for i in synth_data]).transpose(0,2,3,1)

counter=0
for  minibatch in batch(natural_sample,64):
  prob = sess.run(logits,feed_dict={input: minibatch})
  activation = sess.run(act_dict,feed_dict={input: minibatch})
  if counter==0:
    with h5py.File('ResNeXtDenoiseAll_natural_layer_activation.hdf5','w')as f:
      for layer in activation.keys():
        dset=f.create_dataset(layer,data=activation[layer])
  else:
    with h5py.File('ResNeXtDenoiseAll_natural_layer_activation.hdf5','r+')as f:
        for k,v in activation.items():
          print(k)
          data = f[k]
          print(data.shape)
          a=data[...]
          del f[k]
          dset=f.create_dataset(k,data=np.concatenate((a,activation[k]),axis=0))
  counter=counter+1


prob = sess.run(logits,feed_dict={input: synth_sample})
activation = sess.run(act_dict,feed_dict={input: synth_sample})
# if counter==0:
with h5py.File('ResNeXtDenoiseAll_synth_layer_activation.hdf5','w')as f:
  for layer in activation.keys():
    dset=f.create_dataset(layer,data=activation[layer])
# else:
with h5py.File('ResNeXtDenoiseAll_synth_layer_activation.hdf5','r+')as f:
    for k,v in activation.items():
      print(k)
      data = f[k]
      print(data.shape)
      a=data[...]
      del f[k]
      dset=f.create_dataset(k,data=np.concatenate((a,activation[k]),axis=0))


import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
import torch.nn.functional as F

natural_score_dict={}
synth_score_dict={}
# random_list=[2,5,667,89,43]
random_list=[2,10,32,89,43]
#layerlist=['maxpool','layer1[0]','layer1[1]','layer1[2]','layer2[0]','layer2[1]','layer2[2]','layer2[3]','layer3[0]','layer3[1]','layer3[2]','layer3[3]','layer3[4]','layer3[5]','layer4[0]','layer4[1]','layer4[2]','avgpool','fc']
for key in layerlist:
  natural_score_dict[key]=None
  synth_score_dict[key]=None
total_synth_corr=[]
total_natural_corr=[]
cc=0
with h5py.File(f'{model_type}_synth_layer_activation.hdf5','r')as s:
  with h5py.File(f'{model_type}_natural_layer_activation.hdf5','r')as f:
    for seed in random_list:
      for k in layerlist:
        print(k)
        natural_data = f[k]
        synth_data=s[k]
        a=natural_data[...]
        b=synth_data[...]
        # pca=cuPCA(n_components=640)
        pca=PCA(random_state=seed)
        # array=np.asfortranarray(a.reshape(640,-1))
        # X_gpu=gpuarray.to_gpu(array)
        # natural_x_pca = pca.fit_transform(X_gpu)
        natural_x_pca = pca.fit_transform(torch.tensor(a).cpu().detach().reshape(640,-1))
        print(natural_x_pca.shape)
        # array=np.asfortranarray(b.reshape(50,-1))
        # X_gpu=gpuarray.to_gpu(array)
        synth_x_pca = pca.transform(torch.tensor(b).cpu().detach().reshape(neuron_target.shape[0],-1))
        kfold = KFold(n_splits=5, shuffle=True,random_state=seed)
        num_neuron=n1.shape[2]
        natural_prediction= np.empty((640,target.shape[1]), dtype=object)
        synth_prediction=np.empty((neuron_target.shape[0],neuron_target.shape[1]), dtype=object)
        for fold, (train_ids, test_ids) in enumerate(kfold.split(natural_x_pca)):
          clf = Ridge(random_state=seed)
          clf.fit((natural_x_pca)[train_ids],target[train_ids])
          start=fold*10
          end=((fold+1)*10)
          natural_prediction[test_ids]=clf.predict((natural_x_pca)[test_ids])
          # synth_prediction[start:end]=clf.predict((synth_x_pca)[start:end])
          if fold==0:
            synth_prediction=clf.predict((synth_x_pca))
          else:
            synth_prediction=synth_prediction+clf.predict((synth_x_pca))
          if fold==4:
            synth_prediction=synth_prediction/5

        if natural_score_dict[k] is None:
          natural_corr_array= np.array([pearsonr(natural_prediction[:, i], target[:, i])[0] for i in range(natural_prediction.shape[-1])])
          total_natural_corr=natural_corr_array
          natural_score_dict[k] = np.median(natural_corr_array)
          cc=cc+1
        else:
          natural_corr_array= np.array([pearsonr(natural_prediction[:, i], target[:, i])[0] for i in range(natural_prediction.shape[-1])])
          total_natural_corr=np.vstack([total_natural_corr,natural_corr_array])
          natural_score=np.median(natural_corr_array)
          natural_score_dict[k] =np.append(natural_score_dict[k],natural_score)
          cc=cc+1
        if synth_score_dict[k] is None:
          synth_corr_array=np.array([pearsonr(synth_prediction[:, i], neuron_target[:, i])[0] for i in range(synth_prediction.shape[-1])])
          total_synth_corr=synth_corr_array
          synth_score_dict[k] = np.median(synth_corr_array)
        else:
          synth_corr_array=np.array([pearsonr(synth_prediction[:, i], neuron_target[:, i])[0] for i in range(synth_prediction.shape[-1])])
          total_synth_corr=np.vstack([total_synth_corr,synth_corr_array])
          synth_score=np.median(synth_corr_array)
          synth_score_dict[k] =np.append(synth_score_dict[k],synth_score)
        # natural_score_dict[k] = np.median(np.array([pearsonr(natural_prediction[:, i], target[:, i])[0] for i in range(natural_prediction.shape[-1])]))
        # synth_score_dict[k] = np.median(np.array([pearsonr(synth_prediction[:, i], neuron_target[:, i])[0] for i in range(synth_prediction.shape[-1])]))

        print(natural_score_dict[k])
        print(synth_score_dict[k]) 
    print(cc)
    if args.neurowise==True:
      # total_synth_corr=total_synth_corr/len(random_list)
      # total_natural_corr=total_natural_corr/len(random_list)
      np.save(f'gdrive/MyDrive/V4/monkey_+{final_path}/{args.arch}_synth_neuron_corr.npy',total_synth_corr)
      np.save(f'gdrive/MyDrive/V4/monkey_+{final_path}/{args.arch}_natural_neuron_corr.npy',total_natural_corr)


    else:


      from statistics import mean
      new_natural_score_dict = {k:  v.tolist() for k, v in natural_score_dict.items()}
      new_synth_score_dict = {k:  v.tolist() for k, v in synth_score_dict.items()}
      import json
      # Serializing json  
      synth_json = json.dumps(new_synth_score_dict, indent = 4) 
      natural_json = json.dumps(new_natural_score_dict, indent = 4) 
      print(natural_json)
      print(synth_json)

      with open(f"gdrive/MyDrive/V4/monkey_+{final_path}/{args.arch}_natural.json", 'w') as f:
        json.dump(natural_json, f)
      with open(f"gdrive/MyDrive/V4/monkey_+{final_path}/{args.arch}_synth.json", 'w') as f:
        json.dump(synth_json, f)

      natural_mean_dict = {k:  mean(v) for k, v in natural_score_dict.items()}
      synth_mean_dict = {k:  mean(v) for k, v in synth_score_dict.items()}
      json_object = json.dumps(natural_mean_dict, indent = 4) 
      print(json_object)
      with open(f"gdrive/MyDrive/V4/monkey_+{final_path}/{args.arch}_natural_mean.json", 'w') as f:
        json.dump(json_object, f)

      json_object = json.dumps(synth_mean_dict, indent = 4) 
      print(json_object)
      with open(f"gdrive/MyDrive/V4/monkey_+{final_path}/{args.arch}_synth_mean.json", 'w') as f:
        json.dump(json_object, f)
# for i in range(2):

#   sample = cv2.imread(args.input)  # this is a BGR image, not RGB
#   # imagenet evaluation uses standard imagenet pre-processing
#   # (resize shortest edge to 256 + center crop 224).
#   # However, for images of unknown sources, let's just do a naive resize.
#   sample = cv2.resize(sample, (224, 224))

#   prob = sess.run(logits,feed_dict={input: np.array([sample])})

#   activation = sess.run(act_dict,feed_dict={input: np.array([sample])})

#   print("Prediction: ", prob.argmax())

#   synset = ILSVRCMeta().get_synset_words_1000()
#   print("Top 5: ", [synset[k] for k in prob[0].argsort()[-5:][::-1]])

#   print(activation['layer1[0]'].shape)
#   print(type(activation['layer1[0]']))
#   if counter==0:
#     with h5py.File('ResNeXtDenoiseAll_natural_layer_activation.hdf5','w')as f:
#       for layer in activation.keys():
#         dset=f.create_dataset(layer,data=activation[layer])
#   else:
#     with h5py.File('ResNeXtDenoiseAll_natural_layer_activation.hdf5','r+')as f:
#         for k,v in activation.items():
#           print(k)
#           data = f[k]
          
#           a=data[...]
#           del f[k]
#           dset=f.create_dataset(k,data=np.concatenate((a,activation[k]),axis=0))
#   counter=counter+1
