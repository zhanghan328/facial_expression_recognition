#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:26:06 2017

@author: wind
"""

import pandas as pd
import numpy as np
import os
import cv2
data = pd.read_csv('./fer2013.csv')
for i in ['train','test']:
    for k in range(6):
        os.makedirs('data/%s/%s'%(i,k))
for i in range(len(data)):
    label = data['emotion'][i]
    image = data['pixels'][i].split(' ')
    
    image = np.array(image,dtype='float32')
    image.resize(48,48)
    if data['Usage'][i]=='Training':
        cv2.imwrite('data/train/%s/%s.jpg'%(label,i),image)
    else:
        cv2.imwrite('data/test/%s/%s.jpg'%(label,i),image)