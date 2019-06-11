# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:06:53 2019

@author: MUJ
"""
import numpy as np
import random 
import keras
from keras import backend as K
from keras.layers import MaxPooling2D,Conv2D,Input,Add,Flatten,AveragePooling2D,Dense,BatchNormalization,ZeroPadding2D,Activation
from keras.models import Model

import cv2
import os
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
seed = 2109
random.seed = seed
np.random.seed = seed
tf.seed = seed
class DataGen(keras.utils.Sequence):
    def __init__(self,path,ids,batch_size=8,image_size=128):
        self.ids = ids
        self.path =path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end
    
    def __load__(self,id_name):
        image_path = os.path.join(self.path, id_name, "images", id_name)+ ".png"
        mask_path = os.path.join(self.path,id_name,"masks/")
        all_masks = os.listdir(mask_path)
        image = cv2.imread(image_path,1)
        image = cv2.resize(image,(self.image_size,self.image_size))
        
        mask = np.zeros((self.image_size,self.image_size,1))
        for name in all_masks:
            ind_mask = mask_path + name
            mask_mat = cv2.imread(ind_mask,0)
            mask_mat = cv2.resize(mask_mat,(self.image_size,self.image_size))
            mask_mat = np.expand_dims(mask_mat,axis =-1)
            mask = np.maximum(mask_mat,mask);
            
        image = image/255
        mask = mask/255
        return image,mask
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids)- index*self.batch_size
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        image = []
        mask = []
        for id_name in files_batch:
            _img,_mask =self.__load__(id_name)
            image.append(_img)                
            mask.append(_mask)
        image = np.array(image)
        mask = np.array(mask)

        return image,mask
    def on_epoch_end(self):
        pass
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

train_path = "New_folder"
epochs = 5
train_ids = next(os.walk(train_path))[1]
val_data_size = 10
valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

gen = DataGen(train_path,train_ids,batch_size=8,image_size=128)
x,y = gen.__getitem__(56)

r = random.randint(0, len(x)-1)


fig = plt.figure()
fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
ax = fig.add_subplot(1,2,1)
ax.imshow(x[r])

fig = plt.figure()
fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
ax = fig.add_subplot(1,2,1)
ax.imshow(np.reshape(y[r],(128,128)), cmap= "gray")
def Dense_Layer(x,k):
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    x = Conv2D(4*k,(1,1),strides = (1,1))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    x = Conv2D(k,(1,1),strides = (1,1))(x)
    return x

def Dense_Block(x,k):
    
    x1 = Dense_Layer(x,k)
    x1_add = keras.layers.Concatenate()([x1,x])
    x2 = Dense_Layer(x1_add,k)
    x2_add = keras.layers.Concatenate()([x1,x2])
    
    return x2_add
def Dilated_Spatial_Pyramid_Pooling(x,k):
    x = BatchNormalization(axis = 3)(x)
    d1 = Conv2D(k, (1,1), dilation_rate = 2)(x)
    d2 = Conv2D(k, (1,1), dilation_rate = 4)(d1)
    d3 = Conv2D(k, (1,1), dilation_rate = 8)(d2)
    d4 = Conv2D(k, (1,1), dilation_rate = 16)(d3)
    c = keras.layers.Concatenate()([d1,d2,d3,d4])
    return c

    
        
    
def down_block(x,filters, kernel_size = (3, 3), padding = "same",strides =1 ):
    c = Dense_Block(x,filters)
    c = Dense_Block(c,filters)
    p = keras.layers.MaxPool2D((2,2),(2,2))(c)
    return c,p
def up_block(x,skip,filters, kernel_size = (3, 3), padding = "same",strides =1 ):
    us = keras.layers.UpSampling2D((2,2))(x)
    concat = keras.layers.Concatenate()([us,skip])
    c = Dense_Block(concat,filters)
    c = Dense_Block(c,filters)
    return c
def bottleneck(x,filters, kernel_size = (3, 3), padding = "same",strides =1 ):
    c = Dense_Block(x,filters)
    c = Dense_Block(c,filters)
    c = Dilated_Spatial_Pyramid_Pooling(c,filters)
    return c

def UNet():
    f = [32,64,128,256]
    input = keras.layers.Input((128,128,3))
    
    
    p0 = input
    c1,p1 =  down_block(p0,f[0])
    c2,p2 =  down_block(p1,f[1])
    c3,p3 =  down_block(p2,f[2])

    
    bn = bottleneck(p3,f[3])
    
    u1 = up_block(bn,c3,f[2])
    u2 = up_block(u1,c2,f[1])
    u3 = up_block(u2,c1,f[0])
    
    
    outputs = keras.layers.Conv2D(1,(1,1),padding= "same",activation = "sigmoid")(u3)
    model = keras.models.Model(input,outputs)
    return model
model = UNet()
model.compile(optimizer = "adam",loss = "binary_crossentropy", metrics = ["acc"])
model.summary()
train_gen = DataGen(train_path,train_ids,batch_size = 8, image_size = 128)
valid_gen = DataGen(train_path,valid_ids,batch_size = 8, image_size = 128)
train_steps = len(train_ids)//8
valid_steps = len(valid_ids)//8
model.fit_generator(train_gen,validation_data = valid_gen,steps_per_epoch=train_steps,validation_steps =valid_steps,epochs= 5)
model.save("UNetM.h5")
model.save_weights("UNetW.h5")
x,y = valid_gen.__getitem__(3)
result = model.predict(x)
result = result >0.5

fig = plt.figure()
fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
ax = fig.add_subplot(1,2,1)
ax.imshow(np.reshape(y[0]*255,(128,128)), cmap= "gray")
fig = plt.figure()
fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
ax = fig.add_subplot(1,2,1)
ax.imshow(np.reshape(result[0]*255,(128,128)), cmap= "gray")
