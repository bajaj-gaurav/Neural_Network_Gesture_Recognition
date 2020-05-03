#!/usr/bin/env python
# coding: utf-8

# # Gesture Recognition

# ## In this project, we are developing a feature for smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote
# ### Each gesture corresponds to a specific command:
# 
# * Thumbs up:  Increase the volume
# * Thumbs down: Decrease the volume
# * Left swipe: 'Jump' backwards 10 seconds
# * Right swipe: 'Jump' forward 10 seconds  
# * Stop: Pause the movie

# In[8]:


# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')


# In[9]:


import numpy as np
import os
#from scipy.misc import imread, imresize
import datetime
import os
import cv2
import random
import math

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Note: scipy.misc is deprecated. So, will be using cv2 instead

# We set the random seed so that the results don't vary drastically.

# In[17]:


np.random.seed(30)
import random as rn
rn.seed(30)
from keras import backend as K
import tensorflow as tf
tf.set_random_seed(30)


# In[18]:


# function to plot n images using subplots
def plot_image(images, captions=None, cmap=None ):
    f, axes = plt.subplots(1, len(images), sharey=True)
    f.set_figwidth(15)
    for ax,image in zip(axes, images):
        ax.imshow(image, cmap)


# In this block, we are reading the folder names for training and validation. We also set the `batch_size`. Note that we set the batch size in such a way that we are able to use the GPU in full capacity. We kept increasing the batch size until the machine throws an error.

# In[19]:


# list of training folders
train_doc = np.random.permutation(open('./Project_data/train.csv').readlines())
# list of validation folders
val_doc = np.random.permutation(open('./Project_data/val.csv').readlines())


# In[20]:


# The batch size has been varied from 5 to 32 depending on the model layers
# The below value will be for the current model which can be at the model section
batch_size = 32


# ### Note: batch size have varied between 10 to 32 for the 15+ different models tried. The present value present above will be the best value for the present model¶

# ## Generator
# This is one of the most important part of the code. The overall structure of the generator has been given. In the generator, we are going to preprocess the images as we have images of 2 different dimensions as well as create a batch of video frames. We have experimented with `img_idx`, `y`,`z` and normalization such that we got high accuracy.

# * The image size we have is 360\*360 and 120\*160
# *  So, as 160\*120 is the minimum dimension here. We can either reduce everything to 160\*120 but that will be disproportionate for the bigger images. As the bigger images are square in shape, it will be better to have something in the square proportion. There are many options like 40\*40 or 80\*80 or 120\*120. These numbers are multiple of the original image size.
# 
# * We are going ahead with 120\*120 because it will reduce the bigger images by exactly one-third of the original size
# * And for the smaller images, we will not loose any information from x dimension. Just have to crop y dimension

# ### Augmentation technique
# #### We will try to see some of the augmentation technique and decide whether we want to use any. 
# * We will be trying horizontal flip, vertical flip and affine transformation and evaluate whether we want to use any of them
# * We have taken the sample for 120\*160 and 360\*360 to look at the augmentation

# In[7]:


# Function to generate random affine transform on the image for the sample image and validation on augmention
def get_affine_random():
    trans_x = 0
    trans_y = np.random.randint(-12, 12, 1)
    affine_transform = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    return affine_transform


# * We have choosen 10% of the image size for the translation

# In[8]:


# Printing the images after flipping, making some transformation, etc to see how the images look like
# The main aim is to visually check which augmentation technique is feasible using some visual samples
def print_image_augmentation(source_path, folder):
    imgs = os.listdir(source_path + folder)
    print(folder)
    print(source_path + folder +'/'+imgs[1])
    for i in range(30):
        image = cv2.imread(source_path + folder+'/'+imgs[i])#.astype(np.float32)
        image = cv2.resize(image, (120, 120), interpolation = cv2.INTER_AREA)
        #print(image)

        image1 = np.flip(image, 0)
        image2 = np.flip(image, 1)
        image3 = cv2.warpAffine(image, get_affine_random(), (120, 120))
        image4 = cv2.resize(image, (120, 120), interpolation = cv2.INTER_AREA)
        image5 = cv2.resize(image3, (120, 120), interpolation = cv2.INTER_AREA)
        plot_image([image, image1 ,image2, image3, image4, image5])

    # plotting the original image and the RGB channels
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    f.set_figwidth(15)
    ax1.imshow(image)

    # RGB channels
    ax2.imshow(image[:, : , 0])
    ax3.imshow(image[:, : , 1])
    ax4.imshow(image[:, : , 2])
    f.suptitle('Different Channels of Image')


# In[9]:


print_image_augmentation("./Project_data/train/", train_doc[1].split(";")[0])


# In[10]:


print_image_augmentation("./Project_data/train/", "WIN_20180907_15_38_24_Pro_Right Swipe_new")


# ### Conclusion
# #### We will be using 3 augmentation technique and see the result:
# * horizontal flipping
# * vertical transformation for the left and right swipe
# * horizontal transformation for the up and down swipe

# ### Generator code below using the above inferences of augmentation and resizing

# In[11]:


# The image size we have is 360*360 and 120*160
# So, as 160*120 is the minimum dimension here. We can either reduce everything to 160*120 but that will be
# disproportionate for the bigger images. As the bigger images are square in shape, it will be better to have something
# in the square proportion. There are many options like 40*40 or 80*80 or 120*120. These numbers are multiple of the 
# original image size.

# We are going ahead with 120*120 because it will reduce the bigger images by exactly one-third of the original size
# And for the smaller images, we will not loose any information from x dimension. Just have to crop y dimension


# In[12]:


# We will take into account all the frames for a sequence which is 30


# In[21]:


no_of_frames = 30
img_rows = 120
img_cols = 120
img_channels = 3  # RGB


# #### Resize vs cropping
# * We noted that resize gave the better result than cropping
# * Cropping might lead to information loss and we are fine with shrinking the images from single axis incase of 120*160 images. So, we will use resizing only.
# * Cropping code has been commented out in the below code but not removed

# In[22]:


def normalization(image):
    image - np.min(image)/np.max(image) - np.min(image)
    return image


# In[23]:


# Separate function for initialising as we have augmentated data also now
# function to initialise batch_labels and batch_data
def initialise_batch_data_labels(batch_size):
    batch_data = np.zeros((batch_size, no_of_frames, img_rows, img_cols, img_channels)) 
    batch_labels = np.zeros((batch_size,5))
    return batch_data, batch_labels


# In[24]:


# Function to generate random affine transform on the image
def get_transformation_hori():
    trans_y = 0
    trans_x = np.random.randint(-12, 12, 1)
    affine_transform = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    return affine_transform


# In[25]:


# Function to generate random affine transform on the image
def get_transformation_ver():
    trans_x = 0
    trans_y = np.random.randint(-12, 12, 1)
    affine_transform = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    return affine_transform


# In[26]:


# This function is responsible to load the sequence
# Make augmentation
# Cleanup like cropping, resize, etc
# Return the batch data and corresponding labels
def load_and_process_image(batch_size, source_path, batch, t, validation_flag):
    # no_of_frames is the number of images you use for each video, (img_rows, img_cols) is the final size of the input images and 3 is the number of channels RGB
    
    # batch_labels is the one hot representation of the output
    batch_data,batch_labels = initialise_batch_data_labels(batch_size)

    # variable for augumented batch data with affine transformation
    batch_data_flip_hor,batch_labels_flip_hor = initialise_batch_data_labels(batch_size)
    
    # variable for augumented batch data with affine transformation
    batch_data_transformation,batch_labels_transformaton = initialise_batch_data_labels(batch_size)
    
    # variable for augumented batch data with affine transformation
    # batch_data_trans_ver,batch_labels_trans_ver = initialise_batch_data_labels(batch_size)
    
    img_idx = [x for x in range(0, no_of_frames)] #create a list of image numbers you want to use for a particular video
    
    # iterate over the batch_size
    for folder in range(batch_size):
        # read all the images in the folder
        folder_name = t[folder + (batch*batch_size)].strip().split(';')
        imgs = os.listdir(source_path + '/' + folder_name[0])
        # Iterate iver the frames/images of a folder to read them in
        for idx, item in enumerate(img_idx):
            image = cv2.imread(source_path+'/'+ folder_name[0] + '/' + imgs[item])#.astype(np.float32)
            
            # crop the images. Note that the images are of 2 different shape 
            # and the conv3D will throw error if the inputs in a batch have different shapes
            # We noted that resize gave the better result than cropping
            # Also, cropping might lead to information loss
            #if image.shape[0] != image.shape[1]:
            #   h, w, _ = image.shape
            #    img_rows_half = img_rows/2
            #    image=image[0:120,20:140]
            # image = image[int(h/2)-img_rows/2:int(h/2)+img_rows/2, int(w/2)-img_rows/2:int(w/2)+img_rows/2]
            
            
            # resize the images. Note that the images are of 2 different shape 
            # and the conv3D will throw error if the inputs in a batch have different shapes  
            # We noted that resize gave the better result than cropping
            # Also, cropping might lead to information loss
            resized_image = cv2.resize(image, (img_rows,img_cols), interpolation = cv2.INTER_AREA)
            
            batch_data[folder,idx,:,:,0] = normalization(resized_image[:, : , 0])
            batch_data[folder,idx,:,:,1] = normalization(resized_image[:, : , 1])
            batch_data[folder,idx,:,:,2] = normalization(resized_image[:, : , 2])
 
            # Affine transformation
            # horizontal transformation for the up and down swipe
            # vertical transformation for the left swipe and right swipe images
            if int(folder_name[2] in [3, 4]):
                trans_image = cv2.warpAffine(resized_image, get_transformation_hori(), (resized_image.shape[0], resized_image.shape[1]))    
            elif int(folder_name[2] in [0, 1]):
                trans_image = cv2.warpAffine(resized_image, get_transformation_ver(), (resized_image.shape[0], resized_image.shape[1]))
            else:
                choice_list = ["hor", "ver"]
                choice = random.choice(choice_list)
                if choice == "hor":
                    trans_func = get_transformation_hori()
                else:
                    trans_func = get_transformation_ver()
                trans_image = cv2.warpAffine(resized_image, trans_func, (resized_image.shape[0], resized_image.shape[1]))
            batch_data_transformation[folder,idx,:,:,0] = normalization(trans_image[:, : , 0])
            batch_data_transformation[folder,idx,:,:,1] = normalization(trans_image[:, : , 1])
            batch_data_transformation[folder,idx,:,:,2] = normalization(trans_image[:, : , 2])
                
            #Horizontal flip
            flipped_image = np.flip(resized_image,1) 
            batch_data_flip_hor[folder,idx,:,:,0] = normalization(flipped_image[:, : , 0])
            batch_data_flip_hor[folder,idx,:,:,1] = normalization(flipped_image[:, : , 1])
            batch_data_flip_hor[folder,idx,:,:,2] = normalization(flipped_image[:, : , 2])
            
        batch_labels[folder, int(folder_name[2])] = 1

        # Labelling data with exchanged value for horizobtal flip
        # The right swipe becomes left swipe and left swipe becomes right swipe
        if int(folder_name[2]) == 0:
            batch_labels_flip_hor[folder, 1] = 1
        elif int(folder_name[2]) == 1:
            batch_labels_flip_hor[folder, 0] = 1       
        else:
            batch_labels_flip_hor[folder, int(folder_name[2])] = 1

        # labelling for the horizontal and vertical transformation
        batch_labels_transformaton[folder, int(folder_name[2])] = 1

    if validation_flag:
        batch_data_final = batch_data
        batch_labels_final = batch_labels
    else:
        batch_data_final = np.append(batch_data, batch_data_transformation, axis = 0)
        #batch_data_final = np.append(batch_data_final, batch_data_flip_hor, axis = 0)

        batch_labels_final = np.append(batch_labels, batch_labels_transformaton, axis = 0) 
        #batch_labels_final = np.append(batch_labels_final, batch_labels_flip_hor, axis = 0)
        
    #yield the batch_data and the batch_labels
    #plot_image([resized_image, flipped_image ,trans_image])
    return batch_data_final,batch_labels_final


# In[27]:


# This function is filter out percentage of data for ablation testing
def ablation_folder_list(folder_list, ratio = 0.15):
    final_list = []
    list_buc = [[], [], [], [], []]
    for folder in folder_list:
        val = folder.strip().split(";")
        num = int(val[2])
        list_buc[num].append(folder)

    sample_length = math.floor((ratio * len(folder_list)) / 5)
    for i in list_buc:
        if len(i) > sample_length:
            ran = sample_length
        else:
            ran = len(i)
        final_list += random.sample(i, ran)

    return final_list


# In[28]:


# The main generator function
def generator(source_path, folder_list, batch_size, validation_flag=False, ablation_flag=False):
    print( 'Source path = ', source_path, '; batch size =', batch_size)
    if ablation_flag:
        folder_list = ablation_folder_list(folder_list)
    while True:
        t = np.random.permutation(folder_list)  # shuffling so that the data is fed in random order
        num_batches = len(folder_list)//batch_size
        batch = 0
        for batch in range(num_batches): # we iterate over the number of batches
            yield load_and_process_image(batch_size, source_path, batch, t, validation_flag)
 
        # write the code for the remaining data points which are left after full batches
        if (len(folder_list) != batch_size*num_batches):
            batch_size = len(folder_list) - (batch_size*num_batches)
            yield load_and_process_image(batch_size, source_path, batch, t, validation_flag) 


# Note: a video is represented above in the generator as (number of images, height, width, number of channels).

# In[29]:


curr_dt_time = datetime.datetime.now()
train_path = './Project_data/train'
val_path = './Project_data/val'
num_train_sequences = len(train_doc)
print('# training sequences =', num_train_sequences)
num_val_sequences = len(val_doc)
print('# validation sequences =', num_val_sequences)
num_epochs = 15
print ('# epochs =', num_epochs)


# In[22]:


# Uncomment to test the generator
# train_generator = generator(train_path, train_doc, batch_size)
# next(train_generator)


# In[ ]:





# ## Model
# Here we made the model using different functionalities that Keras provides. We used `Conv3D` and `MaxPooling3D` for the 3D convolution model followed by `TimeDistributed` Conv2D + RNN model. The last layer is the softmax. The network in designed in such a way that the model is able to give good accuracy.

# ### Model using Conv3D and MaxPooling3D

# ### Note:
# The below model was finalised after running many models with many variation.
# Please go through the write-up to know which all models were tried.

# In[34]:


from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

num_filters = [8,16,32,64]
num_dense = [256,128,5]
num_classes = 5

input_shape=(no_of_frames,img_rows,img_cols,img_channels)

# Define model
model = Sequential()

model.add(Conv3D(num_filters[0], kernel_size=(3,3,3), input_shape=input_shape, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling3D(pool_size=(2,2,2)))

model.add(Conv3D(num_filters[1], kernel_size=(3,3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling3D(pool_size=(2,2,2)))

model.add(Conv3D(num_filters[2], kernel_size=(1,3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling3D(pool_size=(2,2,2)))

model.add(Conv3D(num_filters[3], kernel_size=(1,3,3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(MaxPooling3D(pool_size=(2,2,2)))

#Flatten Layers
model.add(Flatten())

model.add(Dense(num_dense[0], activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_dense[1], activation='relu'))
model.add(Dropout(0.5))

#softmax layer
model.add(Dense(num_dense[2], activation='softmax'))


# In[35]:


optimiser = Adam(0.001) # considering the default value of 0.001
model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model.summary())


# ### Note: We have experimented with the learning rate. Both 0.001 and 0.0005 is a good learning rate to start with

# ## Ablation Run

# ### 15 percent of the data for ablation testing
# 

# In[36]:


ablation_num_train_sequences = math.floor(num_train_sequences * 0.15)
ablation_num_val_sequences = math.floor(num_val_sequences * 0.15)

if (ablation_num_train_sequences%batch_size) == 0:
    steps_per_epoch = int(ablation_num_train_sequences/batch_size)
else:
    steps_per_epoch = int(ablation_num_train_sequences/batch_size) + 1

if (ablation_num_val_sequences%batch_size) == 0:
    validation_steps = int(ablation_num_val_sequences/batch_size)
else:
    validation_steps = int(ablation_num_val_sequences/batch_size) + 1

ablation_train_generator = generator(train_path, train_doc, batch_size, False, True)
ablation_val_generator = generator(val_path, val_doc, batch_size, True, True)
model.fit_generator(ablation_train_generator, steps_per_epoch=steps_per_epoch, epochs=5, verbose=1,
          validation_data=ablation_val_generator, validation_steps=validation_steps, 
          class_weight=None, workers=1, initial_epoch=0)


# In[37]:


train_generator = generator(train_path, train_doc, batch_size)
val_generator = generator(val_path, val_doc, batch_size, True)


# In[38]:


model_name = 'model_init' + '_' + str(curr_dt_time).replace(' ','').replace(':','_') + '/'
    
if not os.path.exists(model_name):
    os.mkdir(model_name)
        
filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, cooldown=1, verbose=1)# write the REducelronplateau code here
callbacks_list = [checkpoint, LR]


# The `steps_per_epoch` and `validation_steps` are used by `fit_generator` to decide the number of next() calls it need to make.

# In[39]:


if (num_train_sequences%batch_size) == 0:
    steps_per_epoch = int(num_train_sequences/batch_size)
else:
    steps_per_epoch = (num_train_sequences//batch_size) + 1

if (num_val_sequences%batch_size) == 0:
    validation_steps = int(num_val_sequences/batch_size)
else:
    validation_steps = (num_val_sequences//batch_size) + 1


# Let us now fit the model. This will start training the model and with the help of the checkpoints, we'll be able to save the model at the end of each epoch.

# In[40]:


model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)


# In[ ]:





# ## Model using TimeDistributed Conv2D + RNN 

# In[30]:


from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import LSTM
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation, Dropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


# ### Note:
# The below model was finalised after running many models with many variation.
# Please go through the write-up to know which all models were tried.

# In[78]:


input_shape=(no_of_frames,img_rows,img_cols,img_channels)
num_classes = 5

model = Sequential()
model.add(TimeDistributed(Conv2D(8, (3, 3), padding='same'), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(16, (3,3), padding='same')))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling2D((2, 2))))

model.add(TimeDistributed(Conv2D(32, (3,3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))

model.add(TimeDistributed(Conv2D(128, (3,3), padding='same', activation='relu')))
#model.add(TimeDistributed(Conv2D(128, (3,3), padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))

#model.add(TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu')))
#model.add(TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu')))
#model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

#model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
#model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
#model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Flatten()))

model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=False, dropout=0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# In[79]:


optimiser = Adam()
model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model.summary())


# ## Ablation Run

# ### 15 percent of the data for ablation testing

# In[80]:


ablation_num_train_sequences = math.floor(num_train_sequences * 0.15)
ablation_num_val_sequences = math.floor(num_val_sequences * 0.15)

if (ablation_num_train_sequences%batch_size) == 0:
    steps_per_epoch = int(ablation_num_train_sequences/batch_size)
else:
    steps_per_epoch = int(ablation_num_train_sequences/batch_size) + 1

if (ablation_num_val_sequences%batch_size) == 0:
    validation_steps = int(ablation_num_val_sequences/batch_size)
else:
    validation_steps = int(ablation_num_val_sequences/batch_size) + 1

ablation_train_generator = generator(train_path, train_doc, batch_size, False, True)
ablation_val_generator = generator(val_path, val_doc, batch_size, True, True)
LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, cooldown=1, verbose=1)
callbacks_list = [LR]
model.fit_generator(ablation_train_generator, steps_per_epoch=steps_per_epoch, epochs=10, verbose=1,
          callbacks=callbacks_list, validation_data=ablation_val_generator, validation_steps=validation_steps, 
          class_weight=None, workers=1, initial_epoch=0)


# In[81]:


train_generator = generator(train_path, train_doc, batch_size)
val_generator = generator(val_path, val_doc, batch_size, True)


# In[82]:


model_name = 'model_init_timedistributed' + '_' + str(curr_dt_time).replace(' ','').replace(':','_') + '/'
    
if not os.path.exists(model_name):
    os.mkdir(model_name)
        
filepath = model_name + 'model-timedistributed-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, cooldown=1, verbose=1)
callbacks_list = [checkpoint, LR]


# In[83]:


if (num_train_sequences%batch_size) == 0:
    steps_per_epoch = int(num_train_sequences/batch_size)
else:
    steps_per_epoch = (num_train_sequences//batch_size) + 1

if (num_val_sequences%batch_size) == 0:
    validation_steps = int(num_val_sequences/batch_size)
else:
    validation_steps = (num_val_sequences//batch_size) + 1


# In[84]:


model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)


# In[ ]:




