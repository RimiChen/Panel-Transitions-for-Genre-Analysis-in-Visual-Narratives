# %%
#Loading up all libraries
import gc
import numpy as np 
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input, Lambda
from keras.layers.merge import Concatenate, Add
from keras import applications  
from keras.utils.np_utils import to_categorical  
import matplotlib.pyplot as plt  
import math  
import cv2  
import datetime
from os import walk
import json
# import torch

DATA_FOLDER = "remove_data"
JSON_NAME = DATA_FOLDER+"/result/"
TEST_FOLDER = DATA_FOLDER+"/guess/"

# %%
# dimensions of our images.  
img_width, img_height = 224, 224  
   
top_model_weights_path = 'bottleneck_fc_model.h5'  
train_data_dir = DATA_FOLDER+"/train/"  
validation_data_dir = DATA_FOLDER+"/validation/"
test_data_dir = DATA_FOLDER +"/test/"  
   


   
# number of epochs to train top model  
epochs = 100  
# batch size used by flow_from_directory and predict_generator  
batch_size = 16  

# print("============= Load pretrained model ==========")

# # %%
# #Loading vgc16 model
# vgc_16 = applications.VGG16(include_top=False, weights='imagenet')  

# # %%
# start = datetime.datetime.now()
# datagen = ImageDataGenerator(rescale=1. / 255)  


# print("============= Create training set ==========")   
# generator = datagen.flow_from_directory(  
#      train_data_dir,  
#      target_size=(img_width, img_height),  
#      batch_size=batch_size,  
#      class_mode=None,  
#      shuffle=False)  
   
# nb_train_samples = len(generator.filenames)  
# num_classes = len(generator.class_indices)  
   
# predict_size_train = int(math.ceil(nb_train_samples / batch_size))  



# print("============= train bottleneck_features=========")   
# bottleneck_features_train = vgc_16.predict_generator(generator, predict_size_train)  
# np.save('bottleneck_features_train.npy', bottleneck_features_train)
# end= datetime.datetime.now()
# elapsed= end-start
# print ('Time: ', elapsed)


# print("============= Create validation set==========")   

# # %%
# start = datetime.datetime.now()
# generator = datagen.flow_from_directory(  
#      validation_data_dir,  
#      target_size=(img_width, img_height),  
#      batch_size=batch_size,  
#      class_mode=None,  
#      shuffle=False)  
   
# nb_validation_samples = len(generator.filenames)  
   
# predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))  
   
# bottleneck_features_validation = vgc_16.predict_generator(  
#      generator, predict_size_validation)  
   
# np.save('bottleneck_features_validation.npy', bottleneck_features_validation) 
# end= datetime.datetime.now()
# elapsed= end-start
# print ('Time: ', elapsed)


# print("============= Create test set==========")   

# # %%
# start = datetime.datetime.now()
# generator = datagen.flow_from_directory(  
#      test_data_dir,  
#      target_size=(img_width, img_height),  
#      batch_size=batch_size,  
#      class_mode=None,  
#      shuffle=False)  
   
# nb_test_samples = len(generator.filenames)  
   
# predict_size_test = int(math.ceil(nb_test_samples / batch_size))  
   
# bottleneck_features_test = vgc_16.predict_generator(  
#      generator, predict_size_test)  
   
# np.save('bottleneck_features_test.npy', bottleneck_features_test) 
# end= datetime.datetime.now()
# elapsed= end-start
# print ('Time: ', elapsed)

# %%
"""
## Start from here
"""

print("############  Start from here")

# %%
start = datetime.datetime.now()
datagen_top = ImageDataGenerator(rescale=1./255)  
generator_top = datagen_top.flow_from_directory(  
         train_data_dir,  
         target_size=(img_width, img_height),  
         batch_size=batch_size,  
         class_mode='categorical',  
         shuffle=False)  

   
nb_train_samples = len(generator_top.filenames)  
num_classes = len(generator_top.class_indices)  
# nb_train_samples_2 = len(generator_top_2.filenames)  
# num_classes_2 = len(generator_top_2.class_indices)  
   
# load the bottleneck features saved earlier  
train_data = np.load('bottleneck_features_train.npy')  
# train_data_2 = np.load('bottleneck_features_train_2.npy')     
# # get the class lebels for the training data, in the original order  
train_labels = generator_top.classes  
   
# convert the training labels to categorical vectors  
train_labels = to_categorical(train_labels, num_classes=num_classes) 
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)

# %%
generator_top = datagen_top.flow_from_directory(  
         validation_data_dir,  
         target_size=(img_width, img_height),  
         batch_size=batch_size,  
         class_mode=None,  
         shuffle=False)  
   
nb_validation_samples = len(generator_top.filenames)  
   
validation_data = np.load('bottleneck_features_validation.npy')  
   

validation_labels = generator_top.classes  
validation_labels = to_categorical(validation_labels, num_classes=num_classes)  
# %%
#image_array_path = DATA_FOLDER+"/guess/"


image_array_path = TEST_FOLDER

model_vgg = applications.VGG16(include_top=False, weights='imagenet')  

start_from = "0_0"
processing_flag = False

for (_, folders, _) in walk(image_array_path):
     for single_folder in folders:
          json_name = str(single_folder)+".json"
          ans = []
          if single_folder == start_from:
               processing_flag = True
          else:
               print("skip: "+single_folder)
          
          if processing_flag == True:
               for (_, _, file_names) in walk(image_array_path+single_folder):
                    # new_path = str(file_names)+ file_postfix
                    for single_file in file_names:
                         # new_path
                         # image_path = image_array_path+single_file
                         image_path = image_array_path+single_folder+"/"+single_file
                         print(image_path)
               
               #image_path = 'data/2_3.jpg'  

                         orig = cv2.imread(image_path)  

                         print("[INFO] loading and preprocessing image...")  
                         image = load_img(image_path, target_size=(224, 224))  
                         image_result = img_to_array(image)  

                         # important! otherwise the predictions will be '0'  
                         image_result = image_result / 255  
                         image_result = np.expand_dims(image_result, axis=0)  
                         # print(image)

                         image.close()
                         
                         # %%
                         # build the VGG16 network  


                         # get the bottleneck prediction from the pre-trained VGG16 model  
                         bottleneck_prediction = model_vgg.predict(image_result)  
                    
                                                            
                         # # build top model  
                         model2 = Sequential()  
                         model2.add(Flatten(input_shape=bottleneck_prediction.shape[1:])) 
                         #model.add(Flatten(input_shape=bottleneck_prediction_2.shape[1:]))   
                         model2.add(Dense(256, activation='relu'))  
                         model2.add(Dropout(0.5))  
                         model2.add(Dense(num_classes, activation='sigmoid'))  

                         model2.load_weights(top_model_weights_path)  

                         # use the bottleneck prediction on the top model to get the final classification  
                         class_predicted = model2.predict_classes(bottleneck_prediction)
                         prediction_score = model2.predict(bottleneck_prediction)
                         # #############  DEBUG
                         # print("prediction score= ")
                         # print(prediction_score)
                         # print(type(prediction_score))
                         # print(np.shape(prediction_score))
                         # #############
                         
                         del model2
                         for i in range(3): gc.collect()
                         
                         # %%
                         inID = class_predicted[0]  
                         
                         #########
                         label_score = prediction_score[0, inID]
                         #########
                         
                         
                         class_dictionary = generator_top.class_indices  

                         inv_map = {v: k for k, v in class_dictionary.items()}  

                         label = inv_map[inID]  

                         # get the prediction label  
                         print("Image name: {}, Image ID: {}, Label: {}, label_score: {} ".format(image_path, inID, label, label_score ))
                         
                         new_image = {}
                         new_image["image_name"] = image_path
                         new_image["label"] = label
                         new_image["score"] = str(label_score)
                         
                         ans.append(new_image)
                         

                         
                         
               json_file = open(JSON_NAME+json_name+".json", "w")
               # magic happens here to make it pretty-printed
               json_file.write(json.dumps(ans, indent=4))
               json_file.close()
          else:
               print("Not here yet")
