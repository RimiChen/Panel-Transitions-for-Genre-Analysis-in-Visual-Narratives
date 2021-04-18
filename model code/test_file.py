# %%
#Loading up all libraries
import numpy as np 
import tensorflow as tf
from random import randrange
import os

#### limit memeory before keras was imported

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
     try:
          for gpu in gpus:
               tf.config.experimental.set_memory_growth(gpu, True)

     except RuntimeError as e:
          print(e)

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
record_json = DATA_FOLDER+"/record.json" 
   
start_from = "0_0"
processing_flag = False

   
# number of epochs to train top model  
epochs = 100  
# batch size used by flow_from_directory and predict_generator  
batch_size = 32  

print("============= Load pretrained model ==========")

# %%
#Loading vgc16 model
vgc_16 = applications.VGG16(include_top=False, weights='imagenet')  

# %%
start = datetime.datetime.now()
datagen = ImageDataGenerator(rescale=1. / 255)  


print("============= Create training set ==========")   
generator = datagen.flow_from_directory(  
     train_data_dir,  
     target_size=(img_width, img_height),  
     batch_size=batch_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_train_samples = len(generator.filenames)  
num_classes = len(generator.class_indices)  
   
predict_size_train = int(math.ceil(nb_train_samples / batch_size))  



print("============= train bottleneck_features=========")   
# bottleneck_features_train = vgc_16.predict_generator(generator, predict_size_train)  
bottleneck_features_train = vgc_16.predict(generator, predict_size_train)  
np.save('bottleneck_features_train.npy', bottleneck_features_train)
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)


print("============= Create validation set==========")   

# %%
start = datetime.datetime.now()
generator = datagen.flow_from_directory(  
     validation_data_dir,  
     target_size=(img_width, img_height),  
     batch_size=batch_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_validation_samples = len(generator.filenames)  
   
predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))  
   
bottleneck_features_validation = vgc_16.predict_generator(  
     generator, predict_size_validation)  
   
np.save('bottleneck_features_validation.npy', bottleneck_features_validation) 
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)


print("============= Create test set==========")   

# %%
start = datetime.datetime.now()
generator = datagen.flow_from_directory(  
     test_data_dir,  
     target_size=(img_width, img_height),  
     batch_size=batch_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_test_samples = len(generator.filenames)  
   
predict_size_test = int(math.ceil(nb_test_samples / batch_size))  
   
bottleneck_features_test = vgc_16.predict_generator(  
     generator, predict_size_test)  
   
np.save('bottleneck_features_test.npy', bottleneck_features_test) 
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)

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

# generator_top_2 = datagen_top.flow_from_directory(  
#          train_data_dir_2,  
#          target_size=(img_width, img_height),  
#          batch_size=batch_size,  
#          class_mode='categorical',  
#          shuffle=False)  

   
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
start = datetime.datetime.now()
model = Sequential()


# tensor_1 = tf.convert_to_tensor(layer_1) 
# # tensor_2 = tf.convert_to_tensor(layer_2) 
# print(tensor_1) 
# print(type(tensor_1)) 
# print(tensor_1.shape) 

model.add(Flatten(input_shape=train_data.shape[1:]))


model.add(Dense(256, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(num_classes, activation='sigmoid'))  

model.compile(optimizer='rmsprop',  
          loss='categorical_crossentropy', metrics=['accuracy'])  

print("===================="+ str(validation_data.shape) + ", "+ str(validation_labels.shape))

history = model.fit(train_data, train_labels,  
      epochs=50,  
      batch_size=batch_size,  
      validation_data=(validation_data, validation_labels))  

model.save_weights(top_model_weights_path)  

(eval_loss, eval_accuracy) = model.evaluate(  
 validation_data, validation_labels, batch_size=batch_size, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))  

random_index = str(randrange(65536))


print("[INFO] Loss: {}".format(eval_loss))  
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)





# %%
# plt.figure(1)  

# # summarize history for accuracy  

# plt.subplot(211)  
# plt.plot(history.history['accuracy'])  
# plt.plot(history.history['val_accuracy'])  
# plt.title('model accuracy')  
# plt.ylabel('accuracy')  
# plt.xlabel('epoch')  
# plt.legend(['train', 'test'], loc='upper left')  

# # summarize history for loss  

# plt.subplot(212)  
# plt.plot(history.history['loss'])  
# plt.plot(history.history['val_loss'])  
# plt.title('model loss')  
# plt.ylabel('loss')  
# plt.xlabel('epoch')  
# plt.legend(['train', 'test'], loc='upper left')  
# plt.show()  



########################  Finish training


# %%
#image_array_path = DATA_FOLDER+"/guess/"
image_array_path = TEST_FOLDER

#### load recroded file

# threshold = 30.0

# # magic happens here to make it pretty-printed

# model_acc = {}
# model_acc["acc"] = eval_accuracy * 100

# if float(model_acc["acc"]) < threshold:

#      print("Low accuracy, restart once!")


# else:

random_index = str(randrange(65536))

# json_file = open("accuracy"+"_"+random_index+".json", "w")
# json_file.write(json.dumps(model_acc, indent=4))
# json_file.close() 

load_records = []

with open(record_json) as record_file:
     load_records = json.load(record_file)
     record_file.close()

continue_records = []
[continue_records.append(record) for record in load_records if record not in continue_records] 
# for record in load_records:
#      continue_records.append(record)
     


for (_, folders, _) in walk(image_array_path):
     for single_folder in folders:
          json_name = single_folder+".json"
          ans = []
          #if single_folder == start_from: #or single_folder not in continue_records:
          if single_folder not in continue_records:
               processing_flag = True
               continue_records.append(single_folder)

          else:
               if single_folder in continue_records:
                    print("skip: exist "+single_folder)
               else:
                    continue_records.append(single_folder)
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
                         image = img_to_array(image)  

                         # important! otherwise the predictions will be '0'  
                         image = image / 255  
                         image = np.expand_dims(image, axis=0)  
                         # print(image)


                         # %%
                         # build the VGG16 network  
                         model = applications.VGG16(include_top=False, weights='imagenet')  

                         # get the bottleneck prediction from the pre-trained VGG16 model  
                         bottleneck_prediction = model.predict(image)  
                    
                    
                         # #############  DEBUG
                         # print("bottleneck_prediction = ")
                         # print(bottleneck_prediction)
                         # print(type(bottleneck_prediction))
                         # print(np.shape(bottleneck_prediction))
                         # #############
                         
                         # build top model  
                         model = Sequential()  
                         model.add(Flatten(input_shape=bottleneck_prediction.shape[1:])) 
                         #model.add(Flatten(input_shape=bottleneck_prediction_2.shape[1:]))   
                         model.add(Dense(256, activation='relu'))  
                         model.add(Dropout(0.5))  
                         
                         #### add class number
                         num_classes = 6 
                         model.add(Dense(num_classes, activation='sigmoid'))  

                         model.load_weights(top_model_weights_path)  

                         # use the bottleneck prediction on the top model to get the final classification  
                         class_predicted = model.predict_classes(bottleneck_prediction)
                         prediction_score = model.predict(bottleneck_prediction)
                         #############  DEBUG
                         print("prediction score= ")
                         print(prediction_score)
                         print(type(prediction_score))
                         print(np.shape(prediction_score))
                         #############
                         
                         
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
                         
                         
               json_file = open(JSON_NAME+json_name, "w")
               # magic happens here to make it pretty-printed
               json_file.write(json.dumps(ans, indent=4))
               json_file.close() 
               
               continue_records.append(single_folder)         
               json_file = open(record_json, "w")
               # magic happens here to make it pretty-printed
               json_file.write(json.dumps(continue_records, indent=4))
               json_file.close() 
