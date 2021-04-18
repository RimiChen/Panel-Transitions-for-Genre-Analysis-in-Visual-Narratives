# %%
#Loading up all libraries
import numpy as np 
import tensorflow as tf
from random import randrange
import os
import os.path
from os import path
import random
from random import choices
import glob
import shutil

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
from keras.utils import np_utils
from keras.models import load_model

import matplotlib.pyplot as plt  
import matplotlib.image as mpimg

from PIL import Image
import subprocess
import math  
import cv2  
import datetime
from os import walk
import json
# import torch

DATA_FOLDER = "remove_data"
JSON_NAME = DATA_FOLDER+"/result/"
TEST_FOLDER = DATA_FOLDER+"/guess/"
EPOCHS = 10

# %%
# dimensions of our images.  
img_width, img_height = 224, 224  
   
top_model_weights_path = 'bottleneck_fc_model.h5'
top_model_path = 'half_model.h5'    
train_data_dir = DATA_FOLDER+"/train/"  
validation_data_dir = DATA_FOLDER+"/validation/"
test_data_dir = DATA_FOLDER +"/test/"  
record_json = DATA_FOLDER+"/record.json" 
   
start_from = "0_0"
processing_flag = False

   
# number of epochs to train top model  
epochs = EPOCHS
# batch size used by flow_from_directory and predict_generator  
batch_size = 32  



def split_images(train_propotion, validation_propotion, data_root, pool_root, target_folder):
     
     target_pool = pool_root + target_folder

     image_pool = []
     for (_, _, file_names) in walk(target_pool):
          for single_image_file in file_names:
               print("target image = "+single_image_file)
               image_pool.append(single_image_file)
               
               # image_path  = target_pool+single_image_file
               # image_pool.append(image_path)

     # train_images = []
     # vaildation_images = []
     # test_images = []

     train_number = math.floor(train_propotion*len(image_pool))
     validation_number = math.floor(validation_propotion*len(image_pool))

     chosen_train_list = choices(image_pool, k=train_number)
     new_image_pool = list(set(image_pool) - set(chosen_train_list)) 

     chosen_validation_list = choices(new_image_pool, k=validation_number)

     chosen_test_list = list(set(new_image_pool) - set(chosen_validation_list))
     
     
     
     if not os.path.exists(data_root+"/train/"+target_folder):
         os.makedirs(data_root+"/train/"+target_folder)

     if not os.path.exists(data_root+"/test/"+target_folder):
         os.makedirs(data_root+"/test/"+target_folder)

     if not os.path.exists(data_root+"/validation/"+target_folder):
         os.makedirs(data_root+"/validation/"+target_folder)



     for single_image in chosen_train_list:
          old_image_path = pool_root + target_folder + single_image
          new_image_path = data_root+"/train/"+target_folder+single_image
          print("copy \""+old_image_path+"\" to \""+new_image_path+"\"")
          img = Image.open(old_image_path)
          img.save(new_image_path, 'JPEG')

     for single_image in chosen_validation_list:
          old_image_path = pool_root + target_folder +single_image
          new_image_path = data_root+"/validation/"+target_folder+single_image
          print("copy \""+old_image_path+"\" to \""+new_image_path+"\"")

          img = Image.open(old_image_path)
          img.save(new_image_path, 'JPEG')

     for single_image in chosen_test_list:
          old_image_path = pool_root + target_folder + single_image
          new_image_path = data_root+"/test/"+target_folder+single_image
          print("copy \""+old_image_path+"\" to \""+new_image_path+"\"")

          img = Image.open(old_image_path)
          img.save(new_image_path, 'JPEG')



def copy_image_to_folder(old_path, new_path):
     for (_, _, file_names) in walk(old_path):
          for single_image_file in file_names:
               image_path  = old_path+single_image_file
               
               img = Image.open(image_path)
               img.save(new_path+single_image_file, 'JPEG')


def give_feedback(image_path, image_transition):
     #feed = random.randint(0, 1)
     # open image with transition
     # get answer
     feed = 0
     print("=============================")
     print(image_path)

     # p = subprocess.Popen('python show_image.py'+" "+image_path)
     # # p = subprocess.Popen(["display", "/"+image_path])
     # feed = input("Is "+image_transition+" correct? 0 ~6:")
     # p.kill()

     fig=plt.figure()
 
     fig.set_tight_layout(True)
     plt.ion()
     image = mpimg.imread(image_path)
     imgplot = plt.imshow(image)
     plt.show(block=False)
     feed = input("Is "+image_transition+" correct? 0 ~6:")
     
     
     # if int(feed) > 6:
     #      print("not correct feedback")
     # el


     # feed = input("write your grade: ")
     while True:
          try:
               num = int(feed)
               break
          except:
               feed = input("Is "+image_transition+" correct? 0 ~6:")

          

     if int(feed) == 0:
          print("Wrong prediction!")

     plt.close()

     return feed

def get_feedback(image_path, image_transition):
     temp_feedback  = {}
     
     temp_feedback["image_path"] = image_path
     temp_feedback["transition"] = image_transition
     temp_feedback["feedback"] = give_feedback(image_path, image_transition)

     return temp_feedback



def predict_image(image_path):
     # image_path = image_array_path+single_folder+"/"+single_file
     # print(image_path)

     ans = []

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
     model.add(Dense(num_classes, activation='sigmoid'))  

     model.load_weights(top_model_weights_path)  

     # use the bottleneck prediction on the top model to get the final classification  
     class_predicted = model.predict_classes(bottleneck_prediction)
     prediction_score = model.predict(bottleneck_prediction)
     #############  DEBUG
     # print("prediction score= ")
     # print(prediction_score)
     # print(type(prediction_score))
     # print(np.shape(prediction_score))
     # #############
     
     
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
     
     # ans.append(new_image)

     return new_image     


print("============= Load pretrained model ==========")

# %%
#Loading vgc16 model
vgc_16 = applications.VGG16(include_top=False, weights='imagenet')  

# %%
start = datetime.datetime.now()
datagen = ImageDataGenerator(rescale=1. / 255)  



model_acc = {}

print("============= Create training set ==========")   
generator = datagen.flow_from_directory(  
     train_data_dir,  
     target_size=(img_width, img_height),  
     batch_size=batch_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_train_samples = len(generator.filenames)  
num_classes = len(generator.class_indices)  

model_acc["train"] = nb_train_samples

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
model_acc["validation"] =  nb_validation_samples

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
model_acc["test"] =  nb_test_samples
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

#### if weights already exist, load and keep training
if path.exists(top_model_path):
    print("SYSTEM: load exsiting model")
    
    model = load_model(top_model_path) 
else:
    print("SYSTEM: create new model")

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
      epochs=EPOCHS,  
      batch_size=batch_size,  
      validation_data=(validation_data, validation_labels))  

model.save_weights(top_model_weights_path)  
model.save(top_model_path)


(eval_loss, eval_accuracy) = model.evaluate(  
 validation_data, validation_labels, batch_size=batch_size, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))  

random_index = str(randrange(65536))


print("[INFO] Loss: {}".format(eval_loss))  
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)



model_acc["acc"] = eval_accuracy * 100

random_index = str(randrange(65536))

json_file = open("accuracy"+"_"+random_index+".json", "w")
json_file.write(json.dumps(model_acc, indent=4))
json_file.close() 


#### predict 100 images, and write the results to json file
# take 10 folders from ./guess
folder_number = 10
test_image_root = TEST_FOLDER

# DATA_FOLDER


folder_list = []
chosen_list = []

for (_, folders, _) in walk(TEST_FOLDER):
     for single_folder in folders:
          folder_list.append(TEST_FOLDER+single_folder+"/")


chosen_list = choices(folder_list, k=folder_number)

print("\n".join(chosen_list))
     

# copy images (and rename) from the selected folders to ./feedback
FEEDBACK_FOLDER = DATA_FOLDER+"/feedback/"

new_test_image_list = []


for single_folder in chosen_list:
     for (_, _, file_names) in walk(single_folder):
          for single_image in file_names:
               # print(single_image)
               
               book_page = single_folder.replace(TEST_FOLDER,"")
               book_page = book_page.replace("/","")

               new_image_name =book_page+"_"+single_image



               old_image_path = single_folder+single_image
               new_image_path = FEEDBACK_FOLDER+new_image_name
               new_test_image_list.append(new_image_path)

               img = Image.open(old_image_path)
               img.save(new_image_path, 'JPEG')


# print("\n".join(new_test_image_list))



# predict ./feedback

answers = []
for (_, _, file_names) in walk(FEEDBACK_FOLDER):
     for single_image_file in file_names:
          image_path  = FEEDBACK_FOLDER+single_image_file
          
          ans = predict_image(image_path)
          answers.append(ans)


print(json.dumps(answers, indent=4))


feedbacks = []
for ans in answers:
     feedback_result = get_feedback(ans[ "image_name"], ans["label"])
     feedbacks.append(feedback_result)



json_file = open(DATA_FOLDER+"/"+"temp_feedback"+random_index+".json", "w")
# magic happens here to make it pretty-printed
json_file.write(json.dumps(feedbacks, indent=4))
json_file.close() 

shutil.rmtree(FEEDBACK_FOLDER, ignore_errors=True)

if not os.path.exists(FEEDBACK_FOLDER):
    os.makedirs(FEEDBACK_FOLDER)

# # clean ./feedback
# # copy image in .test/ ./train ./ vaildation to pool

# # train_data_dir = DATA_FOLDER+"/train/"  
# # validation_data_dir = DATA_FOLDER+"/validation/"
# # test_data_dir = DATA_FOLDER +"/test/"  
# # record_json = DATA_FOLDER+"/record.json" 


# POOL_PATH = DATA_FOLDER+"/pool/"
# # create folders

# ACTION_PATH = POOL_PATH+"Action/"
# if not os.path.exists(ACTION_PATH):
#     os.makedirs(ACTION_PATH)

# ASPECT_PATH = POOL_PATH+"Aspect/"    
# if not os.path.exists(ASPECT_PATH):
#     os.makedirs(ASPECT_PATH)

# MOMENT_PATH = POOL_PATH+"Moment/"
# if not os.path.exists(MOMENT_PATH):
#     os.makedirs(MOMENT_PATH)

# NON_PATH = POOL_PATH+"Non_sequitur/"
# if not os.path.exists(NON_PATH):
#     os.makedirs(NON_PATH)

# SCENE_PATH = POOL_PATH+"Scene/"
# if not os.path.exists(SCENE_PATH):
#     os.makedirs(SCENE_PATH)

# SUBJECT_PATH = POOL_PATH+"Subject/"
# if not os.path.exists(SUBJECT_PATH):
#     os.makedirs(SUBJECT_PATH)

# for (_, _, file_names) in walk(FEEDBACK_FOLDER):
#      for single_image_file in file_names:
#           image_path  = FEEDBACK_FOLDER+single_image_file
          
#           ans = predict_image(image_path)
#           answers.append(ans)


# copy_image_to_folder(train_data_dir+"Action/", ACTION_PATH)
# copy_image_to_folder(validation_data_dir+"Action/", ACTION_PATH)
# copy_image_to_folder(test_data_dir+"Action/", ACTION_PATH)

# copy_image_to_folder(train_data_dir+"Aspect/", ASPECT_PATH)
# copy_image_to_folder(validation_data_dir+"Aspect/", ASPECT_PATH)
# copy_image_to_folder(test_data_dir+"Aspect/", ASPECT_PATH)

# copy_image_to_folder(train_data_dir+"Moment/", MOMENT_PATH)
# copy_image_to_folder(validation_data_dir+"Moment/", MOMENT_PATH)
# copy_image_to_folder(test_data_dir+"Moment/", MOMENT_PATH)

# copy_image_to_folder(train_data_dir+"Non_sequitur/", NON_PATH)
# copy_image_to_folder(validation_data_dir+"Non_sequitur/", NON_PATH)
# copy_image_to_folder(test_data_dir+"Non_sequitur/", NON_PATH)

# copy_image_to_folder(train_data_dir+"Scene/", SCENE_PATH)
# copy_image_to_folder(validation_data_dir+"Scene/", SCENE_PATH)
# copy_image_to_folder(test_data_dir+"Scene/", SCENE_PATH)

# copy_image_to_folder(train_data_dir+"Subject/", SUBJECT_PATH)
# copy_image_to_folder(validation_data_dir+"Subject/", SUBJECT_PATH)
# copy_image_to_folder(test_data_dir+"Subject/", SUBJECT_PATH)



# ## if feedback is correct, copy to pool
# for reviewed_image in feedbacks:
#      if reviewed_image["feedback"] != 0:
#           old_path = reviewed_image["image_path"]
#           image_name = old_path.replace("remove_data/feedback/","")
#           #### first version, right or wrong

# # TRANSITION_MAPPING["Action"] = 1.0
# # TRANSITION_MAPPING["Aspect"] = 2.0
# # TRANSITION_MAPPING["Subject"] = 3.0
# # TRANSITION_MAPPING["Scene"] = 4.0
# # TRANSITION_MAPPING["Moment"] = 5.0
# # TRANSITION_MAPPING["Non_sequitur"] = 6.0


#           if int(reviewed_image["feedback"]) == 1:
#                new_path = POOL_PATH+"Action/"+image_name
#           elif int(reviewed_image["feedback"]) == 2:
#                new_path = POOL_PATH+"Aspect/"+image_name
#           elif int(reviewed_image["feedback"]) == 3:
#                new_path = POOL_PATH+"Subject/"+image_name
#           elif int(reviewed_image["feedback"]) == 4:
#                new_path = POOL_PATH+"Scene/"+image_name
#           elif int(reviewed_image["feedback"]) == 5:
#                new_path = POOL_PATH+"Moment/"+image_name
#           elif int(reviewed_image["feedback"]) == 6:
#                new_path = POOL_PATH+"Non_sequitur/"+image_name


#           #### TODO: second version, assigned feeadback
#           img = Image.open(old_path)
#           img.save(new_path, 'JPEG')


# # clean feedback
# # files = glob.glob(FEEDBACK_FOLDER)

# # for f in files:
# #      print(f)
# #      os.chmod(f, 0o777)
# #      os.remove(f)

# # no permission, so delete by hand

# # clean ./test
# # clean ./train
# # clean ./validation

# #### matching the results, if prediction matchs human annotation, add these to 3 folders
# # looping the image and show the prediction to retain feedback, if yes go ./correct, else go ./wrong
# # split ./correct to train, test, evaluate


# # ASPECT_PATH = POOL_PATH+"Aspect/"  
# #   
# # os.rmdir(FEEDBACK_FOLDER)
# # os.rmdir(DATA_FOLDER+"/test/")
# # os.rmdir(DATA_FOLDER+"/train/")
# # os.rmdir(DATA_FOLDER+"/validation/")

# shutil.rmtree(FEEDBACK_FOLDER, ignore_errors=True)

# if not os.path.exists(FEEDBACK_FOLDER):
#     os.makedirs(FEEDBACK_FOLDER)
# if not os.path.exists(DATA_FOLDER+"/test/"):
#     os.makedirs(DATA_FOLDER+"/test/")
# if not os.path.exists(DATA_FOLDER+"/train/"):
#     os.makedirs(DATA_FOLDER+"/train/")
# if not os.path.exists(DATA_FOLDER+"/validation/"):
#     os.makedirs(DATA_FOLDER+"/validation/")


# train_pro = 0.8
# validation_pro = 0.1

# split_images(train_pro, validation_pro, DATA_FOLDER, POOL_PATH, "Action/")
# split_images(train_pro, validation_pro, DATA_FOLDER, POOL_PATH, "Aspect/")
# split_images(train_pro, validation_pro, DATA_FOLDER, POOL_PATH, "Moment/")
# split_images(train_pro, validation_pro, DATA_FOLDER, POOL_PATH, "Non_sequitur/")
# split_images(train_pro, validation_pro, DATA_FOLDER, POOL_PATH, "Scene/")
# split_images(train_pro, validation_pro, DATA_FOLDER, POOL_PATH, "Subject/")



