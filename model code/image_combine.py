#### this script combine the two images from pre/post folders become 1
from os import walk
from os import path
import os
import json
from PIL import Image, ImageTk

# ANNOTATION_ROOT_FOLDER = "reading_order_annotation/"
# IMAGE_ROOT_FOLDER = "ordered_crop/"
# TARGET_ALL_FOLDER = "all_data/all/"
TARGET_PRE_FOLDER = "remove_data/pre/"
TARGET_POST_FOLDER = "remove_data/post/"
TARGET_PAIR_FOLDER = "remove_data/pair/"


# walk over the folder to get transitions  post/
# loop over the images in a transition folder, for each image, find corresponding one from pre/
# combine the tow images become a larger one

def combine_image(image1_path, image2_path, image1_folder, image2_folder, save_path):
    print("combine: "+image1_path+" , "+image2_path)
    
    #Read the two images
    image1_new_path = image1_folder+image1_path
    image2_new_path = image2_folder+image2_path
    
    image1 = Image.open(image1_new_path)
    # image1.show()
    image2 = Image.open(image2_new_path)
    # image2.show()
    #resize, first image
    image1 = image1.resize((256, 256))
    image2 = image2.resize((256, 256))
    image1_size = image1.size
    image2_size = image2.size
    new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
    new_image.paste(image1,(0,0))
    new_image.paste(image2,(image1_size[0],0))
    new_image.save(save_path +image1_path.replace(".jpg","")+"-"+image2_path.replace(".jpg","")+".jpg","JPEG")
    # new_image.show()    
    

for (_, folders, _) in walk(TARGET_POST_FOLDER):
    # new_path = str(file_names)+ file_postfix
    for single_folder in folders:
        print("===============")
        print(single_folder)
        print("===============")
        
        tranistion_post_folder_path = TARGET_POST_FOLDER+single_folder+"/"
        tranistion_pre_folder_path = TARGET_PRE_FOLDER+single_folder+"/"
        
        target_folder = TARGET_PAIR_FOLDER+single_folder+"/"
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        folder_name_pre = target_folder        
        
        for (_, _, file_names) in walk(tranistion_post_folder_path):
            for single_image in file_names:
                # print(single_image)
                post_image = single_image
                pre_image_string = single_image.replace(".jpg","")
                pre_image_string = pre_image_string.split("_")
                pre_image_name =pre_image_string[0]+"_"+pre_image_string[1]+"_"+str(int(pre_image_string[2])+1)+".jpg"
                
                # if end panel
                print("Pair:" +single_image+" , "+pre_image_name+" , transition: "+single_folder)
                combine_image(single_image, pre_image_name, TARGET_POST_FOLDER+single_folder+"/", TARGET_PRE_FOLDER+single_folder+"/", TARGET_PAIR_FOLDER+single_folder+"/")
 