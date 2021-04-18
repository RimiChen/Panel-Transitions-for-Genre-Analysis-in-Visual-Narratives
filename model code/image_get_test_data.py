#### this script combine the two panels become test data
from os import walk
from os import path
import os
import json
from PIL import Image, ImageTk


TARGET_SOURCE_FOLDER = "remove_data/ordered_crop/"
TARGET_GUESS_FOLDER = "remove_data/guess/"



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

# loop over data folder



for (_, folders, _) in walk(TARGET_SOURCE_FOLDER):
    for single_folder in folders:
        print("=================")
        print(single_folder)
        print("=================")

# for each folder, make panels in a page in one folder
        image_folder_path = TARGET_SOURCE_FOLDER+single_folder+"/"
        
        book_folder = TARGET_GUESS_FOLDER+single_folder+"/"
        # if not os.path.exists(book_folder):
        #     os.makedirs(book_folder)
        
        page_count = 0
        panel_count = 0        
        for (_, _, file_names) in walk(image_folder_path):
            #print(file_names)
            for single_image in file_names:
                if ".jpg" in single_image: 
                    file_name_string = single_image.replace(".jpg","")
                    file_name_string = file_name_string.split("_")
                    # print(file_name_string)
                    current_page = int(file_name_string[0])
                    current_panel = int(file_name_string[1])
                    
                    #panel_count = panel_count + 1        
                    if  current_panel < panel_count:
                        # new page 
                        page_count = page_count +1
                    else:
                        print("in same page")
                        #panel_count = 0
                    panel_count = current_panel
                    
                    book_page_folder_path = TARGET_GUESS_FOLDER+str(single_folder)+"_"+str(page_count)+"/"
                    if not os.path.exists(book_page_folder_path):
                        os.makedirs(book_page_folder_path)
                    
                    cur_image = str(page_count)+"_"+str(panel_count)+".jpg"
                    next_image= str(page_count)+"_"+str(panel_count+1)+".jpg"
                    cur_image_path = TARGET_SOURCE_FOLDER+str(single_folder)+"/"+cur_image
                    next_image_path = TARGET_SOURCE_FOLDER+str(single_folder)+"/"+next_image
                    print(cur_image_path+"   ,   "+next_image_path)
                    if os.path.exists(cur_image_path) and os.path.exists(next_image_path):
                        # both exist combine them
                        combine_image(
                            cur_image,
                            next_image,
                            TARGET_SOURCE_FOLDER+str(single_folder)+"/",
                            TARGET_SOURCE_FOLDER+str(single_folder)+"/",
                            book_page_folder_path
                            )
                    # panel_count = panel_count + 1
                

            
