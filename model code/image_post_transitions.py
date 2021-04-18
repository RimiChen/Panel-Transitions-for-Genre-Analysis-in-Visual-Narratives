#### this script separate the images according to transition labels after it

from os import walk
from os import path
import os
import json
from PIL import Image, ImageTk

# set the root folder
# looping over the annotation files
# if the transition is not recorded, pass this part
# else, save the image to corresponding folders
# whilee separating rename the files to make them match
# separete images to test, train, vaildation

TEST_RATE = 0.1
VALIDATION_RATE = 0.1
TRAIN_RATE = 1 - (TEST_RATE +VALIDATION_RATE)

ANNOTATION_ROOT_FOLDER = "reading_order_annotation_remove/"
IMAGE_ROOT_FOLDER = "remove_data/ordered_crop/"
TARGET_ALL_FOLDER = "remove_data/all/"
TARGET_PRE_FOLDER = "remove_data/pre/"
TARGET_POST_FOLDER = "remove_data/post/"

def get_book_name(book_path, postfix):
    book_name_string = book_path.split("/")
    book_name = book_name_string[-1].replace(postfix, "")
    
    return book_name
def crop_image(image_obj):
    
    # Opens a image in RGB mode 
    #im = Image.open(r"C:\Users\Admin\Pictures\geeks.png") 
    im = image_obj
    
    # Size of the image in pixels (size of orginal image) 
    # (This is not mandatory) 
    width, height = im.size 
    
    # Setting the points for cropped image 
    left = 0
    top = 0
    right = im.size[0]
    bottom = im.size[1]
    
    # Cropped image of above dimension 
    # (It will not change orginal image) 
    im_crop = im.crop((left, top, right, bottom)) 
    
    # Shows the image in image viewer 
    #im_crop.show()
    
    return im_crop

def load_panel_transitions(annotation_path, file_postfix):
    panel_array = []
    ## structure
    panel = {}
    panel["image"] = ""
    panel["Transitions"] =[]
    panel["path"] = ""
    
    #### set label sets in here
    Labels = [
        "ACTION",
        "ASPECT",
        "MOMENT",
        "SUBJECT",
        "SCENE",
        "NON"
    ]
    
    #### create dictionary to contain image paths (for grouping the data)
    Transitions = {}
    for label in Labels:
        if label not in Transitions:
            #### add path in here after
            Transitions[label] = []
    
    annotation_path_array = []
    for (_, _, file_names) in walk(annotation_path):
        # new_path = str(file_names)+ file_postfix
        for single_file in file_names:
            # new_path
            new_path = annotation_path+single_file
            # print(new_path)
            annotation_path_array.append(new_path)

    # book_count = 0
    for annotation_path in annotation_path_array:
        print("SYSTEM: processing \'"+annotation_path+"\'")
        book_name = get_book_name(annotation_path, file_postfix)
        
        
        annotation = []
        with open(annotation_path) as json_file:
            annotations = json.load(json_file)
            json_file.close()
            
        # rename image 
        # save to folder
        # print(annotations)
        # print(type(annotations))
        page_count = 0
        for page in annotations["pages"]:
            # print(page)
            # print(annotations["pages"][page])
            target = annotations["pages"][page]

            if target["panel_number"] > 0:
                # more than one panel
                transitions_of_page = target["transitions"]
                if len(transitions_of_page) > 0:
                    # this part have annotation
                    # print("has annotaiton")
                    transition_count = 0
                    for annotation_transition in transitions_of_page:
# TARGET_ALL_FOLDER = "all_data/all/"
# TARGET_PRE_FOLDER = "all_data/pre/"
# TARGET_POST_FOLDER = "all_data/post/"
                        folder_name_all = TARGET_ALL_FOLDER+annotation_transition
                        if not os.path.exists(folder_name_all):
                            os.makedirs(folder_name_all)
                        folder_name_pre = TARGET_PRE_FOLDER+annotation_transition
                        if not os.path.exists(folder_name_pre):
                            os.makedirs(folder_name_pre)
                        folder_name_post = TARGET_POST_FOLDER+annotation_transition
                        if not os.path.exists(folder_name_post):
                            os.makedirs(folder_name_post)



                        image_index_1 =transition_count
                        image_index_2 =transition_count + 1
                        
                        transition_count = transition_count + 1 
                        
                        image_1_file = str(book_name) +"_"+ str(page_count) +"_"+str(image_index_1)+ ".jpg"
                        image_2_file = str(book_name) +"_"+ str(page_count) +"_"+str(image_index_2)+ ".jpg"
                        

                        
                        ori_image_name_1 = str(page_count) +"_"+str(image_index_1)+ ".jpg"
                        ori_image_name_2 = str(page_count) +"_"+str(image_index_2)+ ".jpg"
                        
                        image_file_1 = IMAGE_ROOT_FOLDER+str(book_name)+"/"+ori_image_name_1
                        image_file_2 = IMAGE_ROOT_FOLDER+str(book_name)+"/"+ori_image_name_2
                        if path.exists(image_file_1) and path.exists(image_file_2):
                            print(image_1_file +"--"+str(annotation_transition)+"--"+image_2_file)
                            image_obj = Image.open(IMAGE_ROOT_FOLDER+str(book_name)+"/"+ori_image_name_1)
                            new_image = crop_image(image_obj)
                            new_image.save(TARGET_ALL_FOLDER+annotation_transition+"/"+image_1_file) 
                            new_image.save(TARGET_POST_FOLDER+annotation_transition+"/"+image_1_file) 

                            image_obj = Image.open(IMAGE_ROOT_FOLDER+str(book_name)+"/"+ori_image_name_2)
                            new_image = crop_image(image_obj)
                            new_image.save(TARGET_ALL_FOLDER+annotation_transition+"/"+image_2_file)                         
                            new_image.save(TARGET_PRE_FOLDER+annotation_transition+"/"+image_2_file)                         

                page_count = page_count + 1
            
if __name__ == "__main__":
    load_panel_transitions(ANNOTATION_ROOT_FOLDER, "_order.json")