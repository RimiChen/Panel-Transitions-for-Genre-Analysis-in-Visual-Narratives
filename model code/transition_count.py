# load the annotation files in a folder and count the labels

from os import walk
import json

DATA_FOLDER = "remove_data"
RESULT_PATH = DATA_FOLDER+"/old_result/"
TARGET_PATH = DATA_FOLDER+"/old_count/"


transition_count = {}

for (_, folders, _) in walk(RESULT_PATH):
    for single_folder in folders:
        count_target = RESULT_PATH +single_folder
        for (_, _, file_names) in walk(count_target):
            for single_file in file_names:
                #print(single_file)
                with open(RESULT_PATH+single_folder+"/"+single_file) as json_file:
                    annotations = json.load(json_file)
                    json_file.close()
                    
                    for record in annotations:
                        label = record["label"]
                        if label not in transition_count:
                            transition_count[label] = 1
                        else:
                            transition_count[label] = transition_count[label] + 1
                            

        json_file = open(TARGET_PATH+"count_result_"+single_folder.replace("/","")+".json", "w")
        # magic happens here to make it pretty-printed
        json_file.write(json.dumps(transition_count, indent=4))
        json_file.close() 