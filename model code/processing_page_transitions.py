#### this script is used to parse the transition sequence from pages
# read from recored json files
# the panels orders are axises, values represents the transitions
# make them become vectors, normalized with same dimentions

### map the transitions to values
from sklearn.cluster import KMeans
from os import walk
import json
import numpy as np
import random

max_dimension = 15

## load transition labels
# looping book folders
# looping pages in a book
# load transitions
# map transitions to number
# normalize and add zero to make the vector 1 X max_dimention matrix
# save page to dictionary
###
# book_number:{
#   "#page":[[transition 1, transition 2, transition 3, .....]]
# }
###

NORMALIZE_FLAG = False

FILE_POSTFIX = "_original_latest"

TRANSITION_MAPPING = {}
TRANSITION_MAPPING["Action"] = 1.0
TRANSITION_MAPPING["Aspect"] = 2.0
TRANSITION_MAPPING["Subject"] = 3.0
TRANSITION_MAPPING["Scene"] = 4.0
TRANSITION_MAPPING["Moment"] = 5.0
TRANSITION_MAPPING["Non_sequitur"] = 6.0


# 20210410 ROOT_FOLDER = "./remove_data/old_result/"
ROOT_FOLDER = "./remove_data/new_result/"

narrative_seuqneces = {}

def mapping(transition_label):
    transition_value = TRANSITION_MAPPING[transition_label]
    return transition_value


def normalize_sequence(max_dim, input_vector):
    result_vector = input_vector

    for i in range(max_dim):
        if len(result_vector) < max_dim:
            result_vector.append(0.0)

    return result_vector



### launch the main process
if __name__ == "__main__":
   
    for (_, folders, _) in walk(ROOT_FOLDER):
        for single_folder in folders:
            # print(single_folder)
            page_folder = ROOT_FOLDER+str(single_folder)+"/"

            # inital the dictionary
            narrative_seuqneces[single_folder] = {}

            for (_, _, target_folders) in walk(page_folder):
                for page in target_folders:
                    # print("========================================")
                    # print(page)

                    # json_file_name = page.replace(".json.json",".json")
                    json_file_name = page
                    page_index = page.replace(".json.json", "")
                    # print(json_file_name)
                    json_file_path = page_folder+"/"+json_file_name

                    with open(json_file_path) as record_file:
                        load_records = json.load(record_file)
                        record_file.close()
                    

                    ## a page
                    page_sequence = []
                    for item in load_records:
                        page_sequence.append(mapping(item["label"]))

                    if NORMALIZE_FLAG == True:

                        normalized_sequence = normalize_sequence(max_dimension, page_sequence)
                        narrative_seuqneces[single_folder][page_index] = normalized_sequence

                    else:
                        narrative_seuqneces[single_folder][page_index] = page_sequence
                    # print(narrative_seuqneces[single_folder][page])
            
    ### save to json file
    json_file = open("narrative_sequecce"+FILE_POSTFIX, "w")
    # magic happens here to make it pretty-printed
    json_file.write(json.dumps(narrative_seuqneces, indent=4))
    json_file.close() 

    #### test data
    # X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    
    data_list = []
    data_index = {}
    book_sequence_count = {}

    for book in narrative_seuqneces:
        book_sequence_count[book] = {}
        for page_sequence in narrative_seuqneces[book]:

            target_vector = narrative_seuqneces[book][page_sequence]

            if str(target_vector) not in book_sequence_count[book]:
                book_sequence_count[book][str(target_vector)] = 1
            else:
                book_sequence_count[book][str(target_vector)] = book_sequence_count[book][str(target_vector)] + 1
            
            data_list.append(target_vector)
            data_index[len(data_list) -1] = page_sequence
            # print(target_vector)



  


    json_file = open("narrative_sequecce_index"+FILE_POSTFIX, "w")
    # magic happens here to make it pretty-printed
    json_file.write(json.dumps(data_index, indent=4))
    json_file.close() 

                                
    json_file = open("narrative_sequecce_count"+FILE_POSTFIX, "w")
    # magic happens here to make it pretty-printed
    json_file.write(json.dumps(book_sequence_count, indent=4))
    json_file.close() 



    if NORMALIZE_FLAG == True:

        ### K means
        
        num_cluster_centroid = 12

        data_for_clustering = np.array(data_list)
        kmeans = KMeans(n_clusters=num_cluster_centroid, random_state=0).fit(data_for_clustering)
        # print(kmeans.labels_)

        cluster_result_list = list(kmeans.labels_)

        index_count = 0
        cluster_results_dictionary = {}  
        for list_index in cluster_result_list:
            page_key = data_index[index_count]
            index_count = index_count +1

            # print(list_index)

            cluster_results_dictionary[page_key] = str(list_index)


        print(cluster_results_dictionary)


        json_file = open("narrative_sequecce_cluster"+FILE_POSTFIX, "w")
        # magic happens here to make it pretty-printed
        json_file.write(json.dumps(cluster_results_dictionary, indent=4))
        json_file.close() 



