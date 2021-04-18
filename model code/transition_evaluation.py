#### this script is used to compare the similarity bwtween two sets of annotations

### map the transitions to values
from sklearn.cluster import KMeans
from os import walk
import json
import numpy as np
import random
from collections import Counter
import math
from sklearn.metrics import cohen_kappa_score


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

FILE_POSTFIX = "_comparison"

TRANSITION_MAPPING = {}
TRANSITION_MAPPING["Action"] = 1.0
TRANSITION_MAPPING["Aspect"] = 2.0
TRANSITION_MAPPING["Subject"] = 3.0
TRANSITION_MAPPING["Scene"] = 4.0
TRANSITION_MAPPING["Moment"] = 5.0
TRANSITION_MAPPING["Non_sequitur"] = 6.0
TRANSITION_MAPPING["Other"] = 6.0


ROOT_FOLDER = "./remove_data/ground_truth_comparison/"

narrative_seuqneces = {}



#### cos distance : https://stackoverflow.com/questions/14720324/compute-the-similarity-between-two-lists/14720386
def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)


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
   
    # for (_, folders, _) in walk(ROOT_FOLDER):
    #     for single_folder in folders:
    #         # print(single_folder)
    #         page_folder = ROOT_FOLDER+str(single_folder)+"/"

    # inital the dictionary


    # narrative_seuqneces[single_folder] = {}
    page_folder = ROOT_FOLDER
    load_records = {}

    for (_, _, target_folders) in walk(page_folder):
        narrative_seuqneces = {}


        # load_records["eval"] = []
        # load_records["original"] = []

        for page in target_folders:
            # print("========================================")
            print(page)

            # json_file_name = page.replace(".json.json",".json")
            json_file_name = page
            # page_index = page.replace(".json.json", "")
            page_index = str(page)
            # print(json_file_name)
            json_file_path = page_folder+"/"+json_file_name

            with open(json_file_path) as record_file:

                if json_file_path.find("eval") >= 0:
                    load_records["eval"] = json.load(record_file)
                else:
                    load_records["original"] = json.load(record_file)
                
                record_file.close()
            

            ## a page
    page_sequence = {}
    page_sequence["eval"] = []
    page_sequence["original"] =[]
    map_sequence = {}
    map_sequence["eval"] = []
    map_sequence["original"] =[]

    transition_count = {}
    transition_count["eval"] = {}
    transition_count["original"] = {}


    
    
    similar_count = {}


    for item in load_records["eval"]:
        print(item)

        if item == "pages":
            # print(item)
            # print(load_records[item])

            for page in load_records["eval"][item]:
                if len(load_records["eval"][item][page]["transitions"]) != 0:
                    # print("============================================")
                    # print(load_records["eval"][item][page]["transitions"]) 
                    # print(load_records["original"][item][page]["transitions"])

                    # assume every page has same length
                    if len(load_records["eval"][item][page]["transitions"]) == len(load_records["original"][item][page]["transitions"]):
                        # print("============================================")
                        # print(load_records["eval"][item][page]["transitions"]) 
                        # print(load_records["original"][item][page]["transitions"])
                        # evaluation set
                        for eval_transition in load_records["eval"][item][page]["transitions"]:
                            # print(eval_transition)
                            page_sequence["eval"].append(eval_transition)
                            map_sequence["eval"].append(mapping(eval_transition))
                            
                            if eval_transition not in transition_count["eval"]:
                                transition_count["eval"][eval_transition] = 1
                            else:
                                transition_count["eval"][eval_transition] = transition_count["eval"][eval_transition] + 1




                        # original set
                        for orig_transition in load_records["original"][item][page]["transitions"]:
                            # print(eval_transition)
                            page_sequence["original"].append(orig_transition)
                            map_sequence["original"].append(mapping(orig_transition))


                            if orig_transition == "Other":
                                temp_key = "Non_sequitur"

                                if temp_key not in transition_count["original"]:
                                    transition_count["original"][temp_key] = 1
                                else:
                                    transition_count["original"][temp_key] = transition_count["original"][temp_key] + 1                            
                            else:
                                if orig_transition not in transition_count["original"]:
                                    transition_count["original"][orig_transition] = 1
                                else:
                                    transition_count["original"][orig_transition] = transition_count["original"][orig_transition] + 1
                        




    #### count similarity
    #
    anno_1_name = "2"
    anno_2_name = "3"
    annotator1 = map_sequence["eval"]
    annotator2 = map_sequence["original"]
    
    score = cohen_kappa_score(annotator1, annotator2)

    print('Cohen\'s Kappa:',score)

    record_kappa = {}
    record_kappa["annotator_"+anno_1_name] = annotator1
    record_kappa["annotator_"+anno_2_name] = annotator2
    record_kappa["kappa"] = score


    json_file = open("annotation_kappa"+anno_1_name+"_"+anno_2_name, "w")
    # magic happens here to make it pretty-printed
    json_file.write(json.dumps(record_kappa, indent=4))
    json_file.close() 


    # Method compare the difference count
    # Cos distance
    counter_eval = Counter(page_sequence["eval"])
    counter_original = Counter(page_sequence["original"])
    print(counter_eval)
    print(counter_original)


    diff_dictionary = {}
    diff_dictionary["count"] = 0
    
    diff_count = 0
    for transition in map_sequence["eval"]:
        if  float(map_sequence["eval"][diff_count]) == float(map_sequence["original"][diff_count]):
            diff_dictionary["count"] = diff_dictionary["count"] + 1
            diff_dictionary[str(diff_count)] = 1

        diff_count = diff_count + 1

    # cos difference
    cos_diff = counter_cosine_similarity(counter_eval, counter_original)
    print("COS distance = "+str(cos_diff))


    # page_sequence = {}
    # page_sequence["eval"] = []
    # page_sequence["original"] =[]
    # map_sequence = {}
    # map_sequence["eval"] = []
    # map_sequence["original"] =[]

    # transition_count = {}
    # transition_count["eval"] = {}
    # transition_count["original"] = {}

    # similar_count = {}


    json_file = open("annotation_page_sequence"+FILE_POSTFIX, "w")
    # magic happens here to make it pretty-printed
    json_file.write(json.dumps(page_sequence, indent=4))
    json_file.close() 

    json_file = open("annotation_map_sequence"+FILE_POSTFIX, "w")
    # magic happens here to make it pretty-printed
    json_file.write(json.dumps(map_sequence, indent=4))
    json_file.close() 


    json_file = open("annotation_transition_count"+FILE_POSTFIX, "w")
    # magic happens here to make it pretty-printed
    json_file.write(json.dumps(transition_count, indent=4))
    json_file.close() 

    json_file = open("annotation_transition_difference"+FILE_POSTFIX, "w")
    # magic happens here to make it pretty-printed
    json_file.write(json.dumps(diff_dictionary, indent=4))
    json_file.close() 






            #         page_sequence.append(mapping(item["label"]))

            #     if NORMALIZE_FLAG == True:

            #         normalized_sequence = normalize_sequence(max_dimension, page_sequence)
            #         narrative_seuqneces[page_index] = normalized_sequence

            #     else:
            #         narrative_seuqneces[page_index] = page_sequence
            #     # print(narrative_seuqneces[single_folder][page])
                
            # ### save to json file
            # json_file = open("narrative_sequecce"+FILE_POSTFIX+"_"+page_sequence, "w")
            # # magic happens here to make it pretty-printed
            # json_file.write(json.dumps(narrative_seuqneces[page_index], indent=4))
            # json_file.close() 

    # #### test data
    # # X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    
    # data_list = []
    # data_index = {}
    # book_sequence_count = {}

    # for book in narrative_seuqneces:
    #     book_sequence_count[book] = {}
    #     for page_sequence in narrative_seuqneces[book]:

    #         target_vector = narrative_seuqneces[book][page_sequence]

    #         if str(target_vector) not in book_sequence_count[book]:
    #             book_sequence_count[book][str(target_vector)] = 1
    #         else:
    #             book_sequence_count[book][str(target_vector)] = book_sequence_count[book][str(target_vector)] + 1
            
    #         data_list.append(target_vector)
    #         data_index[len(data_list) -1] = page_sequence
    #         # print(target_vector)



  


    # json_file = open("narrative_sequecce_index"+FILE_POSTFIX, "w")
    # # magic happens here to make it pretty-printed
    # json_file.write(json.dumps(data_index, indent=4))
    # json_file.close() 

                                
    # json_file = open("narrative_sequecce_count"+FILE_POSTFIX, "w")
    # # magic happens here to make it pretty-printed
    # json_file.write(json.dumps(book_sequence_count, indent=4))
    # json_file.close() 



    # if NORMALIZE_FLAG == True:

    #     ### K means
        
    #     num_cluster_centroid = 12

    #     data_for_clustering = np.array(data_list)
    #     kmeans = KMeans(n_clusters=num_cluster_centroid, random_state=0).fit(data_for_clustering)
    #     # print(kmeans.labels_)

    #     cluster_result_list = list(kmeans.labels_)

    #     index_count = 0
    #     cluster_results_dictionary = {}  
    #     for list_index in cluster_result_list:
    #         page_key = data_index[index_count]
    #         index_count = index_count +1

    #         # print(list_index)

    #         cluster_results_dictionary[page_key] = str(list_index)


    #     print(cluster_results_dictionary)


    #     json_file = open("narrative_sequecce_cluster"+FILE_POSTFIX, "w")
    #     # magic happens here to make it pretty-printed
    #     json_file.write(json.dumps(cluster_results_dictionary, indent=4))
    #     json_file.close() 



