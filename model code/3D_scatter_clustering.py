####  This script is draw points in 3D space with chosed axis
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
import numpy as np
import json
from os import walk
import random
from sklearn import metrics
from scipy.spatial.distance import cdist


CLUSTER_CENTROID = 4

JSON_PATH = "length1/"
JSON_PREFIX = "count_result_"

all_axis = [
    "Subject",
    "Scene",
    "Aspect",
    "Action",
    "Moment",
    "Non_sequitur",
    "length_error",
    "key_error"
    ]


GENRE_MAPPING = {}
GENRE_MAPPING["Animal"] = 0
GENRE_MAPPING["Battle"] = 1
GENRE_MAPPING["Fantasy"] = 2
GENRE_MAPPING["4_panels_cartoon"] = 3
GENRE_MAPPING["Historical_drama"] = 4
GENRE_MAPPING["Horror"] = 5
GENRE_MAPPING["Humor"] = 6
GENRE_MAPPING["Love_romance"] = 7
GENRE_MAPPING["Romantic_comedy"] = 8
GENRE_MAPPING["Science_fiction"] = 9
GENRE_MAPPING["Sports"] = 10
GENRE_MAPPING["Suspense"] = 11

TARGET_MAPPING = {}
TARGET_MAPPING["Boy"] = 0
TARGET_MAPPING["Girl"] = 1
TARGET_MAPPING["Lady"] = 2
TARGET_MAPPING["Young_men"] = 3
 

BOOK_GENRE = {}
BOOK_GENRE["0"] = ["Girl",	"Love_romance"]
BOOK_GENRE["1"] = ["Boy", "Battle" ]
BOOK_GENRE["2"] = ["Boy", "4_panels_cartoon"]
BOOK_GENRE["3"] = ["Lady", "Love_romance"]
BOOK_GENRE["4"] = ["Boy", "Science_fiction"]
BOOK_GENRE["5"] = ["Boy", "Romantic_comedy" ]
BOOK_GENRE["6"] = ["Young_men", "Science_fiction"]
BOOK_GENRE["7"] = ["Girl", "Romantic_comedy"]
BOOK_GENRE["8"] = ["Boy", "Fantasy"]
BOOK_GENRE["9"] = ["Boy", "Science_fiction"]
BOOK_GENRE["10"] = ["Boy", "Humor"]
BOOK_GENRE["11"] = ["Young_men", "Historical_drama"]
BOOK_GENRE["12"] = ["Girl", "Science_fiction"]
BOOK_GENRE["13"] = ["Boy", "Sports" ]
BOOK_GENRE["14"] = ["Boy", "Battle"]
BOOK_GENRE["15"] = ["Girl", "Romantic_comedy"]
BOOK_GENRE["16"] = ["Young_men", "Battle"]
BOOK_GENRE["17"] = ["Girl", "Animal"]
BOOK_GENRE["18"] = ["Boy", "Science_fiction"]
BOOK_GENRE["19"] = ["Young_men", "Animal"]
BOOK_GENRE["20"] = ["Lady", "Battle"]
BOOK_GENRE["21"] = ["Young_men", "Humor"]
BOOK_GENRE["22"] = ["Girl", "Fantasy"]
BOOK_GENRE["23"] = ["Boy", "Love_romance"]
BOOK_GENRE["24"] = ["Boy", "Historical_drama"]
BOOK_GENRE["25"] = ["Young_men", "Suspense"]
BOOK_GENRE["26"] = ["Girl", "Love_romance"]
BOOK_GENRE["27"] = ["Girl", "Science_fiction"]
BOOK_GENRE["28"] = ["Boy", "Science_fiction"]
BOOK_GENRE["65"] = ["Lady", "4_panels_cartoon"]
BOOK_GENRE["92"] = ["Young_men","4_panels_cartoon"]
BOOK_GENRE["105"] = ["Lady", "4_panels_cartoon"]



import matplotlib,numpy
import pylab

cmap = matplotlib.colors.ListedColormap ( numpy.random.rand ( 256,3))
# pylab.imshow ( Z, cmap = cmap)
# pylab.show()


def normalize_vector(input_vector):
    ### normalize input list (vecter)
    total = 0
    normalized_vector = []
    for number in input_vector:
        total = total + number
    
    for number in input_vector:
        if total != 0:
            new_number = float(number/float(total))
        else:
            new_number = float(0)
            
        normalized_vector.append(new_number)
        
    return normalized_vector
    
def chose_axis(axis_1, axis_2, axis_3, data):
    ### choose needed 3 axis
    
    x_axis = axis_1
    # print(regroup_axis(data, axis_1))
    column_x = regroup_axis(data, axis_1)
    y_axis= axis_2
    # print(regroup_axis(data, axis_2))
    column_y = regroup_axis(data, axis_2)
    z_axis = axis_3
    # print(regroup_axis(data, axis_3))
    column_z = regroup_axis(data, axis_3)
    print(" x = "+x_axis+", y = "+y_axis+" z = "+z_axis)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    n = 100
    marker_sign = "o"
    x_array = np.array(column_x)
    y_array = np.array(column_y)
    z_array = np.array(column_z)
    
    # xs = randrange(n, 23, 32)
    # ys = randrange(n, 0, 100)
    # zs = randrange(n, 10, 20)
    
    ax.scatter(x_array, y_array, z_array, marker=marker_sign)

    ax.set_xlabel(axis_1)
    ax.set_ylabel(axis_2)
    ax.set_zlabel(axis_3)

    plt.show()   
     
def chose_axis_w_cluster(axis_1, axis_2, axis_3, data, cluster_labels, max_label):
    ### choose needed 3 axis
    
    x_axis = axis_1
    # print(regroup_axis(data, axis_1))
    column_x = regroup_axis(data, axis_1)
    y_axis= axis_2
    # print(regroup_axis(data, axis_2))
    column_y = regroup_axis(data, axis_2)
    z_axis = axis_3
    # print(regroup_axis(data, axis_3))
    column_z = regroup_axis(data, axis_3)
    print(" x = "+x_axis+", y = "+y_axis+" z = "+z_axis)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 100
    marker_sign = "o"
    x_array = np.array(column_x)
    y_array = np.array(column_y)
    z_array = np.array(column_z)
    
    # xs = randrange(n, 23, 32)
    # ys = randrange(n, 0, 100)
    # zs = randrange(n, 10, 20)


    # scatter_x = np.array([1,2,3,4,5])
    # scatter_y = np.array([5,4,3,2,1])
    group = np.array(cluster_labels)
    
    
    

    #### TODO: make the color dictionary randomly generated.
    cdict = {
        0: 'red', 1: 'blue', 2: 'green',
        3: 'orange', 4: 'purple', 5: 'brown',
        6: 'pink', 7: 'cyan', 8: 'gold',
        9: 'salmon', 10: 'darkturquoise', 11: 'orchid'
    }
    # cdict = {}
    # for index in range(max_label):
    #     r = random.random()

    #     b = random.random()

    #     g = random.random()

    #     color = (r, g, b)
    #     print(color)
    #     rgb = np.random.rand(3,)        
    #     cdict[index] = rgb
    # print(cdict)
    # new_cmap = rand_cmap(100, type='bright', first_color_black=True, last_color_black=False, verbose=True)

    # fig, ax = plt.subplots()
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(x_array[ix], y_array[ix], z_array[ix], c = cdict[g], label = g)
        # ax.scatter(x_array[ix], y_array[ix], z_array[ix], c = cmap[g], label = g)

    ax.legend()
    # plt.show()

    
    # ax.scatter(x_array, y_array, z_array, marker=marker_sign)

    ax.set_xlabel(axis_1)
    ax.set_ylabel(axis_2)
    ax.set_zlabel(axis_3)

    plt.show()   

def axis_mapping(axis_name, name_array):
    ## mapping name to index
    
    target_index = -1
    list_count = 0
    for name in name_array:
        if name == axis_name:
            target_index = list_count
            break
        
        list_count = list_count + 1
        
    return target_index
            
def regroup_axis(data, axis_name):
    # load a vector array (many rows), and regroup then column wise
    target_vector = []
    for vector in data:
        ### expected [0.1, 0.2]
        ### expected [0.2, 0.3]
        ### expected [0.1, 0.2]
        # print(vector)
        target_vector.append(vector[axis_mapping(axis_name, all_axis)])
        #  print(vector[axis_mapping(axis_name, all_axis)])    
    
    return target_vector 

def mapping_number_transition(number):

    TRANSITION_MAPPING = {}
    TRANSITION_MAPPING["1.0"] = "Action"
    TRANSITION_MAPPING["2.0"] = "Aspect"
    TRANSITION_MAPPING["3.0"] = "Subject"
    TRANSITION_MAPPING["4.0"] ="Scene"
    TRANSITION_MAPPING["5.0"] = "Moment"
    TRANSITION_MAPPING["6.0"] ="Non_sequitur"
    TRANSITION_MAPPING["length_error"] = "length_error"  


    if number in TRANSITION_MAPPING:
        transition = TRANSITION_MAPPING[number]
    else:
        transition =  "key_error"    

    return transition

def json_2_vector(json_path, label_array):
    # load counts from json file and make them a list (be used as a vector)
    target_vector = []
    label_index = {}

    label_index_count = 0
    for label in label_array:
        if label not in label_index:
            label_index[label] = label_index_count
            label_index_count = label_index_count + 1

    for label in label_index:
        # a vector with all zeros
        target_vector.append(float(0))

    with open(json_path) as json_file:
        # print(json_path)

        transition_counts = json.load(json_file)
        json_file.close()
        
        for transition_count in transition_counts:
            # target_vector.append(transition_counts[transition_count])
            # target_vector[label_index[transition_count]] = float(transition_counts[transition_count])
            target_vector[label_index[mapping_number_transition(transition_count)]] = float(transition_counts[transition_count])
    
    return target_vector




    
if __name__ == "__main__":
    print("Show the points in vector space")
    
    #### test imagine data
    # test_data_1 = [0.1, 0.2, 0.3, 0.0, 0.2, 0.1, 0.1] 
    # test_data_2 = [0.2, 0.3, 0.1, 0.1, 0.1, 0.0, 0.2] 
    # data_1 = {}
    # data_2 = {}
    
    # data_1["Action"] = test_data_1[0]
    # data_1["Aspect"] = test_data_1[1]
    # data_1["Moment"] = test_data_1[2]
    # data_1["Non_sequitur"] = test_data_1[3]
    # data_1["Other"] = test_data_1[4]
    # data_1["Scene"] = test_data_1[5]
    # data_1["Subject"] = test_data_1[6]

    # data_2["Action"] = test_data_2[0]
    # data_2["Aspect"] = test_data_2[1]
    # data_2["Moment"] = test_data_2[2]
    # data_2["Non_sequitur"] = test_data_2[3]
    # data_2["Other"] = test_data_2[4]
    # data_2["Scene"] = test_data_2[5]
    # data_2["Subject"] = test_data_2[6]

    
    # data = [test_data_1, test_data_2]
    
    # chose_axis("Action", "Subject", "Aspect", data)
    
    # JSON_PATH
    
    data = []
    book_target_label = []
    book_genre_label = []
    book_names = []
    print(JSON_PATH)
    for (_, _, file_names) in walk(JSON_PATH):
        # print(file_names)
        for single_file in file_names:
            print(single_file)
            book_index = single_file.replace("book_sequence_", "")
            book_index = book_index.replace("_len-1.json", "")
            print("book index = "+str(book_index))
            print("first genre = "+ BOOK_GENRE[str(book_index)][0])
            print("second genre = "+ BOOK_GENRE[str(book_index)][1])
            print("color index = "+ str(TARGET_MAPPING[BOOK_GENRE[str(book_index)][0]]) + ", and " + str(GENRE_MAPPING[BOOK_GENRE[str(book_index)][1]]))

            book_target_label.append(TARGET_MAPPING[BOOK_GENRE[str(book_index)][0]])
            book_genre_label.append(GENRE_MAPPING[BOOK_GENRE[str(book_index)][1]])

            book_names.append(book_index)

            book_vector = json_2_vector(JSON_PATH+single_file, all_axis)
            normalized_book_vector = normalize_vector(book_vector)
            
            data.append(normalized_book_vector)
            print(normalized_book_vector)     
    


    book_genre_count = {}
    book_target_count = {}
    for target in book_target_label:
        # print(genre)
        if target not in book_target_count:
            book_target_count[target] = 1
        else:
            book_target_count[target] = book_target_count[target] + 1

    for genre in book_genre_label:
        # print(genre)
        if genre not in book_genre_count:
            book_genre_count[genre] = 1
        else:
            book_genre_count[genre] = book_genre_count[genre] + 1


    json_file = open("target_title_count.json", "w")
    # magic happens here to make it pretty-printed
    json_file.write(json.dumps(book_target_count, indent=4))
    json_file.close()  
    json_file = open("genre_title_count.json", "w")
    # magic happens here to make it pretty-printed
    json_file.write(json.dumps(book_genre_count, indent=4))
    json_file.close()     
    # chose_axis("Action", "Subject", "Aspect", data) 
    #print(type(data))
    

    ### cluster similarity

    
    ### data is the target data
    ### k-means
    
    num_cluster_centroid = CLUSTER_CENTROID
    #### test data
    # X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    
    data_for_clustering = np.array(data)
    kmeans = KMeans(n_clusters=num_cluster_centroid, random_state=0).fit(data_for_clustering)
    print(kmeans.labels_)


    # distortions = []
    # inertias = []
    # mapping1 = {}
    # mapping2 = {}
    # K = range(1, 10)
    
    # for k in K:
    #     # Building and fitting the model
    #     kmeanModel = KMeans(n_clusters=k).fit(data_for_clustering)
    #     kmeanModel.fit(data_for_clustering)
    
    #     distortions.append(sum(np.min(cdist(data_for_clustering, kmeanModel.cluster_centers_,
    #                                         'euclidean'), axis=1)) / data_for_clustering.shape[0])
    #     inertias.append(kmeanModel.inertia_)
    
    #     mapping1[k] = sum(np.min(cdist(data_for_clustering, kmeanModel.cluster_centers_,
    #                                 'euclidean'), axis=1)) / data_for_clustering.shape[0]
    #     mapping2[k] = kmeanModel.inertia_



    # for key, val in mapping1.items():
    #     print(f'{key} : {val}')

    # plt.plot(K, distortions, 'bx-')
    # plt.xlabel('K centroids')
    # plt.ylabel('Distortion')
    # plt.title('The Elbow Method using Distortion')
    # plt.show()


    # for key, val in mapping2.items():
    #     print(f'{key} : {val}')

    # plt.plot(K, inertias, 'bx-')
    # plt.xlabel('K centroids')
    # plt.ylabel('Inertia')
    # plt.title('The Elbow Method using Inertia')
    # plt.show()

    # print(kmeans.predict([[0, 0], [12, 3]]))
    print(kmeans.cluster_centers_)
    book_genre_record = {}
    
    book_count = 0

    book_cluster = {}
    book_cluster["0"] = []
    book_cluster["1"] = []
    book_cluster["2"] = []
    book_cluster["3"] = []

    book_cluster["Animal"] = []
    book_cluster["Battle"] = []
    book_cluster["Fantasy"] = []
    book_cluster["4_panels_cartoon"] = []
    book_cluster["Historical_drama"] = []
    book_cluster["Horror"] = []
    book_cluster["Humor"] = []
    book_cluster["Love_romance"] = []
    book_cluster["Romantic_comedy"] = []
    book_cluster["Science_fiction"] = []
    book_cluster["Sports"] = []
    book_cluster["Suspense"] = []

    for item in list(kmeans.labels_):
        book_genre_record[str(book_count)] = {}
        book_genre_record[str(book_count)]["book"] = str(book_names[book_count])
        book_genre_record[str(book_count)]["cluster"] = str(item)
        book_genre_record[str(book_count)]["data"] = str(data[book_count])


        book_cluster[str(item)].append(str(book_names[book_count]))
        genre_name = BOOK_GENRE[str(book_names[book_count])][1]
        book_cluster[genre_name].append(str(book_names[book_count]))
        book_count  = book_count + 1 

    json_file = open("book_genre.json", "w")
    # magic happens here to make it pretty-printed
    json_file.write(json.dumps(book_genre_record, indent=4))
    json_file.close()      

    json_file = open("book_cluster.json", "w")
    # magic happens here to make it pretty-printed
    json_file.write(json.dumps(book_cluster, indent=4))
    json_file.close() 

    data_clustering_labels = kmeans.labels_
    # print(type(data_clustering_labels))
    # print(numpy.array(book_genre_label))
    ## change clustering labels according to genres

    # Clean the book genre, we want 1, 2, 7, 9
    new_book_target_list = []
    new_book_genre_list = []

    # adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])
    cluster_similarity = []
    # cluster_count = 0
    for cluster_1 in book_cluster:
        #print(cluster_1)
        for cluster_2 in book_cluster:
            print(cluster_1+", "+cluster_2)
            cluster_similarity_node = {}
            cluster_similarity_node["cluster_1"] = cluster_1
            cluster_similarity_node["cluster_2"] = cluster_2
            # cluster_similarity_node["similarity"] =  adjusted_rand_score(book_cluster[cluster_1], book_cluster[cluster_2])
            cluster_similarity_node["similarity"] =  list(set.intersection(set(book_cluster[cluster_1]), set(book_cluster[cluster_2])))
            cluster_similarity_node["overlap"] = len(cluster_similarity_node["similarity"])
            if len(book_cluster[cluster_2]) == 0:
                cluster_similarity_node["out_of"] = 0
            else:    
                cluster_similarity_node["out_of"] = float( len(cluster_similarity_node["similarity"]) / len(book_cluster[cluster_2]))

            # cluster_count = cluster_count +1

            cluster_similarity.append(cluster_similarity_node)
            
    json_file = open("cluster_similarity.json", "w")
    # magic happens here to make it pretty-printed
    json_file.write(json.dumps(cluster_similarity, indent=4))
    json_file.close() 

    # cdict = {
    #     0: 'red', 1: 'blue', 2: 'green',
    #     3: 'orange', 4: 'purple', 5: 'brown',
    #     6: 'pink', 7: 'cyan', 8: 'gold',
    #     9: 'salmon', 10: 'darkturquoise', 11: 'orchid'
    # }
    needed_data = []
    book_count = 0
    for genre in book_genre_label:
        new_book_genre_list.append(genre)
        # new_book_genre_list_2.append(1)

        needed_data.append(data[book_count])

        # print(genre)
        # if int(genre) == 1:
        #     new_book_genre_list.append(0)
        #     # new_book_genre_list_2.append(0)

        #     needed_data.append(data[book_count])
        # elif int(genre) == 7:
        #     new_book_genre_list.append(1)
        #     # new_book_genre_list_2.append(1)

        #     needed_data.append(data[book_count])
        # # elif int(genre) == 7:
        #     new_book_genre_list.append(2)
        #     new_book_genre_list_2.append(2)

        #     needed_data.append(data[book_count])
        # elif int(genre) == 9:
        #     new_book_genre_list.append(3)
        #     new_book_genre_list_2.append(3)

        #     needed_data.append(data[book_count])
        # else:
        #     new_book_genre_list.append(6)
        
        book_count = book_count + 1

        


    # chose_axis_w_cluster("Action", "Subject", "Aspect", data, data_clustering_labels,num_cluster_centroid) 
    chose_axis_w_cluster("Action", "Subject", "Aspect", needed_data, numpy.array(new_book_genre_list),num_cluster_centroid) 



