####  This script is draw points in 3D space with chosed axis
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import json
from os import walk
import random


JSON_PATH = "remove_data/old_count/"
JSON_PREFIX = "count_result_"

all_axis = [
    "Subject",
    "Scene",
    "Aspect",
    "Action",
    "Moment",
    "Non_sequitur"
    ]



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
        transition_counts = json.load(json_file)
        json_file.close()
        
        for transition_count in transition_counts:
            # target_vector.append(transition_counts[transition_count])
            target_vector[label_index[transition_count]] = float(transition_counts[transition_count])
    
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
    print(JSON_PATH)
    for (_, _, file_names) in walk(JSON_PATH):
        # print(file_names)
        for single_file in file_names:
            print(single_file)
            
            book_vector = json_2_vector(JSON_PATH+single_file, all_axis)
            normalized_book_vector = normalize_vector(book_vector)
            
            data.append(normalized_book_vector)
            print(normalized_book_vector)     
    
    # chose_axis("Action", "Subject", "Aspect", data) 
    #print(type(data))
    
    
    ### data is the target data
    ### k-means
    
    num_cluster_centroid = 6
    #### test data
    # X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    
    data_for_clustering = np.array(data)
    kmeans = KMeans(n_clusters=num_cluster_centroid, random_state=0).fit(data_for_clustering)
    print(kmeans.labels_)

    # print(kmeans.predict([[0, 0], [12, 3]]))
    print(kmeans.cluster_centers_)
    data_clustering_labels = kmeans.labels_
    chose_axis_w_cluster("Action", "Subject", "Aspect", data, data_clustering_labels,num_cluster_centroid) 

