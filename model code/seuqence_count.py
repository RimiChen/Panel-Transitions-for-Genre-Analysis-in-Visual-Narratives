### This script count the sequences

### import
import json
import os

### FLAG and PARAMETERS
PAGE_WISE_FLAG = True
ALL_DATASET = True

MAX_SEQUENCE = 6
SEQUENCE_LENGTH = 1

STORE_FOLDER_PATH = "./length_2_"+str(SEQUENCE_LENGTH )+"/"


ROOT_PATH = "./"
TRANSITION_SEQUENCE_PATH = "narrative_sequecce_original_latest"
# otherwise separate as books


def generte_possible_list(sequence_length, key_set):
    
    current_list = []
    for layer in range(sequence_length):
        new_list = []

        if len(current_list) == 0:
            for want_key in key_set:
                new_list.append(str(want_key))
        else:    
            for current_key in current_list:
                for want_key in key_set:
                    # [int(s) for s in example_string.split(',')]
                    new_list.append(str(current_key)+", "+str(want_key))

            
        current_list = new_list

    return current_list 


def store_result(result_records, file_prefix, file_name, folder_path):
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    json_file = open(folder_path+file_prefix+"_"+file_name+".json", "w")
    # magic happens here to make it pretty-printed
    json_file.write(json.dumps(result_records, indent=4))
    json_file.close()

def update_dictionary_count(target, new_record, number):
    if new_record in target:
        target[new_record] = target[new_record] + number
    else:
        target[new_record] = number

    return target

### string count: https://www.geeksforgeeks.org/python-count-overlapping-substring-in-a-given-string/
def CountOccurrences(string, substring):
  
    # Initialize count and start to 0
    count = 0
    start = 0
  
    # Search through the string till
    # we reach the end of it
    while start < len(string):
  
        # Check if a substring is present from
        # 'start' position till the end
        pos = string.find(substring, start)
  
        if pos != -1:
            # If a substring is present, move 'start' to
            # the next position from start of the substring
            start = pos + 1
  
            # Increment the count
            count += 1
        else:
            # If no further substring is present
            break
    # return the value of count
    return count
  
# Driver Code
# string = "GeeksforGeeksforGeeksforGeeks"
# print(CountOccurrences(string, "GeeksforGeeks"))


#### Launch the main process
if __name__ == "__main__":

    ### read data
    ### load_records is the target variable
    json_file_path = ROOT_PATH +TRANSITION_SEQUENCE_PATH
    with open(json_file_path) as record_file:
        load_records = json.load(record_file)
        record_file.close()


    # list all possiblilty
    target_sequence = {}
    # transition_key_Set = [
    #     1.0,
    #     2.0
    # ]

    transition_key_Set = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0
    ]

    if SEQUENCE_LENGTH > MAX_SEQUENCE:
        print("SYSTEM level 1: exceed max sequence, use "+str(MAX_SEQUENCE)+" instead.")
        possible_list = generte_possible_list(MAX_SEQUENCE, transition_key_Set)

    else:
        print("SYSTEM level 1: sequence length "+str(SEQUENCE_LENGTH)+".")
        possible_list = generte_possible_list(SEQUENCE_LENGTH, transition_key_Set)

    ## DEBUG check the results
    # print("\n".join(possible_list))



    if PAGE_WISE_FLAG == True:
        print("SYSTEM level 1: processing the sequence according to pages")
        
        if ALL_DATASET == True:
            print("SYSTEM level 2: all books")


            file_prefix = "book_sequence"

            for book in load_records:
                result_records = {}
                file_name = book
                # print(book)

                # load_records[book]
                # pages
                for page in load_records[book]:
                    if len(load_records[book][page]) >=  SEQUENCE_LENGTH:
                        # print(str(load_records[book][page]))
                        transition_sequence = str(load_records[book][page])
                        transition_sequence = transition_sequence.replace("[","")
                        transition_sequence = transition_sequence.replace("]","")
                        print(transition_sequence)

                        for seuqence_key in possible_list:
                            # count = transition_sequence.count(seuqence_key)
                            count = CountOccurrences(transition_sequence, seuqence_key)
                            
                            if count > 0 :
                                # print("SYSTEM level 4: \""+seuqence_key+"\", "+str(count)+" in "+str(page))

                                result_records = update_dictionary_count(result_records, seuqence_key, count)
                            # else:
                            #     print("SYSTEM level 4: no \""+seuqence_key+"\""+" in "+str(page))
                            #     update_dictionary_count(result_records, seuqence_key, count)

                    else:
                        # print("SYSTEM level 3: length_error")
                        result_records = update_dictionary_count(result_records, "length_error", 1)

                result_records = dict(sorted(result_records.items(), key=lambda item: item[1], reverse=True))
                store_result(result_records, file_prefix, file_name+"_len-"+str(SEQUENCE_LENGTH), STORE_FOLDER_PATH )

        else:
            print("SYSTEM level 2: separate books")
            file_prefix = "all_sequence"

            # store_result(result_records, file_prefix, file_name)

    else:

        print("SYSTEM: processing all")

        ### TODO (but not neccessary, because we didn't label transitions between pages):