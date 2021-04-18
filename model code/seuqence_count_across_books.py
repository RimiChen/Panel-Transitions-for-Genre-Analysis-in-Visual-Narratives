### This script count the sequences

### import
import json
import os

### FLAG and PARAMETERS
PAGE_WISE_FLAG = True
ALL_DATASET = True

MAX_SEQUENCE = 6
SEQUENCE_LENGTH = 1
CATE_NAME = "plot"
# CATE_NAME = "romance"
# CATE_NAME = "fiction"
# CATE_NAME = "4_panels"
# CATE_NAME = "action"

# STORE_FOLDER_PATH = "./length"+str(SEQUENCE_LENGTH )+"/"
STORE_FOLDER_PATH = "./length_2_"+str(SEQUENCE_LENGTH )+"/"

ROOT_PATH = "./"

# book_sequence_0_len-2
PREFIX = "book_sequence_"

POSIFIX = "_len-"+str(SEQUENCE_LENGTH

)
TRANSITION_SEQUENCE_PATH = "narrative_sequecce_original_latest"
# otherwise separate as books


BOOK_CATE = {}
BOOK_CATE["plot"] = [
    "11",
    "17",
    "19",
    "24",
    "25"
]
BOOK_CATE["action"] = [
    "1",
    "13",
    "14",
    "16",
    "20"
]
BOOK_CATE["fiction"] = [
    "4",
    "6",
    "8",
    "9",
    "12",
    "18",
    "22",
    "27",
    "28"
]
BOOK_CATE["romance"] = [
    "0",
    "3",
    "5",
    "7",
    "15",
    "23",
    "26"
]
BOOK_CATE["4_panels"] = [
    "2",
    "65",
    "92",
    "105"
]



#### Launch the main process
if __name__ == "__main__":

    ### read data
    ### load_records is the target variable
    # json_file_path = ROOT_PATH +TRANSITION_SEQUENCE_PATH
    sequence_record = {}

    for book in BOOK_CATE[CATE_NAME]:
        json_file_path = STORE_FOLDER_PATH + PREFIX + book + POSIFIX + ".json"
        with open(json_file_path) as record_file:
            load_records = json.load(record_file)
            record_file.close()
            print("BOOK:" + str(book) +"====================")

        for record in load_records:
            print(record)
            if record not in sequence_record:
                sequence_record[record] = load_records[record]
            else:
                sequence_record[record] = sequence_record[record] + load_records[record]


    json_file = open(CATE_NAME +"_length_"+str(SEQUENCE_LENGTH)+".json", "w")
    # magic happens here to make it pretty-printed
    json_file.write(json.dumps(sequence_record, indent=4))
    json_file.close()
