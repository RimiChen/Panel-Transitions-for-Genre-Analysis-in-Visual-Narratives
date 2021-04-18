from sklearn.metrics import cohen_kappa_score
import json

TARGET_ROOT = "mixed_book_with_layer/"
TARGET_FILE = "temp_feedback_"+str(8)+".json"

TRANSITION_MAPPING = {}
TRANSITION_MAPPING["Action"] = 1.0
TRANSITION_MAPPING["Aspect"] = 2.0
TRANSITION_MAPPING["Subject"] = 3.0
TRANSITION_MAPPING["Scene"] = 4.0
TRANSITION_MAPPING["Moment"] = 5.0
TRANSITION_MAPPING["Non_sequitur"] = 6.0




def mapping(transition_label):
    transition_value = TRANSITION_MAPPING[transition_label]
    return transition_value


annotator1 = []
annotator2 = []

with open(TARGET_ROOT+TARGET_FILE) as record_file:

    load_records = json.load(record_file)

    for record in load_records:
        # print(record)
        annotator1.append(str(int(mapping(record["transition"]))))
        annotator2.append(str(int(record["feedback"])))

print("=======================")
print("=======================")
print(annotator1)
print("=======================")
print(annotator2)

# annotator1 = map_sequence["eval"]
# annotator2 = map_sequence["original"]

score = cohen_kappa_score(annotator1, annotator2)

print('Cohen\'s Kappa:',score)

record_kappa = {}
record_kappa["prediction"] = annotator1
record_kappa["feedback"] = annotator2
record_kappa["kappa"] = score


json_file = open(TARGET_ROOT+"kappa_"+TARGET_FILE, "w")
# magic happens here to make it pretty-printed
json_file.write(json.dumps(record_kappa, indent=4))
json_file.close() 