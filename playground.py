import json
# with open("data/global/pubs_raw.json", "r", encoding="utf-8") as load:
#     raw_pubs = json.load(load)
# print("1")
import pickle

with open("data/global/name_to_pubs_test_100.json", "r", encoding="utf-8") as load:
     name_to_pubs_test = json.load(load)
     test_name_list = list(name_to_pubs_test.keys())
with open("data/global/name_to_pubs_train_500.json", "r", encoding="utf-8") as load:
    name_to_pubs_train = json.load(load)
    train_name_list = list(name_to_pubs_train.keys())
with open("aminerv2", 'wb') as save:
    pickle.dump([None, train_name_list, test_name_list], save)
print()