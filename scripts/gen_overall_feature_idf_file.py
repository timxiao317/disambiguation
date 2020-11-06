import math

from utils import settings, data_utils
from collections import defaultdict as dd
from utils.cache import LMDBClient
dataset_names =[
    "whoiswho_new",
    "aminerv1",
    "aminerv2",
    "aminerv3",
    "citeseerx",
]

counter = dd(int)
for dataset_name in dataset_names:
    overall_feature_dir = settings.get_overall_feature_dir()
    cnt = 0
    LMDB_NAME = 'pub_authors.feature'
    lc = LMDBClient(dataset_name, LMDB_NAME)
    author_cnt = 0
    with lc.db.begin() as txn:
        for k in txn.cursor():
            features = data_utils.deserialize_embedding(k[1])
            if author_cnt % 10000 == 0:
                print(author_cnt, features[0], counter.get(features[0]))
            author_cnt += 1
            for f in features:
                cnt += 1
                counter[f] += 1
idf = {}
for k in counter:
    idf[k] = math.log(cnt / counter[k])
data_utils.dump_data(dict(idf), overall_feature_dir, "feature_idf.pkl")