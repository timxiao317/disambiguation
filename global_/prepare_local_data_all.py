import argparse
from os.path import join
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from numpy.random import shuffle
from global_.global_model import GlobalTripletModel
from utils.eval_utils import get_hidden_output
from utils.cache import LMDBClient
from utils import data_utils
from utils import settings

IDF_THRESHOLD = 32  # small data


# IDF_THRESHOLD = 10


def dump_inter_emb():
    """
    dump hidden embedding via trained global model for local model to use
    """
    LMDB_NAME = "author_100.emb.weighted"
    lc_input = LMDBClient(test_dataset_name, LMDB_NAME)
    INTER_LMDB_NAME = 'author_triplets.emb'
    lc_inter = LMDBClient(exp_name, INTER_LMDB_NAME)
    global_model = GlobalTripletModel(train_dataset_name, data_scale=1000000)
    trained_global_model = global_model.load_triplets_model()
    name_to_pubs_test = {}
    TRAIN_NAME_LIST, TEST_NAME_LIST = settings.get_split_name_list(test_dataset_name)
    for case_name in TRAIN_NAME_LIST + TEST_NAME_LIST:
        name_to_pubs_test[case_name] = data_utils.load_json(join(settings.get_raw_data_dir(test_dataset_name), case_name), "assignments.json")
    # name_to_pubs_test = data_utils.load_json(settings.get_global_data_dir(dataset_name), 'name_to_pubs_test_100.json')
    for name in name_to_pubs_test:
        print('name', name)
        name_data = name_to_pubs_test[name]
        embs_input = []
        pids = []
        for i, aid in enumerate(name_data.keys()):
            if len(name_data[aid]) < 1:  # n_pubs of current author is too small
                continue
            for pid in name_data[aid]:
                cur_emb = lc_input.get(pid)
                if cur_emb is None:
                    continue
                embs_input.append(cur_emb)
                pids.append(pid)
        embs_input = np.stack(embs_input)
        inter_embs = get_hidden_output(trained_global_model, embs_input)
        for i, pid_ in enumerate(pids):
            lc_inter.set(pid_, inter_embs[i])


def gen_local_data(idf_threshold=10):
    """
    generate local data (including paper features and paper network) for each associated name
    :param idf_threshold: threshold for determining whether there exists an edge between two papers (for this demo we set 29)
    """
    name_to_pubs_test = {}
    TRAIN_NAME_LIST, TEST_NAME_LIST = settings.get_split_name_list(test_dataset_name)
    for case_name in TRAIN_NAME_LIST + TEST_NAME_LIST:
        name_to_pubs_test[case_name] = data_utils.load_json(join(settings.get_raw_data_dir(test_dataset_name), case_name), "assignments.json")
    # name_to_pubs_test = data_utils.load_json(settings.get_global_data_dir(dataset_name), 'name_to_pubs_test_100.json')
    idf = data_utils.load_data(settings.get_overall_feature_dir(), 'feature_idf.pkl')
    INTER_LMDB_NAME = 'author_triplets.emb'
    lc_inter = LMDBClient(exp_name, INTER_LMDB_NAME)
    LMDB_AUTHOR_FEATURE = "pub_authors.feature"
    lc_feature = LMDBClient(test_dataset_name, LMDB_AUTHOR_FEATURE)
    graph_dir = join(settings.get_data_dir(exp_name), 'local', 'graph-{}'.format(idf_threshold))
    os.makedirs(graph_dir, exist_ok=True)
    for i, name in enumerate(name_to_pubs_test):
        print(i, name)
        cur_person_dict = name_to_pubs_test[name]
        pids_set = set()
        pids = []
        pids2label = {}

        # generate content
        wf_content = open(join(graph_dir, '{}_pubs_content.txt'.format(name)), 'w')
        for i, aid in enumerate(cur_person_dict):
            items = cur_person_dict[aid]
            if len(items) < 1:
                continue
            for pid in items:
                pids2label[pid] = aid
                pids.append(pid)
        shuffle(pids)
        for pid in pids:
            cur_pub_emb = lc_inter.get(pid)
            if cur_pub_emb is not None:
                cur_pub_emb = list(map(str, cur_pub_emb))
                pids_set.add(pid)
                wf_content.write('{}\t'.format(pid))
                wf_content.write('\t'.join(cur_pub_emb))
                wf_content.write('\t{}\n'.format(pids2label[pid]))
                print(pids2label[pid])
        wf_content.close()

        # generate network
        pids_filter = list(pids_set)
        n_pubs = len(pids_filter)
        print('n_pubs', n_pubs)
        wf_network = open(join(graph_dir, '{}_pubs_network.txt'.format(name)), 'w')
        for i in range(n_pubs - 1):
            if i % 10 == 0:
                print(i)
            author_feature1 = set(lc_feature.get(pids_filter[i]))
            for j in range(i + 1, n_pubs):
                author_feature2 = set(lc_feature.get(pids_filter[j]))
                common_features = author_feature1.intersection(author_feature2)
                idf_sum = 0
                for f in common_features:
                    idf_sum += idf.get(f, idf_threshold)
                    # print(f, idf.get(f, idf_threshold))
                if idf_sum >= idf_threshold:
                    wf_network.write('{}\t{}\n'.format(pids_filter[i], pids_filter[j]))
        wf_network.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset_name", default="whoiswho_new", type=str)
    parser.add_argument("--train_dataset_name", default="whoiswho_new", type=str)
    args = parser.parse_args()
    train_dataset_name = args.train_dataset_name
    test_dataset_name = args.test_dataset_name
    exp_name = "{}_{}".format(train_dataset_name, test_dataset_name)

    dump_inter_emb()
    gen_local_data(idf_threshold=IDF_THRESHOLD)
    print('done')
