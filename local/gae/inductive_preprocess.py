from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import time
from os.path import join

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import codecs
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import pickle

from local.gae.optimizer import OptimizerAE, OptimizerVAE
from local.gae.input_data import load_local_data
from local.gae.model import GCNModelAE, GCNModelVAE
from local.gae.preprocessing import preprocess_graph, construct_feed_dict, \
    sparse_to_tuple, normalize_vectors, gen_train_edges, cal_pos_weight
from utils.cluster import clustering
from utils.data_utils import load_json
from utils.eval_utils import pairwise_precision_recall_f1, cal_f1
from utils import settings

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')  # 32
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')  # 16
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('name', 'hui_fang', 'Dataset string.')
flags.DEFINE_string('train_dataset_name', "whoiswho_new", "")
flags.DEFINE_string('test_dataset_name', "whoiswho_new", "")
flags.DEFINE_float('idf_threshold', 0., "")
# flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('is_sparse', 0, 'Whether input features are sparse.')

model_str = FLAGS.model
name_str = FLAGS.name
start_time = time.time()


def preprocess(name):
    adj, features, labels = load_local_data(exp_name, IDF_THRESHOLD, name=name)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_train = gen_train_edges(adj)

    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    return adj_norm, adj_label, features


def main():
    train_names, _ = settings.get_split_name_list(train_dataset_name)
    _, test_names = settings.get_split_name_list(test_dataset_name)
    for name in train_names + test_names:
        result = preprocess(name)
        save_local_preprocess_result(result, name)

    # wf = codecs.open(join(settings.get_out_dir(exp_name), 'local_clustering_results.csv'), 'w', encoding='utf-8')


def save_local_preprocess_result(result, name):
    path = join(settings.get_data_dir(exp_name), 'local', 'preprocess-{}'.format(IDF_THRESHOLD), name)
    with open(path, 'wb') as save:
        pickle.dump(result, save)


def load_local_preprocess_result(name):
    path = join(settings.get_data_dir(exp_name), 'local', 'preprocess-{}'.format(IDF_THRESHOLD), name)
    with open(path, 'rb') as load:
        return pickle.load(load)


if __name__ == '__main__':
    # gae_for_na('hongbin_liang')
    # gae_for_na('j_yu')
    # gae_for_na('s_yu')
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_name", default="whoiswho_new", type=str)
    # args = parser.parse_args()
    train_dataset_name = FLAGS.train_dataset_name
    test_dataset_name = FLAGS.test_dataset_name
    IDF_THRESHOLD = FLAGS.idf_threshold
    exp_name = "{}_{}_{}".format(train_dataset_name, test_dataset_name, IDF_THRESHOLD)
    main()
