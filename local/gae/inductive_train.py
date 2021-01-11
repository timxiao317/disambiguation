from __future__ import division
from __future__ import print_function

import argparse
import pickle
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

from local.gae.optimizer import OptimizerAE, OptimizerVAE, OptimizerInductiveAE
from local.gae.input_data import load_local_data
from local.gae.model import GCNModelAE, GCNModelVAE, GCNModelInductiveAE
from local.gae.preprocessing import preprocess_graph, construct_feed_dict, \
    sparse_to_tuple, normalize_vectors, gen_train_edges, cal_pos_weight, construct_feed_dict_inductive
from utils.cluster import clustering
from utils.data_utils import load_json
from utils.eval_utils import pairwise_precision_recall_f1, cal_f1
from utils import settings

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')  # 32
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')  # 16
flags.DEFINE_float('weight_decay', 0.01, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.25, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('input_feature_dim', 64, 'input feature dim')
flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('name', 'hui_fang', 'Dataset string.')
flags.DEFINE_string('train_dataset_name', "whoiswho_new", "")
flags.DEFINE_string('test_dataset_name', "whoiswho_new", "")
flags.DEFINE_string('model_name', "baseline", "")
flags.DEFINE_float('idf_threshold', 0., "")
# flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('is_sparse', 0, 'Whether input features are sparse.')


model_str = FLAGS.model
name_str = FLAGS.name
start_time = time.time()



def load_local_preprocess_result(exp_name, IDF_THRESHOLD, name):
    path = join(settings.get_data_dir(exp_name), 'local', 'preprocess-{}'.format(IDF_THRESHOLD), name)
    with open(path, 'rb') as load:
        return pickle.load(load)

def load_test_names(dataset_name):
    _, TEST_NAME_LIST = settings.get_split_name_list(dataset_name)
    return TEST_NAME_LIST

def main():
    """
        train and evaluate YUTAO results for a specific name
        :param name:  author name
        :return: evaluation results
        """

    # Store original adjacency matrix (without diagonal entries) for later
    # Define placeholders
    placeholders = {
        # 'features': tf.sparse_placeholder(tf.float32),
        'features': tf.placeholder(tf.float32, shape=(None, input_feature_dim)),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'pos_weight': tf.placeholder(tf.float32, shape=()),
        'norm': tf.placeholder(tf.float32),
    }
    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelInductiveAE(placeholders, input_feature_dim)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerInductiveAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=model.pos_weight,
                              norm=model.norm)

    saver = tf.train.Saver()
    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    def get_embs():
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)  # z_mean is better
        return emb

    train_name_list, _ = settings.get_split_name_list(train_dataset_name)
    _, test_name_list = settings.get_split_name_list(test_dataset_name)

    # Train model
    for epoch in range(FLAGS.epochs):
        epoch_avg_cost = 0
        epoch_avg_accuracy = 0
        for name in train_name_list:
            adj_norm, adj_label, features, pos_weight, norm, labels = load_local_preprocess_result(exp_name, IDF_THRESHOLD, name)
            # print('positive edge weight', pos_weight)  # negative edges/pos edges
            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict_inductive(adj_norm, adj_label, features, pos_weight, norm, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                            feed_dict=feed_dict)
            # Compute average loss
            avg_cost = outs[1]
            avg_accuracy = outs[2]
            epoch_avg_cost += avg_cost
            epoch_avg_accuracy += avg_accuracy
            # print(avg_cost, avg_accuracy)


        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(epoch_avg_cost / len(train_name_list)),
              "train_acc=", "{:.5f}".format(epoch_avg_accuracy / len(train_name_list)),
              "time=", "{:.5f}".format(time.time() - t))
        metrics = np.zeros(3)
        tp_fp_fn_sum = np.zeros(3)
        for name in test_name_list:
            adj_norm, adj_label, features, pos_weight, norm, labels = load_local_preprocess_result(exp_name, IDF_THRESHOLD, name)
            feed_dict = construct_feed_dict_inductive(adj_norm, adj_label, features, pos_weight, norm, placeholders)
            emb = get_embs()
            n_clusters = len(set(labels))
            emb_norm = normalize_vectors(emb)
            clusters_pred = clustering(emb_norm, num_clusters=n_clusters)
            tp, fp, fn, prec, rec, f1 = pairwise_precision_recall_f1(clusters_pred, labels)
            tp_fp_fn_sum += np.array([tp, fp, fn])
            metrics += np.array([prec, rec, f1])
        macro_prec = metrics[0] / len(test_name_list)
        macro_rec = metrics[1] / len(test_name_list)
        macro_f1 = cal_f1(macro_prec, macro_rec)
        tp, fp, fn = tp_fp_fn_sum
        micro_precision = tp / (tp + fp)
        micro_recall = tp / (tp + fn)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        print('average,macro_prec:{0:.5f},macro_rec:{1:.5f},macro_f1:{2:.5f},micro_precision:{3:.5f},micro_recall:{4:5f},micro_f1:{5:5f}\n'.format(
            macro_prec, macro_rec, macro_f1, micro_precision, micro_recall, micro_f1))
    path = join(settings.get_data_dir(exp_name), 'local', 'model-{}'.format(IDF_THRESHOLD), model_name)
    saver.save(sess, path)
    # emb = get_embs()
    # n_clusters = len(set(labels))
    # emb_norm = normalize_vectors(emb)
    # clusters_pred = clustering(emb_norm, num_clusters=n_clusters)
    # tp, fp, fn, prec, rec, f1 = pairwise_precision_recall_f1(clusters_pred, labels)
    # print('pairwise precision', '{:.5f}'.format(prec),
    #       'recall', '{:.5f}'.format(rec),
    #       'f1', '{:.5f}'.format(f1))
    # return [tp, fp, fn], [prec, rec, f1], num_nodes, n_clusters


if __name__ == '__main__':
    # gae_for_na('hongbin_liang')
    # gae_for_na('j_yu')
    # gae_for_na('s_yu')
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_name", default="whoiswho_new", type=str)
    # args = parser.parse_args()
    model_name = FLAGS.model_name
    input_feature_dim = FLAGS.input_feature_dim
    train_dataset_name = FLAGS.train_dataset_name
    test_dataset_name = FLAGS.test_dataset_name
    IDF_THRESHOLD = FLAGS.idf_threshold
    exp_name = "{}_{}_{}".format(train_dataset_name, test_dataset_name, IDF_THRESHOLD)
    # main()
    main()