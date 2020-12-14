import pickle
from os.path import abspath, dirname, join
import os

# DATA_SET_NAME = 'whoiswho_new'
PROJ_DIR = dirname(dirname(__file__))
PARENT_PROJ_DIR = dirname(PROJ_DIR)
SPLIT_PATH = join(PARENT_PROJ_DIR, 'split')
OVER_ALL_FEATURE_DIR = join(PROJ_DIR, 'feature')
# RAW_DATA_DIR = join(PARENT_PROJ_DIR, 'sota_data', 'yutao_data', DATA_SET_NAME)
# DATA_DIR = join(PROJ_DIR, 'data', DATA_SET_NAME)
# OUT_DIR = join(PROJ_DIR, 'out', DATA_SET_NAME)
# EMB_DATA_DIR = join(DATA_DIR, 'emb', DATA_SET_NAME)
# GLOBAL_DATA_DIR = join(PROJ_DIR, 'global', DATA_SET_NAME)
# FEATURE_DIR = join(DATA_DIR, 'features')
# with open(join(SPLIT_PATH, DATA_SET_NAME), 'rb') as load:
#     _, TRAIN_NAME_LIST, VAL_NAME_LIST, TEST_NAME_LIST = pickle.load(load)
# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(EMB_DATA_DIR, exist_ok=True)
# os.makedirs(GLOBAL_DATA_DIR, exist_ok=True)
# os.makedirs(FEATURE_DIR, exist_ok=True)

def get_raw_data_dir(DATA_SET_NAME):
    RAW_DATA_DIR = join(PARENT_PROJ_DIR, 'sota_data', 'yutao_data', DATA_SET_NAME)
    return RAW_DATA_DIR
def get_data_dir(DATA_SET_NAME):
    DATA_DIR = join(PROJ_DIR, 'data', DATA_SET_NAME)
    return DATA_DIR
def get_out_dir(DATA_SET_NAME):
    OUT_DIR = join(PROJ_DIR, 'out', DATA_SET_NAME)
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR
def get_embed_data_dir(DATA_SET_NAME):
    DATA_DIR = get_data_dir(DATA_SET_NAME)
    EMB_DATA_DIR = join(DATA_DIR, 'emb', DATA_SET_NAME)
    os.makedirs(EMB_DATA_DIR, exist_ok=True)
    return EMB_DATA_DIR
def get_global_data_dir(DATA_SET_NAME):
    GLOBAL_DATA_DIR = join(PROJ_DIR, 'global', DATA_SET_NAME)
    os.makedirs(GLOBAL_DATA_DIR, exist_ok=True)
    return GLOBAL_DATA_DIR
def get_feature_dir(DATA_SET_NAME):
    # DATA_DIR = get_data_dir(DATA_SET_NAME)
    # FEATURE_DIR = join(DATA_DIR, 'features')
    # os.makedirs(FEATURE_DIR, exist_ok=True)
    # return FEATURE_DIR
    return get_overall_feature_dir()
def get_overall_feature_dir():
    os.makedirs(OVER_ALL_FEATURE_DIR, exist_ok=True)
    return OVER_ALL_FEATURE_DIR
def get_split_name_list(DATA_SET_NAME):
    with open(join(SPLIT_PATH, DATA_SET_NAME), 'rb') as load:
        _, TRAIN_NAME_LIST, TEST_NAME_LIST = pickle.load(load)
    return TRAIN_NAME_LIST, TEST_NAME_LIST
