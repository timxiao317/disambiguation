import pickle
from os.path import abspath, dirname, join
import os

DATA_SET_NAME = 'whoiswho_new'
PROJ_DIR = dirname(dirname(__file__))
PARENT_PROJ_DIR = dirname(PROJ_DIR)
RAW_DATA_DIR = join(PARENT_PROJ_DIR, 'sota_data', 'yutao_data', DATA_SET_NAME)
DATA_DIR = join(PROJ_DIR, 'data')
OUT_DIR = join(PROJ_DIR, 'out')
EMB_DATA_DIR = join(DATA_DIR, 'emb')
SPLIT_PATH = join(PARENT_PROJ_DIR, 'split')
GLOBAL_DATA_DIR = join(PROJ_DIR, 'global')
PARENT_FEATURE_DIR = join(PARENT_PROJ_DIR, 'features')
with open(os.path.join(SPLIT_PATH, DATA_SET_NAME), 'rb') as load:
    _, TRAIN_NAME_LIST, VAL_NAME_LIST, TEST_NAME_LIST = pickle.load(load)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EMB_DATA_DIR, exist_ok=True)
os.makedirs(GLOBAL_DATA_DIR, exist_ok=True)
