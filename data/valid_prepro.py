import json
import h5py
from tqdm import tqdm

PATH = 'cocoqa_data_prepro.json'
PATH_BACK = 'backup/cocoqa_data_prepro.json'


with open(PATH, 'r') as f1:
    js1 = json.load(f1)

with open(PATH_BACK, 'r') as f2:
    js2 = json.load(f2)

for k in js1.keys():
    print(k)
    assert js2[k] == js1[k]


