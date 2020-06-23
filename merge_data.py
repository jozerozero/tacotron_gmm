import os
import random
import shutil
from glob import glob

data_dir_list = ["training_biaobei_wavernn_24k", "training_ljspeech_wavernn_24k"]
root = "/home/lizijian/data0/tacotron_multi_lingual/tacotron"
if not os.path.exists(os.path.join(root, "training_data")):
    os.mkdir(os.path.join(root, "training_data"))

os.mkdir(os.path.join(root, "training_data/mels"))
os.mkdir(os.path.join(root, "training_data/linear"))
os.mkdir(os.path.join(root, "training_data/audio"))

train_file = open(os.path.join(root, "training_data/train.txt"), mode="w")

meta_data_list = list()

for data_dir in data_dir_list:
    data_dir = os.path.join(root, data_dir)

    for data_path in glob(os.path.join(data_dir, "mels", "*.npy")):
        data_name = os.path.basename(data_path)
        new_data_path = os.path.join(root, "training_data", "mels", data_name)
        shutil.copyfile(data_path, new_data_path)

    for data_path in glob(os.path.join(data_dir, "linear", "*.npy")):
        data_name = os.path.basename(data_path)
        new_data_path = os.path.join(root, "training_data", "linear", data_name)
        shutil.copyfile(data_path, new_data_path)

    for data_path in glob(os.path.join(data_dir, "audio", "*.npy")):
        data_name = os.path.basename(data_path)
        new_data_path = os.path.join(root, "training_data", "audio", data_name)
        shutil.copyfile(data_path, new_data_path)

meta_data_list = list()
for index, data_dir in enumerate(data_dir_list):
    for line in open(os.path.join(root, data_dir, "cmu_long.txt")).readlines():
        line = line.strip() + "|%d" % index
        meta_data_list.append(line)

random.shuffle(meta_data_list)

for meta_data in meta_data_list:
    print(meta_data, file=train_file)

