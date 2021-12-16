import os
import pnlp
import pandas as pd
from utils.utils import save_pickle
from sklearn.model_selection import train_test_split


raw_data_dir = "/data/hurunyi/MLTC/raw_data"
my_data_dir = "/data/hurunyi/MLTC/my_data"


def train_val_split(text_id: list, X: list, y: list, valid_size: float, data_dir=None, save=True):
	print("split train data into train and valid!")
	data = []
	for i in range(len(X)):
		data.append((text_id[i], X[i], y[i]))
	train, valid = train_test_split(data, test_size=valid_size, random_state=42)
	if save:
		train_path = os.path.join(data_dir, "train.pkl")
		valid_path = os.path.join(data_dir, "valid.pkl")
		save_pickle(data=train, file_path=train_path)
		save_pickle(data=valid, file_path=valid_path)


def read_raw_data():
	labels, targets, sents, ids, sent_lens = [], [], [], [], []
	data = pd.read_json(os.path.join(raw_data_dir, "train.json"), lines=True)
	all_cates = len(pnlp.read_lines(os.path.join(raw_data_dir, "labels_ids.txt")))

	for cate in range(all_cates):
		data[cate] = data["labels_index"].apply(lambda x: int(cate in x))

	for row in data.values:
		label = row[2]
		target = row[4:]
		sent = "".join(row[1])
		id = row[0]
		sent_len = len(row[1])
		if sent:
			labels.append(label)
			targets.append(target)
			sents.append(sent)
			ids.append(id)
			sent_lens.append(sent_len)

	return labels, targets, sents, ids, sent_lens


def preprocess_raw_data(args):
	labels, targets, _, ids, sent_lens = read_raw_data()

	with open("/home/hurunyi/MLTC/data_chunk_select/wang_lei_data.txt", "r") as f:
		lines = f.readlines()
		sents = [line for line in lines]

	train_val_split(
		text_id=ids, X=sents, y=targets,
		valid_size=args.valid_size,
		data_dir=my_data_dir
	)
