import math
import sys

sys.path.append("..")

import os
import warnings
import random
import logging
import pandas as pd
from bert_score import BERTScorer
import jieba
from argparse import ArgumentParser
from preprocessors.processor import Preprocessor
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import MSELoss
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

warnings.simplefilter('ignore')
logger = logging.getLogger(__name__)

raw_data_path = "/data/hurunyi/MLTC/dataset/train.json"
labels_ids_path = "/data/hurunyi/MLTC/dataset/labels_ids.txt"
stopwords_path = "/data/hurunyi/MLTC/dicts/stopwords.txt"
userdict_path = "/data/hurunyi/MLTC/dicts/userdict.dict"


def seed_everything(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	# some cudnn methods can be random even after fixing the seed
	# unless you tell it to be deterministic
	torch.backends.cudnn.deterministic = True


class MyModel(nn.Module):
	def __init__(self):
		super(MyModel, self).__init__()
		self.bert = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")
		for param in self.bert.parameters():
			param.requires_grad = True
		self.dropout = nn.Dropout(0.1)
		self.fc = nn.Linear(768, 1)

	def forward(self, input_ids, attention_mask=None, token_type_ids=None, head_mask=None):
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			head_mask=head_mask
		)
		# (batch_size, sequence_length, hidden_size)
		encoder_out = outputs.last_hidden_state
		# (batch_size, hidden_size)
		encoder_out = torch.max(encoder_out, dim=1)[0].squeeze(dim=1)
		encoder_out = self.dropout(encoder_out)
		# (batch_size, num_labels)
		out_logits = self.fc(encoder_out)
		return out_logits.squeeze(dim=-1).sigmoid()


class Trainer(object):
	def __init__(self, args, model, optimizer, lr_scheduler):
		super(Trainer, self).__init__()
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model = model.to(self.device)
		self.optimizer = optimizer
		self.lr_scheduler = lr_scheduler
		self.loss = MSELoss()
		self.epochs = args.epochs

	def train_epoch(self, train_data):
		self.model.train()
		pbar = tqdm(enumerate(train_data))
		scaler = GradScaler()
		losses = []
		for i, batch in pbar:
			self.optimizer.zero_grad()
			batch = [t.to(self.device) for t in batch]
			input_ids, attention_mask, token_type_ids, labels = batch
			with autocast():
				logits = self.model(input_ids, attention_mask, token_type_ids)
				loss = self.loss(logits, labels.float())
			losses.append(loss.item())
			scaler.scale(loss).backward()
			scaler.step(self.optimizer)
			self.lr_scheduler.step()
			scaler.update()
			if i % 10 == 0:
				pbar.set_postfix(loss=np.mean(losses))

	def valid_epoch(self, valid_data):
		with torch.no_grad():
			self.model.eval()
			losses = []
			for batch in tqdm(valid_data):
				batch = [t.to(self.device) for t in batch]
				input_ids, attention_mask, token_type_ids, labels = batch
				logits = self.model(input_ids, attention_mask, token_type_ids)
				loss = self.loss(logits, labels)
				losses.append(loss.item())
			return np.mean(losses)

	def train(self, train_data, valid_data):
		seed_everything()
		best_loss = math.inf
		for i in range(self.epochs):
			print(f"\nEpoch: {i+1}")
			self.train_epoch(train_data)
			loss = self.valid_epoch(valid_data)
			print(f"Loss: {loss}\tOld best loss: {best_loss}")
			if loss < best_loss:
				best_loss = loss
				print(f"Update the best loss to {best_loss}")
				torch.save(
					{
						"best_loss": best_loss,
						"model": self.model.state_dict(),
						"optimizer": self.optimizer.state_dict(),
						"lr_scheduler": self.lr_scheduler.state_dict(),
					},
					"best_checkpoint.pt"
				)
				print("New best model saved!")

	@staticmethod
	def test():
		import re
		with open(labels_ids_path, "r") as f:
			label_list = [label_id.split()[0] for label_id in f.readlines()]
		id2label = {i: label for i, label in enumerate(label_list)}
		data = pd.read_json("valid.json", lines=True)
		sents = [re.split(r"，|。|；|？|！|：", "".join(row[1]).strip()) for row in data.values]
		for sent in sents:
			if "" in sent:
				sent.remove("")
		# remove the last [] if exists
		# print(sents[:5])
		# exit()
		# sents = [sent if sent[-1] != [] else sent[:-1] for sent in sents]
		labels = [[id2label[label_id] for label_id in row[2]] for row in data.values]
		tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
		model = MyModel().to("cuda")
		check_point = torch.load("best_checkpoint.pt")
		model.load_state_dict(check_point["model"])

		k = 4
		with torch.no_grad():
			model.eval()
			new_sents = []
			for sent_chunks, sent_labels in tqdm(zip(sents, labels)):
				sent_chunks_cut = [sent.strip() for sent in sent_chunks]
				input = tokenizer(sent_chunks_cut, padding=True, truncation=True, max_length=256, return_tensors="pt")
				for key, val in input.items():
					input[key] = val.to("cuda")
				out = model(input["input_ids"], input["attention_mask"], input["token_type_ids"])
				_, topk_ids = torch.topk(out, k if k <= len(sent_chunks_cut) else len(sent_chunks_cut))
				min_i = -1
				for i, id in enumerate(topk_ids):
					if out[id] > 0.5:
						min_i = i + 1
				if min_i == -1:
					min_i = k if k <= len(sent_chunks_cut) else len(sent_chunks_cut)
				new_sent = ""
				for i in range(min_i):
					new_sent += " " + sent_chunks_cut[topk_ids[i]]
				new_sents.append(new_sent)

			with open("valid_data.txt", "w") as f:
				for sent in new_sents:
					f.write(f"{sent}\n")


def get_dataloader(args):
	tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

	with open("chunks.txt", "r") as f:
		lines = f.readlines()
		labels = [int(line[0]) for line in lines]
		chunks = [line[2:] for line in lines]

	print("Prepare the data...")
	train_chunks, valid_chunks, train_labels, valid_labels = train_test_split(chunks[:50000], labels[:50000], train_size=0.8,
																			  random_state=42)
	train_batch = tokenizer(train_chunks, padding=True, truncation=True, max_length=256, return_tensors="pt")
	train_batch["labels"] = torch.tensor(train_labels)
	valid_batch = tokenizer(valid_chunks, padding=True, truncation=True, max_length=256, return_tensors="pt")
	valid_batch["labels"] = torch.tensor(valid_labels)
	print("Finished!")

	train_dataset = TensorDataset(
		train_batch["input_ids"],
		train_batch["attention_mask"],
		train_batch["token_type_ids"],
		train_batch["labels"]
	)
	valid_dataset = TensorDataset(
		valid_batch["input_ids"],
		valid_batch["attention_mask"],
		valid_batch["token_type_ids"],
		valid_batch["labels"]
	)

	train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)
	valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)

	return train_dataloader, valid_dataloader


def preprocess():
	import re
	with open(labels_ids_path, "r") as f:
		label_list = [label_id.split()[0] for label_id in f.readlines()]
	id2label = {i: label for i, label in enumerate(label_list)}

	preprocessor = Preprocessor("ChineseChar")(stopwords_path=stopwords_path, userdict_path=userdict_path)
	scorer = BERTScorer(lang="zh", rescale_with_baseline=True)

	"""
	data[i]: line in lines
	data[i][0]: id num(int), data[i][1]: sent tokens(list)
	data[i][2]: labels(list), data[i][3]: label num(int)
	"""
	data = pd.read_json(raw_data_path, lines=True)
	sents = [re.split(r"，|。|；|？|！|：", "".join(row[1]).strip()) for row in data.values]
	labels = [[id2label[label_id] for label_id in row[2]] for row in data.values]
	for sent in sents:
		if "" in sent:
			sent.remove("")

	k = 2
	new_sents = []
	chunks_select = []
	chunks_ignore = []
	for sent_chunks, sent_labels in tqdm(zip(sents, labels)):
		sent_chunks_cut = [sent.strip() for sent in sent_chunks]
		sent_labels_cut = [label.strip() for label in sent_labels]
		new_sent = []
		new_sent_scores = []
		for i, sent_label_cut in enumerate(label_list):
			sent_label_cut_list = [sent_label_cut for _ in range(len(sent_chunks_cut))]
			P, R, F1 = scorer.score(sent_chunks_cut, sent_label_cut_list)
			_, topk_ids = torch.topk(F1, k if k <= len(sent_chunks_cut) else len(sent_chunks_cut))
			for id in topk_ids:
				if sent_chunks_cut[id] not in new_sent and F1[id] > 0.2:
					new_sent.append(sent_chunks_cut[id])
					# chunks_select.append(sent_chunks_cut[id])
					new_sent_scores.append(F1[id].item())
			# for id in range(len(sent_chunks_cut)):
			# 	if sent_chunks_cut[id] not in new_sent:
			# 		chunks_ignore.append(sent_chunks_cut[id])
		final_sent = []
		new_sent_scores = torch.tensor(new_sent_scores)
		_, topk_score_ids = torch.topk(new_sent_scores, 4 if 4 < len(new_sent_scores) else len(new_sent_scores))
		for id in topk_score_ids:
			final_sent.append(new_sent[id])
		print(final_sent)
		new_sents.append(" ".join(final_sent))

	with open("valid_data.txt", "w") as f:
		for sent in new_sents:
			f.write(f"{sent}\n")
	#
	# with open("chunks.txt", "w") as f:
	# 	for select in chunks_select:
	# 		f.write(f"1 {select}\n")
	# 	for ignore in chunks_ignore:
	# 		f.write(f"0 {ignore}\n")


def split_valid():
	from sklearn.model_selection import train_test_split
	import json
	from utils.utils import save_pickle

	valid = pd.read_json("valid.json", lines=True)
	# train, valid = train_test_split(data, test_size=0.2, random_state=42)
	# valid.to_json("valid.json", ensure_ascii=False)

	# with open("valid.json", "w") as f:
	# 	for data in valid.values:
	# 		out_dict = {"textid": data[0], "sent": data[1], "labels": data[2]}
	# 		f.write(json.dumps(out_dict, ensure_ascii=False)+"\n")

	with open("/home/hurunyi/MLTC/data_chunk_select/valid_data.txt", "r") as f:
		lines = f.readlines()
		sents = [line for line in lines]

	ids, labels = [], []
	for data in valid.values:
		ids.append(data[0])
		labels.append(data[2])

	data = []
	for i in range(len(sents)):
		target = np.zeros(148)
		for label in labels[i]:
			target[label] = 1
		data.append((ids[i], sents[i], target))

	valid_path = "/data/hurunyi/MLTC/dataset/law.valid.pkl"
	save_pickle(data=data, file_path=valid_path)

	import re
	with open(labels_ids_path, "r") as f:
		label_list = [label_id.split()[0] for label_id in f.readlines()]
	id2label = {i: label for i, label in enumerate(label_list)}

	scorer = BERTScorer(lang="zh", rescale_with_baseline=True)

	"""
	data[i]: line in lines
	data[i][0]: id num(int), data[i][1]: sent tokens(list)
	data[i][2]: labels(list), data[i][3]: label num(int)
	"""
	data = pd.read_json("valid", lines=True)
	sents = [re.split(r"，|。|；|？|！|：", "".join(row[1]).strip()) for row in data.values]
	labels = [[id2label[label_id] for label_id in row[2]] for row in data.values]
	for sent in sents:
		if "" in sent:
			sent.remove("")

	k = 2
	new_sents = []
	chunks_select = []
	chunks_ignore = []
	for sent_chunks in tqdm(sents):
		# flag = 0
		sent_chunks_cut = [sent.strip() for sent in sent_chunks]
		new_sent = []
		chunks_list = [sent_chunks_cut for _ in range(148)]
		P, R, F1 = scorer.score(sent_chunks_cut, label_list)
		print(F1)
		_, topk_ids = torch.topk(F1, k if k <= len(sent_chunks_cut) else len(sent_chunks_cut))
		for id in topk_ids:
			if sent_chunks_cut[id] not in new_sent:
				new_sent.append(sent_chunks_cut[id])
				chunks_select.append(sent_chunks_cut[id])
		for id in range(len(sent_chunks_cut)):
			if sent_chunks_cut[id] not in new_sent:
				chunks_ignore.append(sent_chunks_cut[id])
		new_sents.append(" ".join(new_sent))

	with open("hurunyi_data.txt", "w") as f:
		for sent in new_sents:
			f.write(f"{sent}\n")


def main(args):
	train_dataloader, valid_dataloader = get_dataloader(args)
	model = MyModel()
	optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
	t_total = int(len(train_dataloader) * args.epochs)
	warmup_steps = int(t_total * args.warmup_proportion)
	lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
	trainer = Trainer(args, model, optimizer, lr_scheduler)
	trainer.train(train_dataloader, valid_dataloader)


if __name__ == "__main__":
	parser = ArgumentParser()

	parser.add_argument("--train_batch_size", type=int, default=64)
	parser.add_argument("--valid_batch_size", type=int, default=64)
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--lr", type=float, default=2e-5)
	parser.add_argument("--warmup_proportion", type=float, default=0.1)

	args = parser.parse_args()

	preprocess()
	# main(args)
	# Trainer.test()
	# split_valid()
