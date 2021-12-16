import os
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

import argparse
import torch
from torch import nn
from transformers import BertModel
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AdamW
from train.losses import BCEWithLogLoss
from data.dataloader import get_dataloader
from train.metrics import F1Score
from data.preprocess import preprocess_raw_data

raw_data_dir = "/data/hurunyi/MLTC/raw_data"
my_data_dir = "/data/hurunyi/MLTC/my_data"
bert_path = "hfl/chinese-roberta-wwm-ext-large"
checkpoint_dir = "/data/hurunyi/MLTC/checkpoints/Roberta/best"

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
os.environ['TOKENIZERS_PARALLELISM'] = "false"


class PlRobertaMultiLable(pl.LightningModule):
    def __init__(self, bert_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(1024, 148)

    def forward(self, guid, input_ids, attention_mask=None, token_type_ids=None, head_mask=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask
        )
        # (batch, seq_len, hidden_size)
        encoder_out = outputs.last_hidden_state
        # (batch, hidden_size)
        encoder_out = torch.max(encoder_out, dim=1)[0].squeeze(dim=1)
        encoder_out = self.dropout(encoder_out)
        # (batch, num_labels)
        out_logits = self.fc(encoder_out)
        return guid, out_logits

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5, eps=1e-8)
        return {"optimizer": optimizer}

    @staticmethod
    def compute_metric(logits, labels):
        metric = F1Score(average='both', task_type='binary')
        return metric.name(), metric(logits=logits, target=labels)

    def validation_step(self, val_batch, batch_idx):
        guid, input_ids, input_mask, segment_ids, label_ids = val_batch
        guid, logits = self(guid, input_ids, input_mask, segment_ids)
        loss = BCEWithLogLoss()(target=label_ids.cpu().detach(), output=logits.cpu().detach())
        self.log('val_loss', loss)
        logits = logits.cpu().detach()
        labels = label_ids.cpu().detach()
        return {"logits": logits, "labels": labels}

    def validation_epoch_end(self, outputs):
        logits_all = [output["logits"] for output in outputs]
        labels_all = [output["labels"] for output in outputs]
        logits_all = torch.cat(logits_all, dim=0).float().cpu().detach()
        labels_all = torch.cat(labels_all, dim=0).float().cpu().detach()
        name, value = self.compute_metric(logits=logits_all, labels=labels_all)
        if value:
            self.log("valid_f1", value)
            self.print(f"Epoch: {self.current_epoch} | valid_{name}: {value}")

    def training_step(self, train_batch, batch_idx):
        guid, input_ids, input_mask, segment_ids, label_ids = train_batch
        guid, logits = self(guid, input_ids, input_mask, segment_ids)
        loss = BCEWithLogLoss()(output=logits, target=label_ids)
        self.log('train_loss', loss)
        logits = logits.cpu().detach()
        labels = label_ids.cpu().detach()
        return {"loss":loss, "logits": logits, "labels": labels}

    def training_epoch_end(self, outputs):
        logits_all = [output["logits"] for output in outputs]
        labels_all = [output["labels"] for output in outputs]
        logits_all = torch.cat(logits_all, dim=0).float().cpu().detach()
        labels_all = torch.cat(labels_all, dim=0).float().cpu().detach()
        name, value = self.compute_metric(logits=logits_all, labels=labels_all)
        if value:
            self.log("train_f1", value)
            self.print(f"Epoch: {self.current_epoch} | train_{name}: {value}")


def train(args):
    seed_everything(42)
    train_dataloader = get_dataloader(
        data_path=os.path.join(my_data_dir, "train.pkl"),
        max_seq_len=args.train_max_seq_len,
        is_sorted=0,
        batch_size=args.train_batch_size,
        truncation_method=args.truncation_method,
        weighted_sample=args.weighted_sample
    )
    valid_dataloader = get_dataloader(
        data_path=os.path.join(my_data_dir, "valid.pkl"),
        max_seq_len=args.eval_max_seq_len,
        is_sorted=1,
        batch_size=args.eval_batch_size,
        truncation_method=args.truncation_method
    )
    model = PlRobertaMultiLable(bert_path)
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_f1",
        mode="max",
        dirpath=checkpoint_dir,
        filename="best_{valid_f1:.2f}"
    )
    earlystop_callback = EarlyStopping(monitor="valid_f1", mode="max")
    trainer = pl.Trainer(
        gpus=4, precision=16, strategy="ddp", callbacks=[checkpoint_callback, earlystop_callback],
    )
    trainer.fit(model, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--preprocess_data", action="store_true")
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--train_max_seq_len", default=256, type=int)
    parser.add_argument("--eval_max_seq_len", default=256, type=int)
    parser.add_argument("--truncation_method", type=str, default="head")
    parser.add_argument("--weighted_sample", action="store_true")

    args = parser.parse_args()

    if args.preprocess_data:
        preprocess_raw_data(args)

    if args.train:
        train(args)
