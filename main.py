import os
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

import argparse
import warnings
from data.dataloader import get_dataloader
from data.preprocess import preprocess_raw_data
from utils.utils import seed_everything
from train.losses import BCEWithLogLoss
from train.trainer import Trainer
from train.metrics import AccuracyThresh, MultiLabelReport, F1Score
from transformers import AdamW, get_linear_schedule_with_warmup
from model.my_model import RobertaMultiLable

warnings.simplefilter('ignore')


raw_data_dir = "/data/hurunyi/MLTC/raw_data"
my_data_dir = "/data/hurunyi/MLTC/my_data"
bert_path = "hfl/chinese-roberta-wwm-ext-large"


def train(args):
    seed_everything(args.seed)
    print("Training/evaluation parameters %s", args)

    ########### data ###########
    with open(os.path.join(raw_data_dir, "labels_ids.txt"), "r") as f:
        label_list = [label_id.split('\t')[0] for label_id in f.readlines()]
    id2label = {i: label for i, label in enumerate(label_list)}

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

    ########### model ###########
    print("\n=========  initializing model =========")
    model = RobertaMultiLable(bert_path)
    t_total = int(len(train_dataloader) * args.epochs)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total
    )

    ########### train ###########
    print("\n========= Running training =========")
    print("Num examples = {}".format(len(train_dataloader)))
    print("Num Epochs = {}".format(args.epochs))
    print("Total optimization steps = {}".format(t_total))

    trainer = Trainer(
        model=model,
        epochs=args.epochs,
        criterion=BCEWithLogLoss(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        grad_clip=args.grad_clip,
        batch_metrics=[AccuracyThresh(thresh=0.5)],
        epoch_metrics=[F1Score(average='both', task_type='binary'),
                       MultiLabelReport(id2label=id2label)],
        load_from_checkpoint=args.load_from_checkpoint,
        checkpoint_dir=args.checkpoint_dir
    )
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader, seed=args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--preprocess_data", action="store_true")
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--train_max_seq_len", default=256, type=int)
    parser.add_argument("--eval_max_seq_len", default=256, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--load_from_checkpoint", action="store_true")
    parser.add_argument("--truncation_method", type=str, default="head")
    parser.add_argument("--weighted_sample", action="store_true")

    args = parser.parse_args()

    if args.preprocess_data:
        preprocess_raw_data(args)

    if args.train:
        train(args)
