import numpy as np
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler, TensorDataset
from utils.utils import load_pickle
from callback.progressbar import ProgressBar


vocab_path = "hfl/chinese-roberta-wwm-ext-large"


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeature(object):
    def __init__(self, guid, input_ids, input_mask, segment_ids, label_id, input_len):
        self.guid = guid
        self.label_id = label_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_len = input_len


class BertProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, vocab_path):
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_path)

    @staticmethod
    def create_examples(lines):
        pbar = ProgressBar(n_total=len(lines))
        examples = []
        for i, line in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            label = line[2]
            if isinstance(label, str):
                label = [np.float(x) for x in label.split(",")]
            else:
                label = [np.float(x) for x in list(label)]
            text_b = None
            example = InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)
            pbar.batch_step(step=i, info={}, bar_type='create examples')
        return examples

    def create_features(self, examples, max_seq_len, truncation_method):
        pbar = ProgressBar(n_total=len(examples))
        features = []
        for ex_id, example in enumerate(examples):
            guid = example.guid
            tokens_a = self.tokenizer.tokenize(example.text_a)
            label_id = example.label

            # Account for [CLS] and [SEP] with '-2'
            if len(tokens_a) > max_seq_len - 2:
                if truncation_method == "head":
                    tokens_a = tokens_a[:max_seq_len - 2]
                elif truncation_method == "head_tail":
                    tokens_a = tokens_a[:128] + tokens_a[-(max_seq_len - 2 - 128):]
                else:
                    tokens_a = tokens_a[:max_seq_len - 2]

            tokens = ['<s>'] + tokens_a + ['</s>']
            segment_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_len - len(input_ids))
            input_len = len(input_ids)

            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_len
            assert len(input_mask) == max_seq_len
            assert len(segment_ids) == max_seq_len

            # if ex_id < 2:
            #     print("\n*** Example ***")
            #     print(f"guid: {example.guid}" % ())
            #     print(f"tokens: {' '.join([str(x) for x in tokens])}")
            #     print(f"input_ids: {' '.join([str(x) for x in input_ids])}")
            #     print(f"input_mask: {' '.join([str(x) for x in input_mask])}")
            #     print(f"segment_ids: {' '.join([str(x) for x in segment_ids])}")

            feature = InputFeature(
                guid = guid,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                input_len=input_len
            )
            features.append(feature)
            pbar.batch_step(step=ex_id, info={}, bar_type='create features')
        return features

    @staticmethod
    def create_dataset(features):
        all_guid = torch.tensor(
            [f.guid for f in features], dtype=torch.long
        )
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.long
        )
        dataset = TensorDataset(
            all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )
        return dataset


def get_weights(dataset):
    weights = []
    with open("label_freq.txt", "r") as f:
        freq = [line.split()[1] for line in f.readlines()]

    for data in dataset:
        weight = 0
        label = data[4]
        for val in np.nonzero(label)[0]:
            weight += int(freq[val.item()])
        weights.append(100/weight)
    assert len(dataset) == len(weights)
    return weights


def get_dataloader(data_path, max_seq_len, is_sorted,
                   batch_size, truncation_method, weighted_sample=False):
    processor = BertProcessor(vocab_path)
    data = load_pickle(data_path)
    examples = processor.create_examples(lines=data)
    features = processor.create_features(
        examples=examples,
        max_seq_len=max_seq_len,
        truncation_method=truncation_method
    )
    dataset = processor.create_dataset(features)
    if is_sorted:
        sampler = SequentialSampler(dataset)
    elif weighted_sample:
        sampler = WeightedRandomSampler(get_weights(dataset), len(dataset))
    else:
        sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=32)
    return dataloader
