import pickle as pkl
from joblib import Parallel, delayed

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler, Dataset, DataLoader


def has_answer(answer_pos):
    if answer_pos[0] == 0 and answer_pos[1] == 0:
        return 0
    else:
        return 1


def find_tokenized_answer(tokenizer, document, answer, input_ids):
    span = None

    if (answer or (" " + answer)) not in document:
        idx = [0, 0]
        answerable = 0
        span = [idx, answerable]

    if (" " + answer) in document:
        ans_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]

    else:
        ans_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    for i in range(len(input_ids) - len(ans_ids) + 1):

        if input_ids[i : i + len(ans_ids)] == ans_ids:
            idx = [i, i + len(ans_ids)]
            answerable = 1
            span = [idx, answerable]

    if span == None:
        idx = [0, 0]
        answerable = 0
        span = [idx, answerable]

    return span


def tokenized_examples(documents, answers, tokenizer, doc_id):

    tokenized_examples = tokenizer(documents, max_length=512, padding="max_length")
    input_ids, attention_mask = (
        tokenized_examples["input_ids"],
        tokenized_examples["attention_mask"],
    )

    ans_tok_pos = Parallel(n_jobs=1)(
        delayed(find_tokenized_answer)(tokenizer, document, answer, ids)
        for document, answer, ids in zip(documents, answers, input_ids)
    )
    ans_tok_pos = np.array(ans_tok_pos, dtype=object)

    if doc_id is not None:
        examples = np.zeros((len(documents), 6), dtype=object)
        examples[:, 5] = doc_id

    else:
        examples = np.zeros((len(documents), 5), dtype=object)
    examples[:, 0] = input_ids
    examples[:, 1] = attention_mask
    examples[:, 2] = ans_tok_pos[:, 0]
    examples[:, 3] = ans_tok_pos[:, 1]
    examples[:, 4] = answers

    return examples


class build_dataset(Dataset):
    def __init__(self, documents, answers, tokenizer, doc_id=None):
        self.examples = tokenized_examples(documents, answers, tokenizer, doc_id)
        self.doc_id = doc_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def batchify_feature(batch, mode):
    input_ids = [ex[0] for ex in batch]
    attention_mask = [ex[1] for ex in batch]
    ans_tok_pos = [ex[2] for ex in batch]
    answerable = [ex[3] for ex in batch]

    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(attention_mask)
    ans_tok_pos = torch.LongTensor(ans_tok_pos)
    answerable = torch.LongTensor(answerable)

    if mode == "dev":
        answer = [ex[4] for ex in batch]
        return input_ids, attention_mask, ans_tok_pos, answerable, answer

    if mode == "test":
        answer = [ex[4] for ex in batch]
        doc_id = [ex[5] for ex in batch]
        return input_ids, attention_mask, ans_tok_pos, answerable, answer, doc_id

    return input_ids, attention_mask, ans_tok_pos, answerable


def generate_data_loader(batch_size, train_path, dev_path, test_path, tokenizer):

    train_df = pkl.load(open(train_path, "rb"))
    dev_df = pkl.load(open(dev_path, "rb"))
    test_df = pkl.load(open(test_path, "rb"))

    train_df = train_df.head(2000)
    dev_df = dev_df.head(400)
    test_df = test_df.head(400)

    train_dataset = build_dataset(
        train_df.document.to_list(), train_df.answer.to_list(), tokenizer
    )
    train_sampler = RandomSampler(train_dataset)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=lambda x: batchify_feature(x, "train"),
    )

    dev_dataset = build_dataset(
        dev_df.document.to_list(), dev_df.answer.to_list(), tokenizer
    )
    dev_sampler = RandomSampler(dev_dataset)
    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        sampler=dev_sampler,
        pin_memory=True,
        collate_fn=lambda x: batchify_feature(x, "dev"),
    )

    test_dataset = build_dataset(
        test_df.document.to_list(),
        test_df.answer.to_list(),
        tokenizer,
        doc_id=test_df.index.to_list(),
    )
    test_sampler = SequentialSampler(test_dataset)
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        pin_memory=True,
        collate_fn=lambda x: batchify_feature(x, "test"),
    )

    return (
        train_dataset,
        train_data_loader,
        dev_dataset,
        dev_data_loader,
        test_dataset,
        test_data_loader,
    )
