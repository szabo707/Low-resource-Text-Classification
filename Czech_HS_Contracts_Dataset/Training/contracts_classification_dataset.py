# The tests depend on used the text of the contracts:
# 1 - metainfo
# 2 - subject
# 3 - first 400 tokens of text
# 4 - first 400 of (metainfo + text)
# 5 - first 400 of (subject + text)
# 6 - first window
# 7 - metainfo + first window
# 8 - subject + first window
# 9 - all windows
# 10 - metainfo + all windows
# 11 - subject + all windows
# 12 - first two windows
# 13 - metainfo + first two windows
# 14 - subject + first two windows
# 15 - first 500 tokens of text
# 16 - first 500 of (metainfo + text)
# 17 - first 500 of (subject + text)
#
#
# with argument "main_cat" train only main categories

import os
import sys
import urllib.request
import zipfile
import jsonlines
import json

import numpy as np
from json_parser import ContractsParser

class ContractsClassificationDataset:

    class Dataset:
        LABELS = None

        def __init__(self, data_file, tokenizer, train=None, shuffle_batches=True, seed=42, test=0, main_cat=None):
            self._data = {
                "tokens": [],
                "labels": [],
                "contracts": [],
            }
            self._label_map = train._label_map if train else {}
            self.LABELS = train.LABELS if train else []

            for contract_ID, line in enumerate(data_file):
                line = line.decode("utf-8").rstrip("\r\n")
                data = json.loads(line)
                contract = ContractsParser(data)

                ranges = contract.getWindowsRanges()
                plainText = contract.getPlainTextContent()
                predmet = contract.getPredmet()

                label = int(contract.getLabel())

                if main_cat:
                    # for training main categories
                    # get main category
                    label = round(label, -2)
                    #print(label)

                prijemce = contract.getPrijemce('ALL')
                platce = contract.getPlatce('ALL')

                if test == 1:
                    text = predmet + ". " + platce + ". " + prijemce
                    if not train and label not in self._label_map:
                        self._label_map[label] = len(self._label_map)
                        self.LABELS.append(label)
                    label = self._label_map.get(label, -1)
                    self._data["tokens"].append(tokenizer(text))
                    self._data["labels"].append(label)
                    self._data["contracts"].append(contract_ID)

                elif test == 2:
                    text = predmet
                    if not train and label not in self._label_map:
                        self._label_map[label] = len(self._label_map)
                        self.LABELS.append(label)
                    label = self._label_map.get(label, -1)
                    self._data["tokens"].append(tokenizer(text))
                    self._data["labels"].append(label)
                    self._data["contracts"].append(contract_ID)

                elif test == 3:
                    text = plainText
                    if not train and label not in self._label_map:
                        self._label_map[label] = len(self._label_map)
                        self.LABELS.append(label)
                    label = self._label_map.get(label, -1)

                    tokens = tokenizer(text[:16000])
                    tokens = tokens[:-2][:400] + tokens[-2:]
                    self._data["tokens"].append(tokens)
                    self._data["labels"].append(label)
                    self._data["contracts"].append(contract_ID)

                elif test == 4:
                    text = predmet + ". " + platce + ". " + prijemce + ". " + plainText
                    if not train and label not in self._label_map:
                        self._label_map[label] = len(self._label_map)
                        self.LABELS.append(label)
                    label = self._label_map.get(label, -1)

                    tokens = tokenizer(text[:16000])
                    tokens = tokens[:-2][:400] + tokens[-2:]
                    self._data["tokens"].append(tokens)
                    self._data["labels"].append(label)
                    self._data["contracts"].append(contract_ID)

                elif test == 5:
                    text = predmet + ". " + plainText
                    if not train and label not in self._label_map:
                        self._label_map[label] = len(self._label_map)
                        self.LABELS.append(label)
                    label = self._label_map.get(label, -1)

                    tokens = tokenizer(text[:16000])
                    tokens = tokens[:-2][:400] + tokens[-2:]
                    self._data["tokens"].append(tokens)
                    self._data["labels"].append(label)
                    self._data["contracts"].append(contract_ID)

                elif test == 6:
                    text_contract = plainText[ranges[0][0] + 1:ranges[0][1]]
                    text = text_contract

                    if not train and label not in self._label_map:
                        self._label_map[label] = len(self._label_map)
                        self.LABELS.append(label)
                    label = self._label_map.get(label, -1)
                    self._data["tokens"].append(tokenizer(text))
                    self._data["labels"].append(label)
                    self._data["contracts"].append(contract_ID)

                elif test == 7:
                    text_contract = plainText[ranges[0][0] + 1:ranges[0][1]]
                    text = predmet + ". " + platce + ". " + prijemce + ". " + text_contract

                    if not train and label not in self._label_map:
                        self._label_map[label] = len(self._label_map)
                        self.LABELS.append(label)
                    label = self._label_map.get(label, -1)
                    self._data["tokens"].append(tokenizer(text))
                    self._data["labels"].append(label)
                    self._data["contracts"].append(contract_ID)

                elif test == 8:
                    text_contract = plainText[ranges[0][0] + 1:ranges[0][1]]
                    text = predmet + ". " + text_contract

                    if not train and label not in self._label_map:
                        self._label_map[label] = len(self._label_map)
                        self.LABELS.append(label)
                    label = self._label_map.get(label, -1)
                    self._data["tokens"].append(tokenizer(text))
                    self._data["labels"].append(label)
                    self._data["contracts"].append(contract_ID)

                elif test == 9:
                    for w_start, w_end in ranges:
                        text_contract = plainText[w_start + 1:w_end]
                        text = text_contract

                        if not train and label not in self._label_map:
                            self._label_map[label] = len(self._label_map)
                            self.LABELS.append(label)
                        label = self._label_map.get(label, -1)
                        self._data["tokens"].append(tokenizer(text))
                        self._data["labels"].append(label)
                        self._data["contracts"].append(contract_ID)

                elif test == 10:
                    for w_start, w_end in ranges:
                        text_contract = plainText[w_start + 1:w_end]
                        text = predmet + ". " + platce + ". " + prijemce + ". " + text_contract

                        if not train and label not in self._label_map:
                            self._label_map[label] = len(self._label_map)
                            self.LABELS.append(label)
                        label = self._label_map.get(label, -1)
                        self._data["tokens"].append(tokenizer(text))
                        self._data["labels"].append(label)
                        self._data["contracts"].append(contract_ID)

                elif test == 11:
                    for w_start, w_end in ranges:
                        text_contract = plainText[w_start + 1:w_end]
                        text = predmet + ". " + text_contract

                        if not train and label not in self._label_map:
                            self._label_map[label] = len(self._label_map)
                            self.LABELS.append(label)
                        label = self._label_map.get(label, -1)
                        self._data["tokens"].append(tokenizer(text))
                        self._data["labels"].append(label)
                        self._data["contracts"].append(contract_ID)

                elif test == 12:
                    for w_start, w_end in ranges[0:2]:
                        text_contract = plainText[w_start + 1:w_end]
                        text = text_contract

                        if not train and label not in self._label_map:
                            self._label_map[label] = len(self._label_map)
                            self.LABELS.append(label)
                        label = self._label_map.get(label, -1)
                        self._data["tokens"].append(tokenizer(text))
                        self._data["labels"].append(label)
                        self._data["contracts"].append(contract_ID)

                elif test == 13:
                    for w_start, w_end in ranges[0:2]:
                        text_contract = plainText[w_start + 1:w_end]
                        text = predmet + ". " + platce + ". " + prijemce + ". " + text_contract

                        if not train and label not in self._label_map:
                            self._label_map[label] = len(self._label_map)
                            self.LABELS.append(label)
                        label = self._label_map.get(label, -1)
                        self._data["tokens"].append(tokenizer(text))
                        self._data["labels"].append(label)
                        self._data["contracts"].append(contract_ID)

                elif test == 14:
                    for w_start, w_end in ranges[0:2]:
                        text_contract = plainText[w_start + 1:w_end]
                        text = predmet + ". " + text_contract

                        if not train and label not in self._label_map:
                            self._label_map[label] = len(self._label_map)
                            self.LABELS.append(label)
                        label = self._label_map.get(label, -1)
                        self._data["tokens"].append(tokenizer(text))
                        self._data["labels"].append(label)
                        self._data["contracts"].append(contract_ID)

                elif test == 15:
                    text = plainText
                    if not train and label not in self._label_map:
                        self._label_map[label] = len(self._label_map)
                        self.LABELS.append(label)
                    label = self._label_map.get(label, -1)

                    tokens = tokenizer(text[:16000])
                    tokens = tokens[:-2][:500] + tokens[-2:]
                    self._data["tokens"].append(tokens)
                    self._data["labels"].append(label)
                    self._data["contracts"].append(contract_ID)

                elif test == 16:
                    text = predmet + ". " + platce + ". " + prijemce + ". " + plainText
                    if not train and label not in self._label_map:
                        self._label_map[label] = len(self._label_map)
                        self.LABELS.append(label)
                    label = self._label_map.get(label, -1)

                    tokens = tokenizer(text[:16000])
                    tokens = tokens[:-2][:500] + tokens[-2:]
                    self._data["tokens"].append(tokens)
                    self._data["labels"].append(label)
                    self._data["contracts"].append(contract_ID)

                elif test == 17:
                    text = predmet + ". " + plainText
                    if not train and label not in self._label_map:
                        self._label_map[label] = len(self._label_map)
                        self.LABELS.append(label)
                    label = self._label_map.get(label, -1)

                    tokens = tokenizer(text[:16000])
                    tokens = tokens[:-2][:500] + tokens[-2:]
                    self._data["tokens"].append(tokens)
                    self._data["labels"].append(label)
                    self._data["contracts"].append(contract_ID)

                else:
                    print("Missing correct test argument")

            self._size = len(self._data["tokens"])
            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def data(self):
            return self._data

        def size(self):
            return self._size

        def batches(self, size=None):
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            data_tokens = self._data["tokens"]
            data_labels = self._data["labels"]
            data_contracts = self._data["contracts"]

            while len(permutation):
                if size is None:
                    # Determine batch size so that single contract windows form a batch
                    contract = data_contracts[permutation[0]]
                    batch_size = 1
                    while batch_size < len(permutation) and data_contracts[permutation[batch_size]] == contract:
                        batch_size += 1
                else:
                    batch_size = min(size, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                max_sentence_len = max(len(data_tokens[i]) for i in batch_perm)
                tokens = np.zeros([batch_size, max_sentence_len], np.int32)
                labels = np.zeros([batch_size], np.int32)
                for i in range(batch_size):
                    tokens[i, :len(data_tokens[batch_perm[i]])] = data_tokens[batch_perm[i]]
                    labels[i] = data_labels[batch_perm[i]]

                yield tokens, labels

    def __init__(self, dataset, tokenizer, test, main_cat):

        path = "{}.zip".format(dataset)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.jsonl".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    setattr(self, dataset, self.Dataset(dataset_file, tokenizer,
                                                        train=self.train if dataset != "train" else None,
                                                        shuffle_batches=dataset == "train",
                                                        test=test,
                                                        main_cat=main_cat))
