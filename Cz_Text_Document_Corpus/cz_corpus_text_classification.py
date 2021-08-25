import os
import sys
import urllib.request
import zipfile

import numpy as np

class TextClassificationDataset:

    class Dataset:

        LABELS = ['pol', 'zak', 'pod', 'efm', 'spo', 'mag', 'for', 'kul', 'dpr', 'zdr', 'mak', 'slz', 'spl',
                  'sta', 'ene', 'bur', 'ekl', 'met', 'fin', 'obo', 'den', 'pit', 'eur', 'odb', 'str', 'sop',
                  'zem', 'tur', 'sko', 'aut', 'bup', 'mix', 'prg', 'sur', 'med', 'hok', 'ptr']

        def __init__(self, data_file, tokenizer, train=None, shuffle_batches=True, seed=42):
            self._data = {
                "tokens": [],
                "labels": [],
            }
            self._label_map = train._label_map if train else {}

            for line in data_file:
                line = line.decode("utf-8").rstrip("\r\n")
                label_names, text = line.split("\t", maxsplit=1)

                label = np.zeros(len(self.LABELS), np.int32)
                for single_label in label_names.split():
                    if single_label in self.LABELS:
                        label[self.LABELS.index(single_label)] = 1
                    else:
                        continue

                self._data["tokens"].append(tokenizer(text))
                self._data["labels"].append(label)

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

            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                max_sentence_len = max(len(data_tokens[i]) for i in batch_perm)

                tokens = np.zeros([batch_size, max_sentence_len], np.int32)
                labels = np.zeros([batch_size, len(self.LABELS)], np.int32)
                for i in range(batch_size):
                    tokens[i, :len(data_tokens[batch_perm[i]])] = data_tokens[batch_perm[i]]
                    labels[i] = data_labels[batch_perm[i]]

                yield tokens, labels

    def __init__(self, dataset, tokenizer):

        path = "{}.zip".format(dataset)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    setattr(self, dataset, self.Dataset(dataset_file, tokenizer,
                                                        train=self.train if dataset != "train" else None,
                                                        shuffle_batches=dataset == "train"))
