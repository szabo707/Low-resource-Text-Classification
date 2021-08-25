#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import math

import numpy as np
import tensorflow as tf
import transformers
import robeczech_tokenizer

from text_classification_dataset import TextClassificationDataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

maxAcc = 0.1

class Network:
    def __init__(self, args, labels):
        self.strategy = tf.distribute.MirroredStrategy()
        with self.strategy.scope():
            self.model = TFAutoModelForSequenceClassification.from_pretrained("../eol-210409/tf", hidden_dropout_prob=args.dropout,
                                                                              num_labels=labels)
        self.compile_model(False)

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def compile_model(self, bert_trainable):
        with self.strategy.scope():
            self.model.roberta.trainable = bert_trainable
            self.model.compile(tf.keras.optimizers.Adam(),
                               tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                               [tf.keras.metrics.SparseCategoricalAccuracy()])
            self.model.summary()

    def train(self, omr, args):
        init_epochs = args.init
        start_epochs = math.ceil(args.epochs * (args.warmup / 100))
        end_epochs = args.epochs - init_epochs - start_epochs

        init_epochs = range(init_epochs)
        warmup_epochs = range(start_epochs)
        coolingdown_epochs = range(end_epochs)

        def train_epoch(epoch, epochs_size, omr, args, mode):
            global maxAcc

            batch_count = omr.train.size() / args.batch_size
            lr_batch_warmup = args.learning_rate / (epochs_size * batch_count)
            lr_batch_cooling = - args.learning_rate / (epochs_size * batch_count)

            for tokens, labels in omr.train.batches(args.batch_size):
                if mode == "init":
                    self.model.optimizer.learning_rate = 1e-3
                elif mode == "warm":
                    self.model.optimizer.learning_rate = self.model.optimizer.learning_rate + lr_batch_warmup
                else:
                    self.model.optimizer.learning_rate = self.model.optimizer.learning_rate + lr_batch_cooling

                learning_rate = self.model.optimizer.learning_rate

                loss, accuracy = self.model.train_on_batch(tokens, labels)
                step = self.model.optimizer.iterations
                tf.summary.experimental.set_step(step)
                with self._writer.as_default(), tf.summary.record_if(step % 10 == 0):
                    tf.summary.scalar('train/loss', loss)
                    tf.summary.scalar('train/accuracy', accuracy)

                print(learning_rate)

            print('Training epoch: ', epoch + 1, '  Learning rate: ', learning_rate, '  Accuracy: ', accuracy,
                  '  Loss: ', loss)

            self.model.reset_metrics()

            loss_mean = []
            acc_mean = []
            for tokens, labels in omr.dev.batches(args.batch_size):
                loss, accuracy = self.model.test_on_batch(tokens, labels, reset_metrics=False)
                loss_mean.append(loss)
                acc_mean.append(accuracy)

            accuracy = np.mean(acc_mean)
            loss = np.mean(loss_mean)
            print('Dev', epoch + 1, '  Acc: ', accuracy, '  Loss: ', loss)

            if accuracy > maxAcc:
                maxAcc = accuracy
                self.model.save_weights('weights.h5')
                print('!!!!!!!!!!!!!!!!', '  Acc: ', accuracy, '  Loss: ', loss)


        for epoch_init in init_epochs:
            train_epoch(epoch_init, len(init_epochs), omr, args, "init")
        self.compile_model(True)
        self.model.optimizer.learning_rate = 0.0
        for epoch_warm in warmup_epochs:
            train_epoch(epoch_warm, len(warmup_epochs), omr, args, "warm")
        for epoch_cold in coolingdown_epochs:
            train_epoch(epoch_cold, len(coolingdown_epochs), omr, args, "coolingdown")


    def predict(self, dataset, args):
        predictions = []
        for tokens, _ in dataset.batches(args.batch_size):
            prediction = np.argmax(self.model.predict(tokens), axis=-1)
            predictions.append(np.squeeze(prediction, axis=0))
        return np.concatenate(predictions)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--bert", default="jplu/tf-xlm-roberta-base", type=str, help="BERT model.")

    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument("--init", default=1, type=int, help="Number of epochs for activation layer.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning rate.")
    parser.add_argument("--warmup", default=10, type=int, help="Number of % for the linear warmup.")

    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    args.logdir = os.path.join("BERT_final", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    tokenizer = robeczech_tokenizer.RobeCzechTokenizer("../eol-210409/tokenizer")
    tokenizer_encode = lambda sentence: tokenizer.encode(sentence)["input_ids"]


    facebook = TextClassificationDataset("czech_facebook", tokenizer=tokenizer_encode)

    network = Network(args, len(facebook.train.LABELS))
    network.train(facebook, args)
    network.model.load_weights('weights.h5')

    out_path = "sentiment_analysis_predict.txt"

    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for label in network.predict(facebook.test, args):
            print(facebook.test.LABELS[label], file=out_file)
