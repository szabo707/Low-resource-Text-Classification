#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import math

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import transformers
import robeczech_tokenizer

from cz_corpus_text_classification import TextClassificationDataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

maxAcc = 0.4

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
                               tf.keras.losses.BinaryCrossentropy(from_logits=True),
                               tfa.metrics.F1Score(num_classes=37, average="micro", threshold=0.5))
            self.model.summary()

    def train(self, omr, args):
        init_epochs = args.init
        start_epochs = math.ceil(args.epochs * (args.warmup / 100))
        end_epochs = args.epochs - init_epochs - start_epochs

        init_epochs = range(init_epochs)
        warmup_epochs = range(start_epochs)
        coolingdown_epochs = range(end_epochs)

        self.model.optimizer.learning_rate = 0.0

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
                learning_rate = self.model.optimizer.learning_rate  # len kvoli vypisu learning_rate

                loss, accuracy = self.model.train_on_batch(tokens, labels)
                step = self.model.optimizer.iterations
                tf.summary.experimental.set_step(step)
                with self._writer.as_default(), tf.summary.record_if(step % 10 == 0):
                    tf.summary.scalar('train/loss', loss)
                    tf.summary.scalar('train/accuracy', accuracy)

            self.model.reset_metrics()
            # develop evaluate
            for tokens, labels in omr.dev.batches(args.batch_size):
                loss, accuracy = self.model.test_on_batch(tokens, labels, reset_metrics=False)

            self.model.reset_metrics()
            # test evaluate
            for tokens, labels in omr.test.batches(args.batch_size):
                test_loss, test_accuracy = self.model.test_on_batch(tokens, labels, reset_metrics=False)
            #print('Test', epoch + 1, '  Acc: ', test_accuracy, '  Loss: ', test_loss)

            if accuracy > maxAcc:
                maxAcc = accuracy
                self.model.save_weights('weights.h5')
                print('RESULT: ', epoch + 1,  ' Acc: ', test_accuracy)

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
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--bert", default="jplu/tf-xlm-roberta-base", type=str, help="BERT model.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs and learning rate.")
    parser.add_argument("--init", default=2, type=int, help="Number of epochs for activation layer.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate.")
    parser.add_argument("--warmup", default=10, type=int, help="Number of % for the linear warmup.")

    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("BERT", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))
    
    print(args.logdir)

    tokenizer = robeczech_tokenizer.RobeCzechTokenizer("../eol-210409/tokenizer")
    tokenizer_encode = lambda sentence: tokenizer.encode(sentence)["input_ids"]

    cz_corpus = TextClassificationDataset("cz_corpus_5", tokenizer=tokenizer_encode)

    network = Network(args, len(cz_corpus.train.LABELS))
    network.train(cz_corpus, args)
    network.model.load_weights('weights.h5')

    out_path = "cz_corpus_predict.txt"

    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for label in network.predict(cz_corpus.test, args):
            print(cz_corpus.test.LABELS[label], file=out_file)
