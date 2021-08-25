import os
import glob
import numpy as np
import robeczech_tokenizer


# JOBS:
# FOLDS - create 5 folds set for training
# CROP  - crop the document to the length of 300 tokens
# DEV   - create dev file for experiments training

JOB = 'DEV'
LIMIT = 300

tokenizer = robeczech_tokenizer.RobeCzechTokenizer("../tokenizer")
tokenizer_encode = lambda sentence: tokenizer.encode(sentence)["input_ids"]

def wordsIndex(text):
    words = [0]
    for i in range(len(text) - 1):
        if text[i].isspace():
            words.append(i)
    return np.array(words)

def wordSubwords(text):
    words = wordsIndex(text)
    subwords_len = []
    for i in range(len(words)):
        word_text = text[words[i]:words[i + 1] if i + 1 < len(words) else len(text)]
        subwords_len.append(len(tokenizer.encode(word_text, add_special_tokens=False)["input_ids"]))
    return np.array(subwords_len)

def getfirstXtokens(lines, out_path):
    counter = 0
    with open(out_path, "a", encoding='utf-8') as out_file:
        for line in lines:
            cat = line.split("\t")[0]
            text = line.split("\t")[1]
            tokens_number = len(tokenizer.encode(text)["input_ids"])
            tokens_sum = 0
            if tokens_number > LIMIT:
                words_index = wordsIndex(text)
                words_subwords = wordSubwords(text)
                tokens_sum = 0
                for i in range(len(words_subwords)):
                    if tokens_sum > LIMIT:
                        break
                    tokens_sum += words_subwords[i]
                    output_text = text[:words_index[i]]
                counter += 1
                print(counter)
            else:
                output_text = text

            result = cat + "\t" + output_text + "\n"

            out_file.write(result)
            print(result)

def createKfoldFiles(lines, k):
    lines = np.random.permutation(lines)
    number_of_lines = len(lines)
    print(number_of_lines)
    step = int(number_of_lines / k)
    print(step)
    fold = 1
    for r in range(0, number_of_lines, step):
        with open('czech_text_document_corpus_v20/300tokens/test_' + str(fold) + '.txt', 'a') as test, open('czech_text_document_corpus_v20/300tokens/train_'+ str(fold) +'.txt', 'a') as train:
            print(r, r + step)
            for i, line in enumerate(lines):
                if i in range(r, r + step):
                    test.write(line)
                else:
                    train.write(line)

            fold += 1

# Dev is 1/10 of train
def createDev(lines):
    k = 10
    lines = np.random.permutation(lines)
    number_of_lines = len(lines)
    print(number_of_lines)
    step = int(number_of_lines / k)
    print(step)
    for r in range(0, number_of_lines, step):
        with open('czech_text_document_corpus_v20/300tokens/cz_corpus_5/cz_corpus_5_dev.txt', 'a') as dev, \
                open('czech_text_document_corpus_v20/300tokens/cz_corpus_5/cz_corpus_5_train.txt', 'a') as train:
            print(r, r + step)
            for i, line in enumerate(lines):
                if i in range(r, r + step):
                    dev.write(line)
                else:
                    train.write(line)
        break



if JOB == 'FOLDS':
    with open("czech_text_document_corpus_v20/300tokens/train.txt", "r") as f:
        lines = f.readlines()
    createKfoldFiles(lines, 5)

elif JOB == 'CROP':
    with open("czech_text_document_corpus_v20/dev.txt", "r") as f:
        lines = f.readlines()
    out_path = "czech_text_document_corpus_v20/300tokens/dev.txt"
    getfirstXtokens(lines, out_path)

elif JOB == 'DEV':
    with open("czech_text_document_corpus_v20/300tokens/cz_corpus_5/train_5.txt", "r") as f:
        lines = f.readlines()
    createDev(lines)

else:
    print('unknown')

