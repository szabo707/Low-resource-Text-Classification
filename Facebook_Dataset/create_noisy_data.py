import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import random


def createRandomTrainFile():
    PERC = 80
    TYPE = "train"
    with open("czech_facebook/czech_facebook_" + TYPE +".txt") as myfile:
        lines = myfile.readlines()
        N = np.ceil((len(lines) / 100)*PERC)

        i = 0
        for line in lines:
            cat = line.split("\t")[0]
            #cate.append(cat)
            text = line.split("\t")[1]
            if i < N:
                poss = ["p", "n"]
                if cat == "0":
                    new_cat = np.random.choice(poss)
                elif cat == "n":
                    new_cat = np.random.choice(["p", "0"])
                else:
                    new_cat = np.random.choice(["n", "0"])

                with open('czech_facebook/czech_facebook_random_' + str(PERC) + '_' + TYPE +'.txt', 'a') as randFile:
                    result = new_cat + "\t" + text

                    randFile.write(result)
                    print(result)
                i += 1

            else:
                with open('czech_facebook/czech_facebook_random_' + str(PERC) + '_' + TYPE +'.txt', 'a') as randFile:
                    randFile.write(line)

    lines = open('czech_facebook/czech_facebook_random_' + str(PERC) + '_' + TYPE +'.txt').readlines()
    random.shuffle(lines)
    open('czech_facebook/czech_facebook_random_' + str(PERC) + '_' + TYPE + '.txt', 'w').writelines(lines)

    print(N)

createRandomTrainFile()
