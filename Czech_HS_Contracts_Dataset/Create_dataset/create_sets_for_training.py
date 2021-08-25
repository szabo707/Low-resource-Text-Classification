import numpy as np
import random


# Dev is 1/10 of train
def createDev(lines):
    k = 10
    #random_lines = random.shuffle(lines)
    number_of_lines = len(lines)
    print(number_of_lines)
    step = int(number_of_lines / k)
    print(step)
    for r in range(0, number_of_lines, step):
        with open('FINAL_contracts/using contracts/dev.jsonl', 'a') as dev, open('FINAL_contracts/train.jsonl', 'a') as train:
            print(r, r + step)
            for i, line in enumerate(lines):
                if i in range(r, r + step):
                    dev.write(line)
                else:
                    train.write(line)
        break


data_file = open("FINAL_contracts/all_contracts.jsonl").readlines()
random.shuffle(data_file)
open("FINAL_contracts/all_contracts_random2.jsonl", "w").writelines(data_file)

random_data = open("FINAL_contracts/all_contracts_random2.jsonl", "r")
lines = random_data.readlines()
createDev(lines)

random_data.close()