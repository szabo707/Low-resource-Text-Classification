import json
import seaborn as sns
import csv

from pylab import *
from json_parser import ContractsParser
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# EVALUATE:
# all - evaluate all categories
# main - evaluate main categories and create confusion matrix
EVALUATE = 'all'


def getPredictData():
    predict_file = open('predict_file.txt', 'r')
    predict_lines = predict_file.readlines()
    predict_data = []
    for line in predict_lines:
        line = line.rstrip("\n")
        predict_data.append(int(line))
    return predict_data

def generateGoldData():
    test_contracts = open('contracts_test.jsonl', 'r')
    goldData = []
    for obj in test_contracts:
        data = json.loads(obj)
        contract = ContractsParser(eval(data))
        label = int(contract.getLabel())
        goldData.append(int(label))
    return goldData

def createMainData(data):
    main_data = []
    for i in data:
        main_data.append(round(i, -2))
    return main_data

def getMainCatHStest(main):
    gold, hs_predict = [], []
    with open('contracts_test.jsonl', 'r') as test_contracts:
        for obj in test_contracts:
            data = json.loads(obj)
            contract = ContractsParser(eval(data))
            id = int(contract.getID())
            with open('reported_contracts.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=';')
                headings = next(csv_reader)
                for row in reversed(list(csv_reader)):

                        if main:
                            if int(row[1]) == id:
                                #print(row[1], ' ', row[2], ' ', row[4])
                                gold.append(round(int(row[4]), -2))
                                if row[2] != "NULL":
                                    hs_predict.append(round(int(row[2]), -2))
                                else:
                                    hs_predict.append(int(1)) # replace NULL for 1 - not exist category
                                break
                        else:
                            if int(row[1]) == id:
                                # print(row[1], ' ', row[2], ' ', row[4])
                                gold.append(int(row[4]))
                                if row[2] != "NULL":
                                    hs_predict.append(int(row[2]))
                                else:
                                    hs_predict.append(int(1))  # replace NULL for 1 - not exist category
                                break

    print(gold)
    print(hs_predict)
    acc = accuracy_score(gold, hs_predict) * 100
    print(acc)


def get_confusion_matrix(gold_main, predict_main, labels):
    cf_matrix = confusion_matrix(gold_main, predict_main, labels=labels, normalize='true')

    plt.figure(figsize=(28, 22))
    sns.set(font_scale=2)
    sns.heatmap(cf_matrix, annot=True, fmt='.0%',xticklabels=labels, yticklabels=labels, cmap='Oranges')

    plt.xticks(rotation=45, ha='center', fontsize=25)
    plt.yticks(fontsize=25)

    plt.title("Confusion matrix of main categories", fontsize=35)
    plt.xlabel('Predicted label', fontsize=35)
    plt.ylabel('True label', fontsize=35)
    plt.show()



if EVALUATE == 'all':
    gold = generateGoldData()
    predict = getPredictData()
    acc = accuracy_score(gold, predict) * 100
    print(acc)


elif EVALUATE == 'main':
    labels = [0, 10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900,
              11000, 11100, 11200, 11300, 11400, 11500, 11600, 11700, 11800, 11900, 12000]

    gold = generateGoldData()
    predict = getPredictData()

    gold_main = createMainData(gold)
    predict_main = createMainData(predict)

    acc = accuracy_score(gold, predict) * 100
    acc_main = accuracy_score(gold_main, predict_main) * 100

    print(acc)
    print(acc_main)
    get_confusion_matrix(gold_main, predict_main, labels)

else:
    print('unknown')






