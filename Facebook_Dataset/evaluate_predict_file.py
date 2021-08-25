import numpy as np

from numpy import array
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

predictionFile = open('sentiment_analysis_predict.txt', 'r')
predictionLines = predictionFile.readlines()

goldFile = open('gold_czech_facebook_test.txt', 'r')
goldLines = goldFile.readlines()

def listOfClass(data):
    y_data = []
    for line in data:
        y_data.append(line[0][0])
    return y_data

def convertToEncoded(data):
    label_encoder = LabelEncoder()
    values = array(listOfClass(data))
    print(values)
    intEncoded = label_encoder.fit_transform(values)
    return np.array(intEncoded)

def getOnlyPNvalue(y_true, y_pred):
    trueOutList, predOutList = [], []
    for t, p in zip(y_true, y_pred):
        if t == "0":
            continue
        else:
            trueOutList.append(t)
            predOutList.append(p)
    return (trueOutList, predOutList)


y_true = listOfClass(goldLines)
y_pred = listOfClass(predictionLines)
#print(goldLines)

acc = accuracy_score(convertToEncoded(goldLines), convertToEncoded(predictionLines)) * 100

macroF1 = f1_score(y_true, y_pred , average="macro") * 100
perF1 = f1_score(y_true, y_pred , average=None)


print()
print("Accuracy:", acc)
print()
print("F1 score (macro):", "{:.2f}".format(macroF1))
print("Scores for each class:", perF1)
