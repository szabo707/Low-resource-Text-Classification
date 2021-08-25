import jsonlines
import os
import glob
import copy
import ufal.morphodita
import sys

from json_parser import ContractsParser
from windows import Windows


def checkIn(element, array):
    if element in array:
        array.remove(element)
        return array

def addNewElement(element, obj):
    obj.update(element)
    return obj

counter = 0

model = ufal.morphodita.Tagger.load("czech-morfflex-pdt-161115-pos_only.tagger")
keywordsFile = 'all_keywords_uniq'

filename = sys.argv[-1]

code_of_category = int(filename.split("/")[1].split("_")[0])
with jsonlines.open(filename, mode='r') as reader:
    for obj in reader:
        # print(obj)
        contract = ContractsParser(obj)
        plainText = contract.getPlainText()
        prepare_text = contract.textProcessing(plainText)

        new_obj = copy.deepcopy(obj)
        prilohy = new_obj.get('Prilohy')
        for element in prilohy:
            del element['PlainTextContent']

        window_ranges = Windows(prepare_text, model, keywordsFile).getWindows(300, 5)

        addNewElement({"Label": code_of_category}, new_obj)
        addNewElement({"PlainTextContent": prepare_text}, new_obj)
        addNewElement({"WindowsRange": window_ranges}, new_obj)

        with jsonlines.open('contracts_by_relevance/' + filename.split("/")[1], mode='a') as writer:
            writer.write(str(new_obj))
            print(str(new_obj), flush=True)
            counter += 1

        if counter == 1000:
            break


