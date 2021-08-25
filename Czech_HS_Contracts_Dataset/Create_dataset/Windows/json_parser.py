import json
import html
import re

def checkIn(element, array):
    if element in array:
        return array[element]

class ContractsParser():
    def __init__(self, json):
        self.data = json

    def getLabel(self):
        label = self.data.get('Label')
        return label

    def getPredmet(self):
        predmet = self.data.get('predmet')
        return predmet

    def getPlainTextContent(self):
        content = self.data.get('PlainTextContent')
        return content

    def getWindowsRanges(self):
        ranges = self.data.get('WindowsRange')
        return ranges

    def getPrijemce(self, returnData):
        prijemce = self.data.get('Prijemce')
        text_prijemce, adresa, ico, nazev = '', '', '', ''
        for element in prijemce:
            adresa = str(checkIn('adresa', element))
            text_prijemce += adresa
            ico = str(checkIn('ico', element))
            text_prijemce += ico
            nazev = str(checkIn('nazev', element))
            text_prijemce += nazev

        if returnData == 'ALL':
            return text_prijemce
        elif returnData == 'ADRESA':
            return adresa
        elif returnData == 'ICO':
            return ico
        elif returnData == 'NAZEV':
            return nazev

    def getPlatce(self, returnData):
        platce = self.data.get('Platce')
        adresa = str(platce.get('adresa'))
        ico = str(platce.get('ico'))
        nazev = str(platce.get('nazev'))
        text_platca = adresa + ico + nazev

        if returnData == 'ALL':
            return text_platca
        elif returnData == 'ADRESA':
            return adresa
        elif returnData == 'ICO':
            return ico
        elif returnData == 'NAZEV':
            return nazev

    def getID(self):
        id = self.data.get('identifikator').get('idVerze')
        return id

    def getPlainText(self):
        prilohy = self.data.get('Prilohy')
        plainText = ''
        contractText = ''
        for element in prilohy:
            plainText += str(checkIn('PlainTextContent', element)) + ' '
            contractText = " ".join(plainText.split())
        return contractText

    def textProcessing(self, text):
        html_decode = html.unescape(text)
        text_afterWordsPro = self.wordProcessing(html_decode)
        rep = {"X ": "", " . ": " "}
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        removeXdots = pattern.sub(lambda m: rep[re.escape(m.group(0))], text_afterWordsPro)
        output_text = " ".join(removeXdots.split())
        return output_text

    def wordProcessing(self, text):
        word_list = text.split()
        new_word_list = []
        to_remove = "X. "
        pattern = "(?P<char>[" + re.escape(to_remove) + "])(?P=char)+"
        for word in word_list:
            word = re.sub(r'\b([^0-9]+)\1{2,}\b', r'\1', word)
            word = re.sub(pattern, r"\1", word)
            new_word_list.append(word)

        output_text = ' '.join(new_word_list)
        return output_text
