# coding=utf-8
import numpy as np
import transformers
import robeczech_tokenizer
import ufal.morphodita

from lemmatizer import MorphoDiTaLemmatizer
from json_parser import ContractsParser

tokenizer = robeczech_tokenizer.RobeCzechTokenizer("tokenizer")
tokenizer_encode = lambda sentence: tokenizer.encode(sentence)["input_ids"]

model = ufal.morphodita.Tagger.load("czech-morfflex-pdt-161115-pos_only.tagger")
keywordsFile = 'all_keywords_uniq'

class Windows():

    def __init__(self, plainText, model, keywordsFile):
        self.data = plainText
        self.lemmatizer = MorphoDiTaLemmatizer(model)
        self.keywords = set()

        with open(keywordsFile, 'r', encoding="utf-8") as keywords_file:
            for line in keywords_file:
                line = " ".join(line.split())
                self.keywords.add(line)

    def checkKeyWords(self, listOfLemma):
        n_grams, ranges = self.lemmatizer.create_ngrams(3, listOfLemma)
        keyword_starts = []
        keyword_ends = []

        for gram, r, i in zip(n_grams, ranges, range(len(n_grams))):
            if gram in self.keywords:
                keyword_starts.append(int(ranges[i].strip('').split(":")[0]))
                keyword_ends.append(int(ranges[i].strip('').split(":")[1]))

        return  np.array(keyword_starts), np.array(keyword_ends)

    def wordsIndex(self, text):
        words = [0]
        for i in range(len(text)-1):
            if text[i].isspace():
                words.append(i)
        return np.array(words)

    def wordSubwords(self, text):
        words = self.wordsIndex(text)
        subwords_len = []
        for i in range(len(words)):
            word_text = text[words[i]:words[i+1] if i+1 < len(words) else len(text)]
            subwords_len.append(len(tokenizer.encode(word_text, add_special_tokens=False)["input_ids"]))
        return np.array(subwords_len)

    def getWindows(self, maxSize, N):
        text = self.data
        lemmas = self.lemmatizer.lemmatize(text)
        keyword_starts, keyword_ends = self.checkKeyWords(lemmas)
        words_index = self.wordsIndex(text)
        words_subwords = self.wordSubwords(text)

        all_tokens = sum(words_subwords)
        p = all_tokens / maxSize
        if p < N and maxSize < all_tokens:
            N = round(p)
        elif all_tokens < maxSize:
            maxSize = all_tokens
            N = 1

        intermediate_results = []
        results = []
        n = 0

        while (n < N):
            tokens_sum = 0
            sum_ = 0
            left_margin = 0
            window_end = -1

            for i in range(len(words_index)):
                while ( window_end + 1 < len(words_index) and (tokens_sum + words_subwords[window_end + 1]) <= maxSize ):
                    window_end += 1
                    mask = np.logical_and(words_index[i] <= keyword_starts, keyword_ends <= words_index[window_end])
                    sum_ = np.sum(mask)
                    tokens_sum += words_subwords[window_end]
                    if not keyword_starts[keyword_starts >= words_index[i]].size:
                        left_margin = words_index[i]
                    else:
                        left_margin = words_index[i] - np.min(keyword_starts[keyword_starts >= words_index[i]])

                if not keyword_ends[keyword_ends <= words_index[window_end]].size:
                    right_margin = words_index[window_end]
                else:
                    right_margin = words_index[window_end] - np.max(keyword_ends[keyword_ends <= words_index[window_end]])

                margin = np.abs(left_margin + right_margin)
                intermediate_results.append((sum_, words_index[i], words_index[window_end], i, window_end, margin, [left_margin, right_margin]))
                tokens_sum -= words_subwords[i]

            best_window = max(intermediate_results, key=lambda x: (x[0], -x[5]))
            results.append(best_window[1:3])
            intermediate_results = []

            recalculation_mask = np.logical_and(words_index[best_window[3]] <= keyword_starts, keyword_ends <= words_index[best_window[4]])
            keyword_starts = keyword_starts[np.logical_not(recalculation_mask)]
            keyword_ends = keyword_ends[np.logical_not(recalculation_mask)]
            n += 1

        return results


if __name__ == "__main__":
    #only for test
    contract_text = "Objednávka č. – 18188 Odběratel: Dodavatel: Děčínská sportovní, příspěvková organizace DeCe Computers s.r.o. Oblouková 1400 / 6 Žerotínova 378 Děčín Děčín I – Staré město 405 02 405 02 Děčín IČO: 751 07 350 IČO: 44567626 DIČ: CZ75107350 DIČ: Datum: 25.9.2018 E-mail: Způsob platby: faktura Mobil: Text, název a popis zboží Objednáváme u Vás pro Aquapark Děčín: MJ Cena za MJ Server pro odbavovací systém MN ks 328.000,- Kč 1 Rozhodnutí RM č. 18 16 35 01 Vyřizuje: Telefon: Mobil: Schválil: Jaroslav Klouček 412 704 212 Ing. Igor Bayer Bankovní spojení: Razítko a podpis KB Děčín BÚ 35-9603590207 / 0100 FKSP 35-9824320227 / 0100 DĚČÍNSKÁ SPORTOVNÍ, p.o. | Oblouková 1400/6; 405 02 Děčín | IČO: 75107350 | DIČ: CZ75107350 | Bank. spoj.: 35-9603590207/0100 T: +420 412 704 212 | F: +420 412 704 233 | E: info@dcsportovni.cz | w.dcsportovni.cz Potvrzení objednávky č. – 18188 Odběratel: Dodavatel: Děčínská sportovní, příspěvková organizace Oblouková 1400/6 Děčín 405 02 DeCe Computers s.r.o. Žerotínova 378 Děčín I – Staré město 405 02 Děčín Text, název a popis zboží MN MJ Cena za MJ Server pro odbavovací systém 1 ks 328.000,- Kč V souladu se zákonem č. 340/2015 Sb. o registru smluv, akceptujeme vaši objednávku a souhlasíme s jejím zveřejněním. Datum: 27.9.2018 Za dodavatele: Ivan Vítek DĚČÍNSKÁ SPORTOVNÍ, p.o. | Oblouková 1400/6; 405 02 Děčín | IČO: 75107350 | DIČ: CZ75107350 | Bank. spoj.: 35-9603590207/0100 T: +420 412 704 212 | F: +420 412 704 233 | E: info@dcsportovni.cz | w.dcsportovni.cz"

    print(contract_text)
    parser = ContractsParser(contract_text)
    prepare_text = parser.textProcessing(contract_text)
    print(prepare_text)
    windows_data = Windows(prepare_text, model, keywordsFile)
    list_of_windows = windows_data.getWindows(300, 5)
    print(list_of_windows)




