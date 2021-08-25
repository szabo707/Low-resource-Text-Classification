import ufal.morphodita

model = ufal.morphodita.Tagger.load("czech-morfflex-pdt-161115-pos_only.tagger")

class MorphoDiTaLemmatizer:
    def __init__(self, model):
        self._model = model
        #assert self._model is not None
        self._morpho = self._model.getMorpho()
        self._allowed_tags = {"N", "A", "V", "D"} # "C" for numeral, if required

    def lemmatize(self, text):
        forms, lemmas, tokens = ufal.morphodita.Forms(), ufal.morphodita.TaggedLemmas(), ufal.morphodita.TokenRanges()
        tokenizer = self._model.newTokenizer()
        tokenizer.setText(text)

        results = []
        while tokenizer.nextSentence(forms, tokens):
            self._model.tag(forms, lemmas)

            for i in range(len(lemmas)):
                lemma, token = lemmas[i], tokens[i]
                results.append((
                    self._morpho.rawLemma(lemma.lemma).lower() if lemma.tag[0] in self._allowed_tags else None,
                    token.start,
                    token.start + token.length,
                ))
        return results


    def create_ngrams(self, Nmax, stemmed_words, Nmin=1):
        ngrams = []
        ranges = []
        lemmas = [x[0] for x in stemmed_words]
        starts = [x[1] for x in stemmed_words]
        ends = [x[2] for x in stemmed_words]
        for n in range(Nmin, min(Nmax, len(stemmed_words)) + 1):
            for i in range(len(stemmed_words) - n + 1):
                ngram = lemmas[i:(i + n)]
                start = starts[i : (i+n)]
                end = ends[i: (i + n)]
                if None not in ngram:
                    out_ngram = "=".join(ngram)
                    out_ranges = str((start)[0]) + ":" + str((end)[-1])
                    ngrams.append(out_ngram)
                    ranges.append(out_ranges)

        return ngrams, ranges
