import os, glob

sets = ["train"]

for typeOfDateset in sets:
    os.chdir("czech_text_document_corpus_v20/" + typeOfDateset + "_txt/")
    files = [f for f in glob.glob("*.txt")]
    out_path = "../../czech_text_document_corpus_v20/" + typeOfDateset + ".txt"
    for f in sorted(files):
        cat = f.split(".")[0].split("_")[1:]
        categories = " ".join(cat)
        with open(f, 'r') as file:
            text = file.read()
            text = ' '.join(text.split())
        line = categories + "\t" + text
        with open(out_path, "a") as out_file:
            out_file.write(line + "\n")
        print(line)