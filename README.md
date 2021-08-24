# Low-resource-Text-Classification
The aim of the thesis is to evaluate Czech text classification tasks in the low-resource settings. We introduce three datasets, two of which were publicly available and one was created partly by us. This dataset is based on contracts provided by the web platform Hlídač Státu. It has most of the data annotated automatically and only a small part manually. Its distinctive feature is that it contains long contracts in the Czech language. We achieve outstanding results with the proposed model on publicly available datasets, which confirms the sufficient performance of our model. In addition, we performed experimental measurements of noisy data and of various amounts of data needed to train the model on these publicly available datasets. On the contracts dataset, we focused on selecting the right part of each contract and we studied with which part we can get the best result. We have found that for a dataset that contains some systematic errors due to automatic annotation, it is more advantageous to use a shorter but more relevant part of the contract for classification than to take a longer text from the contract and rely on BERT to learn correctly.

---

The following source codes are attached to the master thesis Low-resource Text Classification (2021), A.Szabó, MFF UK. The scripts are divided according to their use on a specific dataset.

## Facebook Dataset

* **sentiment\_analysis\_model.py** - Script for training the model.

* **text\_classification\_dataset.py** - Preparation of data and loading data for training.

* **robeczech\_tokenizer.py** - Tokenizer used for obtaining the subwords.

* **create\_amount\_of\_train\_data.py** - Creation of training data with different amounts of posts from the original training set.

* **create\_noisy\_data.py** - Creation of training data with different amounts of data noise.

* **evaluate\_predict\_file.py** - Calculation of the evaluation from the obtained predict file.

## Czech Text Document Corpus

* **cz\_corpus\_model.py** - Script for training the model.

* **cz\_corpus\_text\_classification.py** - Preparation of data and loading data for training.

* **robeczech\_tokenizer.py** - Tokenizer used for obtaining the subwords.

* **create\_amount\_of\_train\_data.py** - Creation of training data with different amounts of documents from the original training set.

* **merged\_txt\_files.py** - Merging individual text files of documents into one text file.

* **preprocessing\_dataset.py** - Preparation of the acquired text file for training. Reduction of the documents to a length of 300 tokens. Creating folds for training.

## Creating - Czech HS Contracts Dataset (CHSC)
We present the scripts in the order in which was creating the dataset.

* **categoriesCZ.json** - Json file with all labels and names of the categories.

* **notations\_of\_files.txt** - Name list of the categories used for naming the obtained contracts.

* **get\_contracts.py** - Obtaining contracts according to their relevance in the categories from the Hlídač Státu portal.

* **all\_keywords\_uniq.txt** - List of keywords used for obtaining windows from the text of the contract.

* **robeczech\_tokenizer.py** - Tokenizer used for obtaining the subwords.

* **lemmatizer.py** - Script for obtaining the lemmas from the text and for creating the n-grams from obtained lemmas.

* **json\_parser.py** - Parsing the .json contract record. It allows us to return content of used objects, such as subject, recipient and the text of the contract.

* **windows.py** - Obtaining individual windows from the text of the contract with the content of as many keywords as possible. Keywords are centralized in the center of the window.

* **create\_pre\_final\_dataset.py** - Preparation of the contracts into the final form by adding the label, plaintext of the contract and windows to each contract.

* **final\_contracts\_dataset.sh** - Script for AIC cluster used for automatic preparation of final contracts.

* **merged\_json\_files.sh** - Merging all acquired contracts into one .jsonl file.

* **create\_sets\_for\_training.py** - Creation of the training and development sets.

## Training - Czech HS Contracts Dataset (CHSC)

* **contracts\_classification\_model.py** - Script for training the model.

* **contracts\_classification\_dataset.py** - Preparation of data and loading data for training.

* **evaluate\_and\_confusion\_matrix.py** - Evaluation of the predicted file and obtaining the confusion matrix for main categories.


