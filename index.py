from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import math
import json
import zlib
import sys

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
unique_words = set()
unique_doc_ids = set()

TAG_WEIGHTS = {
                'title': 0.5,
                'h1': 0.35,
                'h2': 0.30,
                'h3': 0.25,
                'h4': 0.20,
                'h5': 0.15,
                'h6': 0.10,
                'b': 0.05,
                'strong': 0.05,
              }

class InvertedIndex:
    def __init__(self):
        # Initialize the inverted index as a dictionary of lists, where each list contains (doc_id, tf) tuples
        self.index = defaultdict(list)

    def add_document(self, doc_id):
        print("ADDING DOCUMENT: " + doc_id)
        # Read the document and tokenize the text
        with open('webpages/WEBPAGES_RAW/' + doc_id, 'r', encoding='utf-8') as f:
            html_content = f.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            texts = soup.get_text(" ", strip=True)
            tokens = [word.lower() for word in nltk.word_tokenize(texts) if word.isalnum() and word.lower() not in stop_words]

            if len(tokens) == 0:
                # If no tokens on page, exit
                return
            elif len(tokens) == 1:
                # If only one token, use that
                bigrams = [tokens[0]]
            else:
                # If more than one token, split into 2-grams
                bigrams = list(nltk.ngrams(tokens, 2))
                bigrams = [f"{bigram[0]} {bigram[1]}" for bigram in bigrams]

            # Store document id in unique_doc_ids for analytics
            unique_doc_ids.add(doc_id)

            # Raw count of term in the document
            term_count = Counter(bigrams + tokens)
            unique_words.update(tokens)

            # ADDED CODE FOR HTML TAGS AS AN INDICATOR OF IMPORTANCE
            # Resource: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
            # Resource: https://stackoverflow.com/questions/39755346/beautiful-soup-extracting-tagged-and-untagged-html-text
            term_importance = defaultdict(int)
            for tag, weight in TAG_WEIGHTS.items():
                elements = soup.find_all(tag)
                for element in elements:
                    tag_texts = element.get_text(" ", strip=True)
                    tag_tokens = [word.lower() for word in nltk.word_tokenize(tag_texts) if word.isalnum() and word.lower() not in stop_words]
                    for token in tag_tokens:
                        term_importance[token] += weight

            # Calculate score for each term and update index and doc lengths
            for term, count in term_count.items():
                tf = count / len(tokens)
                # Weight the score with the tf and the importance of the word
                score = tf + term_importance[token]
                self.index[term].append((doc_id, score))


    def calculate_idf(self, total_docs):
        # Calculate IDF values for the index and update the index with TF-IDF values
        for term, postings in self.index.items():
            idf = math.log(total_docs / len(postings))
            for i, (doc_id, tf) in enumerate(postings):
                score = tf * idf
                self.index[term][i] = (doc_id, round(score, 3))

    def store_index(self, filename):
        file_lines = []

        print("Creating storeable index object...")

        # Save the index to a file for later use by search.py
        for term, postings in self.index.items():
            postings_json_no_spaces = str(postings).replace(' ', '')
            line = f"{term}—{postings_json_no_spaces}\n"
            file_lines.append(line)

        print("Cleaning up file...")
        # Remove the default dictionary extra memory from the file
        file_text = ''.join(file_lines).replace("defaultdict(<class'int'>,", '').replace(')','')

        print("Storing index to file...")
        # Store the file
        with open(filename, 'w', encoding='UTF-8') as f:
            f.write(file_text)

        print("Saving compressed version...")
        # Store a compressed version of the file
        compressed = zlib.compress(str.encode(file_text))
        with open(filename + '.zz', 'wb') as f:
            f.write(compressed)

        print("Done!")

def generate():
    # Initialize and populate the inverted index (example)
    index = InvertedIndex()

    with open('webpages/WEBPAGES_RAW/bookkeeping.json', 'r') as f:
        counter = 0
        documents = json.load(f)
        for doc_id in documents.keys():
            # counter += 1
            # if counter > 100:
            #     break;
            index.add_document(doc_id)

        # Calculate IDF values for the index
        total_docs = len(documents)
        index.calculate_idf(total_docs)

        # Store the index to a file
        index.store_index('index.txt')

    print("Unique words: " + str(len(unique_words)))
    print("Unique Doc IDs: " + str(len(unique_doc_ids)))