from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import math
import json

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

TAG_WEIGHTS = {
                'title': 10,
                'h1': 6,
                'h2': 5,
                'h3': 4,
                'h4': 3,
                'h5': 3,
                'h6': 3,
                'b': 2,
                'strong': 2,
                'p': 1,
                'span': 1,
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
            tokens = [word.lower() for word in nltk.word_tokenize(texts) if word.isalpha() and word.lower() not in stop_words]

            # Raw count of term in the document
            term_count = defaultdict(int)
            for token in tokens:
                term_count[token] += 1

            #ADDED CODE FOR HTML TAGS AS AN INDICATOR OF IMPORTANCE
            # Resource: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
            # Resoure: https://stackoverflow.com/questions/39755346/beautiful-soup-extracting-tagged-and-untagged-html-text
            term_importance = defaultdict(int)
            for tag, weight in TAG_WEIGHTS.items():
                elements = soup.find_all(tag)
                for element in elements:
                    tag_texts = element.get_text(" ", strip=True)
                    tag_tokens = [word.lower() for word in nltk.word_tokenize(tag_texts) if word.isalpha() and word.lower() not in stop_words]
                    for token in tag_tokens:
                        if token in term_importance:
                            term_importance[token] += weight
                        else:
                            term_importance[token] = weight

            # Calculate TF-IDF for each term and update index and doc lengths
            for term, count in term_count.items():
                tf = count / len(tokens)
                self.index[term].append((doc_id, tf))

    def calculate_idf(self, total_docs):
        # Calculate IDF values for the index and update the index with TF-IDF values
        for term, postings in self.index.items():
            idf = math.log(total_docs / len(postings))
            for i, (doc_id, tf) in enumerate(postings):
                self.index[term][i] = (doc_id, tf * idf)

    def store_index(self, filename):
        # Save the index to a file for later use by search.py
        with open(filename, 'w') as f:
            postings_json = defaultdict(int)
            for term, postings in self.index.items():
                for doc_id, score in postings:
                    postings_json[doc_id] = score
                f.write(f"{term} — {json.dumps(postings_json)}\n")

if __name__ == '__main__':
    # Initialize and populate the inverted index (example)
    index = InvertedIndex()

    with open('webpages/WEBPAGES_RAW/bookkeeping.json', 'r') as f:
        counter = 0
        documents = json.load(f)
        for doc_id in documents.keys():
            counter += 1
            if counter > 100:
                break;
            index.add_document(doc_id)

        # Calculate IDF values for the index
        total_docs = len(documents)
        index.calculate_idf(total_docs)

        # Store the index to a file
        index.store_index('index.txt')