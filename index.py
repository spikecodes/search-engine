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

class InvertedIndex:
    def __init__(self):
        # Initialize the inverted index as a dictionary of lists, where each list contains (doc_id, tf) tuples
        self.index = defaultdict(list)
        self.tag_weights = {'h1': 3, 'h2': 3, 'h3': 3, 'p': 1} #Just added most frequent tags for now, we can add more in the future

    def add_document(self, doc_id, doc_path):
        # Read the document and tokenize the text
        with open(doc_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            texts = soup.get_text(" ", strip=True)
            tokens = [word.lower() for word in nltk.word_tokenize(texts) if word.isalpha() and word.lower() not in stop_words]

            # TODO: Consider the html tags as an indicator of importance, to be stored as metadata 
            # (e.g., <h1> tags could be weighted more heavily than <p> tags)


            # Count term frequencies in the document
            term_freqs = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1

            #ADDED CODE FOR HTML TAGS AS AN INDICATOR OF IMPORTANCE
            # Resource: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
            # Resoure: https://stackoverflow.com/questions/39755346/beautiful-soup-extracting-tagged-and-untagged-html-text
            for tag, weight in self.tag_weights.items():
                elements = soup.find_all(tag)
                for element in elements:
                    tag_texts = element.get_text(" ", strip=True)
                    tag_tokens = [word.lower() for word in nltk.word_tokenize(tag_texts) if word.isalpha() and word.lower() not in stop_words]
                    for token in tag_tokens:
                        term_freqs[token] += weight

            # Calculate TF-IDF for each term and update index and doc lengths
            for term, freq in term_freqs.items():
                tf = 1 + math.log(freq)
                self.index[term].append((doc_id, tf))

    def calculate_idf(self, total_docs):
        # Calculate IDF values for the index and update the index with TF-IDF values
        for term, postings in self.index.items():
            idf = math.log(total_docs / len(postings))
            for i, (doc_id, tf) in enumerate(postings):
                self.index[term][i] = (doc_id, tf * idf)

    def store_index(self, filename):
        # Save the index to a file for later use by search.py
        with open(filename) as f:
            for term, postings in self.index.items():
                f.write(f"{term}: {postings}\n")

def load_documents():
    # Read the bookkeeping file and load the documents
    documents = [] # List of (doc_id, doc_path) tuples
    with open('webpages/WEBPAGES_RAW/bookkeeping.json', 'r') as f:
        # Parse as json
        bookkeeping = json.load(f)
        for doc_id, doc_path in bookkeeping.items():
            documents.append((doc_id, 'webpages/WEBPAGES_RAW/' + doc_path))

if __name__ == '__main__':
    # Initialize and populate the inverted index (example)
    index = InvertedIndex()
    documents = load_documents()
    for doc_id, doc_path in documents:
        index.add_document(doc_id, doc_path)

    # Calculate IDF values for the index
    total_docs = len(documents)
    index.calculate_idf(total_docs)

    # Store the index to a file
    index.store_index('index.txt')