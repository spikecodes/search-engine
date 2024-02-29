from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import math
import json
import zlib
import concurrent.futures
import numpy as np
from urllib.parse import urlparse
import spacy

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
unique_words = set()
unique_doc_ids = set()
titles = defaultdict(str)


# TAG_WEIGHTS = {
#                 'title': 0.5,
#                 'h1': 0.35,
#                 'h2': 0.30,
#                 'h3': 0.25,
#                 'h4': 0.20,
#                 'h5': 0.15,
#                 'h6': 0.10,
#                 'b': 0.05,
#                 'strong': 0.05,
#               }

TAG_WEIGHTS = {
    'title': 10,
    'h1': 6,
    'h2': 5,
    'h3': 4,
    'h4': 3,
    'h5': 3,
    'h6': 3,
    'b': 2,
    'strong': 2
}

for tag, weight in TAG_WEIGHTS.items():
    TAG_WEIGHTS[tag] = math.log(weight)





def lemma(texts):
    # Load the spaCy English model
    nlp = spacy.load('en_core_web_sm')
    # Lemma
    doc = nlp(texts)

    # Extract lemmatized tokens
    lemmatized_tokens = [token.lemma_ for token in doc]

    # Join the lemmatized tokens into a sentence
    return (' '.join(lemmatized_tokens))


class InvertedIndex:
    def __init__(self):
        # Initialize the inverted index as a dictionary of lists, where each list contains (doc_id, tf) tuples
        self.index = defaultdict(list)
        self.pagerank_scores = {}
        self.df = defaultdict(int)
        self.document_outlinks = defaultdict(set)  # Store outlinks as a set to avoid duplicates

    def get_title(soup_content):
        title_element = soup_content.find('title')
        if title_element:
            return title_element.get_text(strip=True)
        else:
            return "No Title"

    def extract_anchor_words(soup_content):
        # soup_content = BeautifulSoup(html_content, 'html.parser')
        anchor_words = []
        for anchor_tag in soup_content.find_all('a'):
            anchor_text = anchor_tag.get_text(strip=True)
            if anchor_text:
                anchor_words.extend(anchor_text.split())
        return anchor_words

    def extract_domain(self, url):
        # Extracts the domain (netloc) from a given URL.
        # - url (str): The URL from which to extract the domain.
        # Returns:
        # - str: The extracted domain if successful, None otherwise.
        try:
            parsed_url = urlparse(url)
            return parsed_url.netloc
        except Exception as e:
            print(f"Error extracting domain from URL {url}: {e}")
            return None

    def add_document(self, doc_id):
        print("ADDING DOCUMENT: " + doc_id)
        # Read the document and tokenize the text
        with open('webpages/WEBPAGES_RAW/' + doc_id, 'r', encoding='utf-8') as f:
            html_content = f.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            pre_texts = soup.get_text(" ", strip=True)
            texts = lemma(pre_texts)
            #extract title
            title_element = soup.find('title')
            titles[doc_id] = title_element.get_text()

            tokens = [word.lower() for word in nltk.word_tokenize(texts) if
                      word.isalnum() and word.lower() not in stop_words]

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
                    tag_tokens = [word.lower() for word in nltk.word_tokenize(tag_texts) if
                                  word.isalnum() and word.lower() not in stop_words]
                    for token in tag_tokens:
                        # print("token: ", token)
                        # print("weight: ", weight)
                        # print("term_imp[token]: ", term_importance[token])
                        if term_importance[token] < weight:
                            term_importance[token] = weight

            # print("term importance: ", term_importance)

            # {wordintile1: weight,
            # worldintiel2: weight,
            # }

            # Calculate score for each term and update index and doc lengths
            for term, count in term_count.items():
                tf = count / len(tokens)
                # Weight the score with the tf and the importance of the word

                score = tf + term_importance.get(term, 0)
                self.index[term].append((doc_id, score))

            # Extract outlinks
            outlinks = [link['href'] for link in soup.find_all('a', href=True)]
            for outlink in outlinks:
                domain = self.extract_domain(outlink)
                if domain:  # Simple filter to keep only valid URLs; you might need a more sophisticated approach
                    self.document_outlinks[doc_id].add(outlink)

    def calculate_idf(self, total_docs):
        # Calculate IDF values for the index and update the index with TF-IDF values
        for term, postings in self.index.items():
            idf = math.log(total_docs / len(postings))
            for i, (doc_id, tf) in enumerate(postings):
                score = tf * idf
                self.index[term][i] = (doc_id, round(score, 3))

    def calculate_pagerank_scores(self, total_docs):
        # Initialization for PageRank
        doc_id_to_index = {doc_id: i for i, doc_id in enumerate(self.document_outlinks)}
        adjacency_matrix = np.zeros((total_docs, total_docs))

        # Build the adjacency matrix based on outlinks
        for doc_id, outlinks_set in self.document_outlinks.items():
            doc_index = doc_id_to_index[doc_id]
            for outlink in outlinks_set:
                if outlink in doc_id_to_index:  # Check if outlink is within the indexed documents
                    outlink_index = doc_id_to_index[outlink]
                    adjacency_matrix[doc_index][outlink_index] = 1

        # Calculate PageRank
        damping_factor = 0.85
        initial_scores = np.ones(total_docs) / total_docs
        scores = initial_scores
        epsilon = 1.0e-8
        max_iterations = 15

        for iteration in range(max_iterations):
            new_scores = np.ones(total_docs) * (
                    1 - damping_factor) / total_docs + damping_factor * adjacency_matrix.T.dot(scores)
            if np.linalg.norm(new_scores - scores) < epsilon:
                break
            scores = new_scores
        # Update self.pagerank_scores with the calculated PageRank scores

        for doc_id, index in doc_id_to_index.items():
            self.pagerank_scores[doc_id] = scores[index]

    def calculate_idf_and_pagerank(self, total_docs):
        # First, call calculate_idf to update the index with TF-IDF values
        self.calculate_idf(total_docs)

        # Calculate PageRank scores
        self.calculate_pagerank_scores(total_docs)

        # Combine updated TF-IDF and PageRank scores
        for term, postings in self.index.items():
            for i, posting in enumerate(postings):
                doc_id, tf_idf_score = posting  # Unpack the posting to get the TF-IDF score

                # Retrieve the PageRank score for the document
                pagerank_score = self.pagerank_scores.get(doc_id, 0)

                # Combine TF-IDF and PageRank scores
                final_score = tf_idf_score + pagerank_score  # tf_idf_score is used here directly
                # print(f"Doc ID: {doc_id}, TF_idf: {tf_idf_score}, Pagerank: {pagerank_score}")
                # Update the posting with the combined score
                self.index[term][i] = (doc_id, round(final_score, 3))

    def store_index(self, filename):
        file_lines = []

        print("Creating storeable index object...")

        # Save the index to a file for later use by search.py
        for term, postings in self.index.items():
            # Create postings_json which looks like {"doc_id": score, "doc_id": score, ...}
            postings_json = {doc_id: score for doc_id, score in postings}
            postings_json_no_spaces = str(postings_json).replace(' ', '')
            line = f"{term}—{postings_json_no_spaces}\n"
            file_lines.append(line)

        print("Cleaning up file...")
        # Remove the default dictionary extra memory from the file
        file_text = ''.join(file_lines).replace("defaultdict(<class'int'>,", '').replace("'", '"')

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
        documents = json.load(f)
        doc_ids = list(documents.keys())

        counter = 0

        def add_document(doc_id):
            nonlocal counter
            counter += 1
            if counter > 50:
                return
            index.add_document(doc_id)

        # Use ThreadPoolExecutor to run add_document on multiple documents in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(add_document, doc_ids)

        # Calculate IDF values for the index
        total_docs = len(documents)
        # index.calculate_idf(total_docs)
        index.calculate_idf_and_pagerank(total_docs)

        # Store the index to a file
        index.store_index('index.txt')

    print("Unique words: " + str(len(unique_words)))
    print("Unique Doc IDs: " + str(len(unique_doc_ids)))


if __name__ == "__main__":
    generate()
