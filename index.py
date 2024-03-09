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

# The number of documents to generate indexes for
# Set this to -1 to index every document
NUM_DOCS_TO_INDEX = 50

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

documents = defaultdict(str)

stop_words = set(stopwords.words('english'))
unique_words = set()
unique_doc_ids = set()
titles_description = defaultdict(list)
anchor_words = defaultdict(str)

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

    # Join the lemmatized tokens into a string
    return (' '.join(lemmatized_tokens))


class InvertedIndex:
    def __init__(self):
        # Initialize the inverted index as a dictionary of lists, where each list contains (doc_id, tf) tuples
        self.index = defaultdict(list)
        self.pagerank_scores = {}
        self.df = defaultdict(int)
        self.document_outlinks = defaultdict(set)  # Store outlinks as a set to avoid duplicates
        self.positions = defaultdict(lambda: defaultdict(list)) # store positions
        self.document_tfs = {}

    def get_title(soup_content):
        title_element = soup_content.find('title')
        if title_element:
            return title_element.get_text(strip=True)
        else:
            return "No Title"

    def get_anchor_text(self, soup_content):
        # soup_content = BeautifulSoup(html_content, 'html.parser')
        for anchor_tag in soup_content.find_all('a'):
            anchor_text = anchor_tag.get_text(strip=True)
        return anchor_text

    def get_description(self, soup_content):
        metas = soup_content.find_all('meta')  # Get Meta Description
        for m in metas:
            if m.get('name') == 'description':
                description = m.get('content').replace("\n", " ")
                return (description[:160] + '...') if len(description) > 160 else description
        return "No Description"

    def extract_domain(self, url):
        # Extracts the domain (netloc) from a given URL.
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
            title = title_element.get_text()
            title = (title[:65] + '...') if len(title) > 65 else title
            #extract description
            description = self.get_description(soup)
            # print("description: ", description)
            titles_description[doc_id].append((title, description))
            # print(" titles_description[doc_id] : ", titles_description[doc_id])


            #get anchor text from this doc_id, A
            anchor_text = ""
            for anchor_tag in soup.find_all('a'):
                url = anchor_tag.get('href')
                parsed_url = urlparse(url)
                url = parsed_url.netloc + parsed_url.path
                anchor_text = anchor_text + " " + anchor_tag.get_text(strip=True)
                anchor_words[url] = anchor_text

            # add anchor text of this doc id before tokenizing
            #anchor_text = self.get_anchor_text(soup)
            current_url = urlparse(documents[doc_id])
            current_url = current_url.netloc + current_url.path
            if current_url in anchor_words:
                texts += lemma(anchor_words[current_url])


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

            #Track position for word position
            tokens_with_positions = [(word, pos) for pos, word in enumerate(tokens)]
            monogram_count_with_positions = defaultdict(lambda: {"count": 0, "positions": []})
            for term, position in tokens_with_positions:
                monogram_count_with_positions[term]["count"] += 1
                monogram_count_with_positions[term]["positions"].append(position)

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

            document_tf = {}

            # Calculate score for each bigram and update index and doc length
            bigram_count = Counter(bigrams)
            for bigram, count in bigram_count.items():
                tf = count / len(tokens)
                # Weight the score with the tf and the importance of the word
                score = tf + term_importance.get(bigram, 0)
                self.index[bigram].append((doc_id, score))

            # Calculate score for each term and update index and doc length, and position, now calculate tf with position
            for term, data in monogram_count_with_positions.items():
                count = data['count']  #count of the term in the document
                positions = data['positions'] #list of positions where the term presents
                tf = count / len(tokens)
                # Weight the score with the tf and the importance of the word
                score = tf + term_importance.get(term, 0)
                self.index[term].append((doc_id, score))
                self.positions[term][doc_id] = positions  # store positions
                document_tf[term] = tf      #save tf for cosine similarity

            self.document_tfs[doc_id] = document_tf
            with open('document_tfs.json', 'w') as f:
                json.dump(self.document_tfs, f)

            # Extract outlinks
            outlinks = [link['href'] for link in soup.find_all('a', href=True)] #Find all hyperlinks
            for outlink in outlinks:
                domain = self.extract_domain(outlink)
                if domain:  # only keep valid URLs;
                    self.document_outlinks[doc_id].add(outlink)

    def calculate_idf(self, total_docs):
        # Calculate IDF values for the index and update the index with TF-IDF values
        for term, postings in self.index.items():
            idf = math.log(total_docs / len(postings))
            for i, (doc_id, tf) in enumerate(postings):
                score = tf * idf
                self.index[term][i] = (doc_id, round(score, 3))

    def calculate_pagerank_scores(self, total_docs):
        # Create mapping from doc_id to their index in adjacency matrix
        doc_id_to_index = {doc_id: i for i, doc_id in enumerate(self.document_outlinks)}
        adjacency_matrix = np.zeros((total_docs, total_docs)) #reference: https://stackoverflow.com/questions/75293289/adjacency-matrix-using-numpy

        # Build the adjacency matrix based on outlinks
        for doc_id, outlinks_set in self.document_outlinks.items():  # iterate document and its set of outlink
            doc_index = doc_id_to_index[doc_id]    #get matrix index for current document
            for outlink in outlinks_set:           #iterate each outlink of current document
                if outlink in doc_id_to_index:     # Check if outlink points to a document we indexed
                    outlink_index = doc_id_to_index[outlink]  #get matrix index for outlink document
                    adjacency_matrix[doc_index][outlink_index] = 1   #mark 1 to indicate link in the adjacency matrix

        # Calculate PageRank
        damping_factor = 0.85  # when I search it, one of document says"Google uses a damping factor around 0.85"
        initial_scores = np.ones(total_docs) / total_docs  #initialize all documents' score to 1/n
        scores = initial_scores     #first start with initial scores
        epsilon = 1.0e-8    #epsilon value got from searching
        max_iterations = 15  #maximum number of iteration to perform, also got number 15 by searching

        #reference: https://stackoverflow.com/questions/40200070/what-does-axis-0-do-in-numpys-sum-function
        outgoing_link = np.sum(adjacency_matrix, axis=1)  #C(Pi) - from our slides
        outgoing_link[outgoing_link == 0] = 1 # modify zeros in outgoing_link with 1 to avoid division by zero

        # calculate new scores based on the previous iteration's score
        # need to get sum of the pagerank score of all page by geting transpose of the matrix, since we need incoming link here
        # reference: https://stackoverflow.com/questions/54968660/numpy-transposing-a-vector
        # reference: https://stackoverflow.com/questions/5954603/transposing-a-1d-numpy-array
        # pageRank equation from website and our slides week9-pagerank page7: https://www.mathworks.com/help/matlab/math/use-page-rank-algorithm-to-rank-websites.html
        for iteration in range(max_iterations):
            #sum for PR(Pi)/C(Pi) for each page - from our slides
            rank_contributions = adjacency_matrix.T.dot(scores / outgoing_link)
            new_scores = (1 - damping_factor) / total_docs + damping_factor * rank_contributions

            #convergence check Reference: https://arxiv.org/pdf/2108.02997.pdf
            #reference: https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
            #if the change in score is smaller than epsilon, we converged, so stop iteration
            if np.linalg.norm(new_scores - scores) < epsilon:
                break
            scores = new_scores     #update the scores for the next iteration
        # Update self.pagerank_scores with the calculated PageRank scores
        for doc_id, index in doc_id_to_index.items():
            self.pagerank_scores[doc_id] = scores[index]

    # implemented for word position
    def calculate_proximity_score(self, doc_id, query_terms):
        term_positions = [self.positions[term][doc_id] for term in query_terms if
                          term in self.positions and doc_id in self.positions[term]]
        if not term_positions:
            return 0  # No proximity score if not all terms are found

        # Example: Calculate proximity as inverse of average distance between consecutive terms
        min_distance = float('inf')         #initialize to inifinity since we will find the minimum distance
        for i in range(len(term_positions) - 1):
            for pos1 in term_positions[i]:
                for pos2 in term_positions[i + 1]:
                    distance = abs(pos1 - pos2)
                    if distance < min_distance:
                        min_distance = distance

        if min_distance == float('inf'):
            return 0
        return 1 / (min_distance + 1)  # Add 1 to avoid division by zero

    def calculate_idf_and_pagerank(self, total_docs, query_terms=None):
        # First, call calculate_idf to update the index with TF-IDF values
        self.calculate_idf(total_docs)

        # Calculate PageRank scores
        self.calculate_pagerank_scores(total_docs)

        # Combine updated TF-IDF and PageRank scores, and proximity scores
        for term, postings in self.index.items():
            for i, posting in enumerate(postings):
                doc_id, tf_idf_score = posting  # Unpack the posting to get the TF-IDF score

                # Retrieve the PageRank score for the document
                pagerank_score = self.pagerank_scores.get(doc_id, 0)

                #calculate proximity score if query terms are provided
                proximity_score = 0
                if query_terms and term in query_terms:
                    proximity_score = self.calculate_proximity_score(doc_id, query_terms)

                # Combine TF-IDF and PageRank scores, and proximity scores
                final_score = tf_idf_score + pagerank_score + proximity_score  # tf_idf_score is used here directly
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

        print("Storing docs_metadata.txt")
        docs_metadata = json.dumps(titles_description)
        with open("docs_metadata.txt", 'w', encoding='UTF-8') as f:
            f.write(docs_metadata)

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
        global documents
        documents = json.load(f)
        doc_ids = list(documents.keys())

        counter = 0

        def add_document(doc_id):
            nonlocal counter
            counter += 1
            if NUM_DOCS_TO_INDEX != -1:
                if counter > NUM_DOCS_TO_INDEX:
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

