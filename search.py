from collections import defaultdict
import json
import time
import zlib
from helper import lemma
from urllib.parse import urljoin, urlparse
from itertools import combinations
import numpy as np
from numpy.linalg import norm
from collections import Counter

class SearchEngine:
    def __init__(self):
        self.index = defaultdict(list)
        self.pagerank_scores = {}
        self.document_tfs = {}

        # Load the index once and persist it
        self.load_index('index/index.txt')
        self.load_index_tfs('index/document_tfs.json') # Load TF for cosine similarity
        print("Index loaded")

    def load_index(self, filename):
        # Pulls the documents from index.txt and stores them in a list. If index.txt does not exist, throws an error.
        try:
            with open(filename + '.zz', 'rb') as f:
                # Decompress the compressed index
                lines = zlib.decompress(f.read()).decode().split('\n')[:-1]

                # Iterate through lines
                for line in lines:
                    term, postings = line.split('—')

                    # Remove leading and trailing whitespace
                    term = term.strip()
                    postings = postings.strip()

                    # Parse the postings list into an array of (doc_id, score)
                    parsed = json.loads(postings)

                    # Add the term and postings to the index
                    self.index[term] = parsed
        except FileNotFoundError:
            print('Error: index/index.txt.zz not found')
            exit(1)

    def load_index_tfs(self, filename):
        with open(filename, 'r') as f:
            self.document_tfs = json.load(f)

    # Function to search and return results with count of matched words
    def search_and_count(self, query_words):
        results_dict = {}
        # Search for individual words and add 1 to count for each match
        for word in query_words:
            for url, data_list in self.search(word):
                if url in results_dict:
                    results_dict[url]['count'] += 1
                    results_dict[url]['data'].update(data_list)
                else:
                    results_dict[url] = {'count': 1, 'data': set(data_list)}
        
        # Search for pairs of words and add 2 to count for each match
        if len(query_words) == 2:
            query_pair = f"{query_words[0]} {query_words[1]}"
            for url, data_list in self.search(query_pair):
                if url in results_dict:
                    results_dict[url]['count'] += 2
                    results_dict[url]['data'].update(data_list)
                else:
                    results_dict[url] = {'count': 2, 'data': set(data_list)}

        # Break longer queries into pairs if more than 2 words
        # Search for pairs of words and add 2 to count for each match
        if len(query_words) > 2:
            for word1, word2 in combinations(query_words, 2):
                query_pair = f"{word1} {word2}"
                for url, data_list in self.search(query_pair):
                    if url in results_dict:
                        results_dict[url]['count'] += 2  # Increment by 2 as it matches 2 words
                        results_dict[url]['data'].update(data_list)
                    else:
                        results_dict[url] = {'count': 2, 'data': set(data_list)}

        # Deduplicate results by URL, preserve one with highest score
        for url in results_dict:
            unique_data = {}
            for score, title, description in results_dict[url]['data']:
                if title not in unique_data or unique_data[title][0] < score:
                    unique_data[title] = (score, description)
            results_dict[url]['data'] = set((score, title, desc) for title, (score, desc) in unique_data.items())

        return results_dict

    def search(self, query):
        with open('webpages/WEBPAGES_RAW/bookkeeping.json', 'r') as f:
            # Parse the JSON file which maps doc_id to URL of the webpage
            documents = json.load(f)

            #normalize query to calculate cosine similarity
            query_terms = lemma(query).lower().split()
            query_tf = Counter(query_terms)
            total_terms = len(query_terms)
            query_tf_normalized = {}
            for term, count in query_tf.items():
                normalized_frequency = count / total_terms
                query_tf_normalized[term] = normalized_frequency


            # Rank documents based on the query
            scores = defaultdict(list)
            if query in self.index:
                with open("index/docs_metadata.json", 'r') as docs_metadata:
                   titles_description = json.load(docs_metadata)

                for doc_id, tfidf_pageRank in self.index[query].items():
                    doc_url = documents[doc_id]  # Resolve doc_id to the path

                    # add title and , description to scores dictionary
                    title = titles_description[doc_id][-1][0]
                    description = titles_description[doc_id][-1][1]

                    if doc_id in self.document_tfs:
                        doc_tf = self.document_tfs[doc_id]
                        # Assemble vectors for terms present in either the document or the query
                        terms = set()
                        for term in query_tf_normalized.keys():
                            terms.add(term)
                        for term in doc_tf.keys():
                            terms.add(term)

                        query_vector_values = []
                        for term in terms:
                            frequency = query_tf_normalized.get(term, 0)
                            query_vector_values.append(frequency)
                        query_vector = np.array(query_vector_values)

                        doc_vector_values = []
                        for term in terms:
                            frequency + doc_tf.get(term, 0)
                            doc_vector_values.append(frequency)
                        doc_vector = np.array(doc_vector_values)

                        # Compute cosine similarity
                        dot_product = np.dot(query_vector, doc_vector)
                        norm_query = norm(query_vector)
                        norm_doc = norm(doc_vector)
                        if norm_query == 0 or norm_doc == 0:
                            cosine_similarity = 0   #avoid division by zero
                        else:
                            cosine_similarity = dot_product / (norm_query * norm_doc)

                        scores[doc_url].append((tfidf_pageRank + cosine_similarity, title, description))

            # Sort the documents by score
            results = sorted(scores.items(), key=lambda x: x[1][0][0], reverse=True)
            return results

search_engine = SearchEngine()

def run(query):
    start_time = time.time()
    # normalized query
    query = ' '.join(word.lower() for word in query.split() if word.isalnum())
    query = lemma(query)

    # print("query: ", query)

    # Search the query
    query_words = query.split()
    results_dict = search_engine.search_and_count(query_words)

    # Sort results based on count (descending) to prioritize URLs with more query word matches
    sorted_results = sorted(results_dict.items(), key=lambda x: x[1]['count'], reverse=True)

    # Format sorted results to be [(url, [(score, title, description)]), ...]
    results = [(url, list(data['data'])) for url, data in sorted_results]    

    # Store the time taken to search
    time_taken = time.time() - start_time

    num_results = len(results)
    top_20_results = results[:20]
    # print(top_20_results)

    return (time_taken, num_results, top_20_results)


if __name__ == '__main__':
    print(run(input("Enter a search query: ")))
