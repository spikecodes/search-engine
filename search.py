from collections import defaultdict
import json
import time
import zlib
from helper import lemma
from urllib.parse import urljoin, urlparse
from itertools import combinations

class SearchEngine:
    def __init__(self):
        self.index = defaultdict(list)
        self.pagerank_scores = {}

    def read_documents(self, filename):
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
            print('Error: index.txt.zz not found')
            exit(1)

    # Function to search and return results with count of matched words
    def search_and_count(self, query_words):
        results_dict = {}
        # Search for individual words
        for word in query_words:
            for url, data_list in self.search(word):
                if url in results_dict:
                    results_dict[url]['count'] += 1
                    results_dict[url]['data'].update(data_list)
                else:
                    results_dict[url] = {'count': 1, 'data': set(data_list)}
        
        # Search for pairs of words
        if len(query_words) == 2:
            query_pair = f"{query_words[0]} {query_words[1]}"
            for url, data_list in self.search(query_pair):
                if url in results_dict:
                    results_dict[url]['count'] += 2
                    results_dict[url]['data'].update(data_list)
                else:
                    results_dict[url] = {'count': 2, 'data': set(data_list)}

        # Break longer queries into pairs if more than 2 words
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

            # Rank documents based on the query
            scores = defaultdict(list)
            if query in self.index:
                with open("docs_metadata.txt", 'r') as docs_metadata:
                   titles_description = json.load(docs_metadata)

                for doc_id, tfidf_pageRank in self.index[query].items():
                    doc_url = documents[doc_id]  # Resolve doc_id to the path
                    # scores[doc_url] += tfidf_pageRank

                    # add title and , description to scores dictionary
                    title = titles_description[doc_id][-1][0]
                    description = titles_description[doc_id][-1][1]

                    #scores[doc_url].append((tfidf_pageRank, index.titles[doc_id]))
                    scores[doc_url].append((tfidf_pageRank, title, description))

            # Sort the documents by score
            results = sorted(scores.items(), key=lambda x: x[1][0][0], reverse=True)
            return results

def run(query):
    # Initialize and populate the search engine
    search_engine = SearchEngine()
    search_engine.read_documents('index.txt')

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
