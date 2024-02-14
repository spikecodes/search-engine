from collections import defaultdict
import json
import time
import zlib

class SearchEngine:
    def __init__(self):
        self.index = defaultdict(list)

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
            print('Error: index.txt not found')
            exit(1)

    def search(self, query):
        with open('webpages/WEBPAGES_RAW/bookkeeping.json', 'r') as f:
            # Parse the JSON file which maps doc_id to URL of the webpage
            documents = json.load(f)

            # Rank documents based on the query
            scores = defaultdict(float)
            if query in self.index:
                for doc_id, tfidf in self.index[query].items():
                    doc_url = documents[doc_id] # Resolve doc_id to the path
                    scores[doc_url] += tfidf
            
            # Sort the documents by score
            results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return results

def run(query):
    # Initialize and populate the search engine
    search_engine = SearchEngine()
    search_engine.read_documents('index.txt')

    start_time=time.time()

    # Print the search results
    results = search_engine.search(query.lower())

    time_taken=time.time()-start_time

    num_results = len(results)
    top_20_results = results[:20]

    return (time_taken, num_results, top_20_results)