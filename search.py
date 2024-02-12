from collections import defaultdict
import json
import time

class SearchEngine:
    def __init__(self):
        self.index = defaultdict(list)

    def read_documents(self):
        documents = []
        # Pulls the documents from index.txt and stores them in a list. If index.txt does not exist, throws an error.
        try:
            with open('index.txt', 'r') as f:
                for line in f:
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

if __name__ == '__main__':
    # Initialize and populate the search engine
    search_engine = SearchEngine()
    search_engine.read_documents()

    # Perform a search using user input
    query = input('Enter a search query: ')

    start_time=time.time()

    # Print the search results
    results = search_engine.search(query)

    time_taken=time.time()-start_time

    num_results = len(results)
    top_20_results = results[:20]

    print(f"{num_results} results in {time_taken:.3f} seconds")
    for i, (doc_path, score) in enumerate(top_20_results):
        print(f'{i+1}. Score: {score:.2f} for {doc_path}')