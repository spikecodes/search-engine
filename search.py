from collections import defaultdict

class SearchEngine:
    def __init__(self, graph):
        self.index = defaultdict(list)

    def read_documents(self):
        documents = []
        # Pulls the documents from index.txt and stores them in a list. If index.txt does not exist, throws an error.
        try:
            with open('index.txt', 'r') as f:
                for line in f:
                    term, postings = line.split(':')
                
                    # Remove leading and trailing whitespace
                    term = term.strip()
                    postings = postings.strip()

                    # Remove the brackets and split the postings list
                    postings = postings[1:-1].split(', ')
                    postings = [tuple(map(int, posting.split(', '))) for posting in postings]

                    # Add the term and postings to the index
                    self.index[term] = postings
        except FileNotFoundError:
            print('Error: index.txt not found')
            exit(1)

    def search(self, query):
        # Rank documents based on the query
        scores = defaultdict(float)
        for term in query.split():
            if term in self.index:
                for doc_id, tfidf in self.index[term]:
                    scores[doc_id] += tfidf

        # Sort the documents by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs

if __name__ == '__main__':
    # Initialize and populate the search engine
    search_engine = SearchEngine()
    search_engine.read_documents()

    # Perform a search using user input
    query = input('Enter a search query: ')

    # Print the search results
    results = search_engine.search(query)
    for doc_id, score in results:
        print(f'Document {doc_id} - Score: {score:.2f}')