# Search Engine

For Project 3 of CS 121, we were tasked with creating a search engine. The assignment was to index a corpus of web pages and then use that index to search for relevant documents.

#html tags as an indicator of importance
#Resource: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
#Resoure: https://stackoverflow.com/questions/39755346/beautiful-soup-extracting-tagged-and-untagged-html-text
#Get Desciption resource: https://stackoverflow.com/questions/38009787/how-to-extract-meta-description-from-urls-using-python

## How do we compile results for queries?

We use the index to find the documents that contain the search terms. We then rank the documents based on the number of times the search terms appear in the document. We also use the PageRank algorithm to rank the search results.

## How do we handle multi-word (>2) queries?

We handle multi-word queries by breaking them down into individual words and then searching for each word in the index. We then rank the documents based on the number of "matches" for each word.

For example, if a user searches "Irvine Computer Science":

1. We break the query into individual words: ["Irvine", "Computer", "Science"]
2. We search for each word in the index and rank the documents based on the number of "matches" for each word.
   a. Pages with Irvine, Computer, and Science will be ranked highest.
   b. Then, pages with Irvine and Computer (or any other combination of two words) will be ranked next.
   c. Finally, pages with only one of the words will be ranked last.
3. We also factor in the PageRank algorithm to rank the search results.

## How did we go above and beyond?

We used zlib to compress and decompress the object, providing a near-100x difference in storage space.

We also used the Streamlit library to create a simple web interface for the search engine.

We further implemneted the PageRank algorithm to rank the search results.

## To run the program

Install Streamlit: pip install streamlit, pip install spacy,

Run the program: streamlit run main.py
On the browser:

- Click "Generate Index" for generating index
- Enter query in the search bar and click "Search"

Install spacy and download English language model:
pip install spacy
python -m spacy download en_core_web_sm

## Outputted Files

1. `index.txt` — This is the uncompressed index of terms.
2. `index.txt.zz` — This is the zlib-compressed index of terms.
3. `docs_metadata.txt` — This stores the title and description of each document, extracted during the indexing stage.

# Pending Tasks:

- EC - +2 Word position √
- search more than 2 words √
- should save 2-gram in different variable √
- Testing √
