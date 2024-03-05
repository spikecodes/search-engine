# Search Engine

For Project 3 of CS 121, we were tasked with creating a search engine. The assignment was to index a corpus of web pages and then use that index to search for relevant documents.

#html tags as an indicator of importance
#Resource: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
#Resoure: https://stackoverflow.com/questions/39755346/beautiful-soup-extracting-tagged-and-untagged-html-text
#Get Desciption resource: https://stackoverflow.com/questions/38009787/how-to-extract-meta-description-from-urls-using-python

## How did we go above and beyond?

We used zlib to compress and decompress the object, providing a near-100x difference in storage space.


## To run the program
Install Streamlit: pip install streamlit, pip install spacy,

Run the program: streamlit run main.py
On the browser: 
- Click "Generate Index" for generating index
- Enter query in the search bar and click "Search"


Install spacy and download English language model:
    pip install spacy
    python -m spacy download en_core_web_sm

# Pending Tasks:
- EC - +2 Word position
- search more than 2 words???
- should save 2-gram in different variable???
- Testing???