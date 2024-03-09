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

# Testing cases:
- For lemmatization: run vs ran :  give same result
- Case sensitive: IrviNe vs irvine

# PageRank Algorithm explanation:
Reference: https://en.wikipedia.org/wiki/PageRank
**Updated comments in the code, so refer to the comment, and this may help you visualize.
Basically, we have to find popular website where more links we get from other websites, the more important the website is.
First, I extract domains (extract_domain function) to parse URL and their domain names
Extracting Domain Names, then from add_document function, I extract outlinks, and if the link is valid, I store it to document_outlinks 
Doucument_outlink will have something like:
{
     “doc_a” : {“doc_b”, “doc_c”},
     “doc_b” : {“doc_c”},
     “doc_c”: {“doc_a”}
     “doc_d”: {} #let say this one doesn’t have any outlink
}
Then inside the calculate_pagerank_scores function, first create a map from doc_id to their index in the adjacency matrix.

doc_id_to_index creates something like:
{
     “doc_a” : 0,
     “doc_b” : 1,
     “doc_c”:  2,
     “doc_d”:  3,

}
Then creates adjacency matrix by using np.zeros, which will creates square matrix initialize with all zeros. Something like:
[[0,0,0,0,]
 [0,0,0,0,]
 [0,0,0,0,]
 [0,0,0,0,]]

Then loop through document_outlinks to fill adjacency matrix based on outlinks, now it wil becomes:
[[0,1,1,0,] #doc_a has outlink to doc_b and doc_c   #number of outgoing link: 2
 [0,0,1,0,]  #doc_b has outlink to doc_c            #number of outgoing link: 1
 [1,0,0,0,]  #doc_c has outlink to doc_a            #number of outgoing link: 1
 [0,0,0,0,]] # doc_d had no outlink                 #number of outgoing link: 0

When we initialize all the document’s score, it has to have all same value, and everything need to add up to 1, so we have to initiazlie to 1/number of total_docs
Then we use pagerank equation to calculate the pagerank, and if the change in scores is smaller than epsilon, that means, we’ve converged, so we don’t have to iteratively calculate pagerank, so we stop.


# Cosine similarity
Reference:https://en.wikipedia.org/wiki/Cosine_similarity
equation: cosine similarity = (A dot product B) / (magnitude(norm) A * magnitude(norm) B)
"the attribute vectors A and B are usually the term frequency vectors of the documents."
so I used tf to create A and B which is supposed to be query vector and document vector, then I used equation to find cosine similarity.