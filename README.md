# Search Engine

This project is a search engine designed to index a corpus of web pages and provide efficient search capabilities. It utilizes various algorithms and techniques to rank search results based on relevance, including the PageRank algorithm and cosine similarity.

## Features

- 🌐 Efficient indexing of web pages
- 🔍 Advanced search capabilities with multi-word query handling
- 📄 Support for HTML content extraction, including titles and descriptions
- 📊 Proximity scoring to enhance search relevance
- 📦 Batch processing for indexing multiple documents
- 🔄 PageRank algorithm for ranking search results
- 🧠 Cosine similarity calculation for improved result accuracy
- 🔒 Easy setup with Streamlit for a user-friendly interface

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.9 or higher
- Required Python packages: Streamlit, BeautifulSoup, NLTK, SpaCy, NumPy

### Installing Dependencies

You can install the required packages using pip:

```bash
pip install streamlit beautifulsoup4 nltk spacy numpy
python -m spacy download en_core_web_sm
```

### Running the Application

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/search-engine.git
   cd search-engine
   ```

2. Run the application:

   ```bash
   streamlit run main.py
   ```

## Usage

### Web Interface

Once the server is running, you can access the search engine through your web browser at `http://localhost:8501`. You can generate an index and perform searches directly from the interface.

### Indexing Documents

To index documents, place your HTML files in the `webpages/WEBPAGES_RAW/` directory. The application will read and index these files when you click the "Generate Index" button.

## API Endpoints

The search engine provides a simple web interface for interaction. For detailed API documentation, refer to the source code in `search.py` and `index.py`.

## Configuration

The search engine can be configured by modifying the source code. Key parameters include:

- `NUM_DOCS_TO_INDEX`: Set the number of documents to index (default: 50).
- `documents`: A JSON file mapping document IDs to URLs.

## Development

To set up the development environment:

1. Install Python and pip: https://www.python.org/downloads/
2. Clone the repository and navigate to the project directory.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the application: `streamlit run main.py`

## Testing

To run the test suite, you can create unit tests in a separate test file and execute them using:

```bash
pytest
```

## Explanation of Key Concepts

### Word Position and Proximity Scoring

The relevance of search terms is influenced by their proximity within the text. When words are located close to each other, they are deemed more relevant to the search query. The proximity score is calculated as the inverse of the average distance between consecutive occurrences of the same term, ensuring that closely placed terms receive higher relevance.

### Compiling Query Results

The search engine utilizes an index to identify documents containing the search terms. Documents are ranked based on the frequency of term occurrences, with the PageRank algorithm further refining the ranking to prioritize more authoritative sources.

### Handling Multi-Word Queries

For multi-word queries, the search engine breaks down the input into individual words and searches for each in the index. Documents are ranked based on the number of matches for each word, with higher scores for documents containing all query terms.

### Anchor Words

When a document contains links to another, the anchor text (the clickable text in a hyperlink) is added to the target document's index. This enhances the relevance of the linked document, as it reflects how other pages describe it.

### PageRank Algorithm

The PageRank algorithm assesses the importance of web pages based on the number of incoming links from other pages. The more links a page receives, the higher its rank, indicating greater authority. The algorithm constructs an adjacency matrix to represent links between documents and iteratively calculates scores until convergence.

### Cosine Similarity

Cosine similarity measures the similarity between two vectors, typically representing term frequencies in documents. It is calculated as the dot product of the vectors divided by the product of their magnitudes. This metric helps determine how closely related a document is to a search query.

### Advanced Features

The project implements zlib for data compression, significantly reducing storage requirements. A user-friendly interface is created using Streamlit, allowing for easy interaction with the search engine. Additionally, multithreading is employed to expedite index generation, enhancing overall performance.

## To Run the Program

1. Install Streamlit and other dependencies:

   ```bash
   pip install streamlit spacy
   ```

2. Run the program:

   ```bash
   streamlit run main.py
   ```

3. In your browser:

   - Click "Generate Index" to create the index.
   - Enter your query in the search bar and click "Search".

4. Ensure you have SpaCy installed and the English language model downloaded:

   ```bash
   pip install spacy
   python -m spacy download en_core_web_sm
   ```

## Outputted Files

The program generates the following files in the `index/` directory:

1. `index.txt` — The uncompressed index of terms.
2. `index.txt.zz` — The zlib-compressed index of terms.
3. `docs_metadata.json` — Contains the title and description of each document, extracted during indexing.
4. `document_tfs.json` — Stores the term frequency of each document for cosine similarity calculations.

## Testing Cases

- For lemmatization: "run" and "ran" yield the same result.
- Case sensitivity: "IrviNe" and "irvine" are treated as different terms.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
