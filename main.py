import index
import search
import streamlit as st
from urllib.parse import urljoin
import requests
from os.path import exists
from threading import Thread
from streamlit.runtime.scriptrunner import add_script_run_ctx

def run_search():
  time_taken, num_results, top_20_results = search.run(query)
  st.info(f"{num_results} results in {time_taken: .3f} seconds")
  # for i, (doc_path, score_title) in enumerate(top_20_results):
  #   st.write(f'{i+1}. Score: {score:.2f} for {doc_path}')

  for doc_url, doc_scores in top_20_results:
      for i, (score, title, description) in enumerate(doc_scores):
        if not ("PUBLIC-IP" in doc_url or "PUBLIC_IP" in doc_url or "YOUR_IP" in doc_url):
            if "http" not in doc_url:
              abs_url = "https://" + doc_url
            else:
              abs_url = doc_url

            encoded_url = requests.utils.requote_uri(abs_url)

            # If link is over 100 chars, truncate it with an ellipsis
            truncated_url = abs_url
            if len(truncated_url) > 100:
               truncated_url = truncated_url[:100] + '...'

            link_text = f"[{title.strip()}](<{encoded_url}>)"
            st.subheader(link_text)
            st.caption(f'<p style="color: gray">{truncated_url}</p>', unsafe_allow_html=True)
            st.write(description)

# for doc_url, doc_scores in top_20_results:
  #     st.write(f'title: {doc_scores[-1]}. url: {doc_url}')

st.title("CS 121 — Project 3")

st.header('Index')

gen_index_btn = st.button("Generate Index")

# If user clicks generate index, generate the index
if gen_index_btn:
  with st.spinner(text="Generating index..."):
    index.generate()

# If index has already been generated, show search bar
if exists("index/index.txt.zz") and exists("index/docs_metadata.json"):
    st.header('Search')
    query = st.text_input("Query", placeholder="computer science")
    search_button = st.button("Search")
    # When they press enter or click the search bar, run the search
    if query or search_button:
        run_search()


