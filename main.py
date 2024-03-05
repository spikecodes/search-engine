import index
import search
import streamlit as st
from urllib.parse import urljoin
import requests
from os.path import exists

def run_search():
  time_taken, num_results, top_20_results = search.run(query)
  st.info(f"{num_results} results in {time_taken: .3f} seconds")
  # for i, (doc_path, score_title) in enumerate(top_20_results):
  #   st.write(f'{i+1}. Score: {score:.2f} for {doc_path}')

  for doc_url, doc_scores in top_20_results:
      for i, (score, title, description) in enumerate(doc_scores):
        #st.write(f'title: {title}')
        #st.write("URL: <a href=\"{}\">{}</a>".format(doc_url, title))
        if not ("PUBLIC-IP" in doc_url or "PUBLIC_IP" in doc_url or "YOUR_IP" in doc_url):
            abs_url = urljoin('https://', doc_url)
            # link_text = f"[{st.subheader(title)}]({abs_url})"
            # st.markdown(link_text, unsafe_allow_html=True)
            # st.write(description)

            encoded_url = requests.utils.requote_uri(abs_url)

            link_text = f"[{title}]({encoded_url})"
            st.subheader(link_text)
            st.write(description)


# for doc_url, doc_scores in top_20_results:
  #     st.write(f'title: {doc_scores[-1]}. url: {doc_url}')

st.title("CS 121 — Project 3")

st.header('Index')
st.button("Generate Index", on_click=index.generate)

if exists("index.txt.zz") and exists("docs_metadata.txt"):
    st.header('Search')
    query = st.text_input("Query", placeholder="computer science")
    search_button = st.button("Search")
    if query or search_button:
        run_search()


