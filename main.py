import index
import search
import streamlit as st

def run_search():
  time_taken, num_results, top_20_results = search.run(query)
  st.info(f"{num_results} results in {time_taken: .3f} seconds")
  for i, (doc_path, score) in enumerate(top_20_results):
    st.write(f'{i+1}. Score: {score:.2f} for {doc_path}')

st.title("CS 121 — Project 3")

st.header('Index')
st.button("Generate Index", on_click=index.generate)
if len(index.unique_words) > 0:
    st.header('Search')
    query = st.text_input("Query", placeholder="computer science")
    if st.button("Search"):
        run_search()


