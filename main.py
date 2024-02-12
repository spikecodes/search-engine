import index
import search
import streamlit as st

st.title("CS 121 — Project 3")

st.header('Index')
st.button("Generate Index", on_click=index.generate)

st.header('Search')
query = st.text_input("Query", placeholder="computer science")

def run_search():
  time_taken, num_results, top_20_results = search.run(query)

  print(f"{num_results} results in {time_taken:.3f} seconds")
  for i, (doc_path, score) in enumerate(top_20_results):
    print(f'{i+1}. Score: {score:.2f} for {doc_path}')

st.button("Search", on_click=run_search)

st.text('')