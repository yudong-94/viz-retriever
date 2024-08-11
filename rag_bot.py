import numpy as np
import pandas as pd
import streamlit as st
import openai
from rag_bot_func import semantic_search, generate_answer


## create sidebar with chatbot introduction
with st.sidebar:
    st.sidebar.title("About the chatbot")
    st.sidebar.write("This is a RAG-based chatbot that retrieves the most relevant visualizations from [my weekly collections](https://yudong-94.github.io/personal-website/categories/#data-viz). **Please fill in your OpenAI API key to use it**.")
    ## get OpenAI API Key input
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    st.sidebar.write("Author: [Yu Dong](https://github.com/yudong-94)")
    st.sidebar.write("**Example questions for the chatbot:**")
    st.sidebar.write("""
    1. I want to see visualizations on startups.
    2. Can you show me visualizations about the gaming industry?
    3. Any visualizations on economics?
    """)

## set up chatbot title
st.title("📈 Viz Retriever")
st.caption("Find visualizations from [my weekly collections](https://yudong-94.github.io/personal-website/categories/#data-viz)")

## set up chatbot message flow
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Please tell me what visualization you are looking for :)"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

## get user query
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    ## retrieve search results based on the user query
    search_results = semantic_search(openai_api_key, prompt, 10)
    answer, viz_lists, tableau_code_lists = generate_answer(openai_api_key, prompt, search_results)
    ## send the chatbot response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
    ## display visualizations
    for i in range(len(viz_lists)):
        st.chat_message("assistant").write(viz_lists[i])
        st.components.v1.html(tableau_code_lists[i], height=800, width = 800)
