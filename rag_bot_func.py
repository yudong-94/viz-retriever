import numpy as np
import pandas as pd
import requests
import json
import ast
import faiss
import openai


def get_embeddings(api_key, text):
    '''
    This function gets the embeedings of a query from OpenAI API text-embedding-3-small model
    '''
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "text-embedding-3-small",
        "input": text,
        "encoding_format": "float"
    }
    openai_url = "https://api.openai.com/v1/embeddings"
    response = requests.post(openai_url, headers=headers, json=data)

    return response.json()["data"][0]["embedding"]


my_vizzes = pd.read_csv("viz_embeddings.csv")
my_vizzes.head()

def string_to_array(s):
    try:
        lst = ast.literal_eval(s)
        return np.array(lst)
    except:
        return np.nan

my_vizzes["embeddings"] = my_vizzes["embeddings"].apply(string_to_array)

# Building the FAISS Index
# You can choose different types of indices based on your need for speed vs accuracy
# Here, we'll use an IndexFlatL2 index, which is a simple flat (brute-force) index
embedding_matrix = np.vstack(my_vizzes["embeddings"].values).astype('float32')
dimension = embedding_matrix.shape[1] # dimension of the embeddings
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix) # Adding the emebddings to the index

def semantic_search(api_key, query, k):
    '''
    Get the top k similar text to the query
    '''

    # Get the embedding of the query from OpenAI API
    query_embedding_response = get_embeddings(api_key, query)
    query_embedding = np.array(query_embedding_response).astype('float32').reshape(1,-1)

    # Retrieve Results
    distances, indices = index.search(query_embedding, k)
    results = my_vizzes.iloc[indices[0]].copy()
    results.loc[:, "distance"] = distances[0]

    #print(results)

    return results

def generate_answer(api_key, query, search_results):
    '''
    Generate chatbot response from the retrieved results
    '''

    search_results = search_results[search_results['distance']<=1.5]
    viz_info = search_results['description'].str.cat(sep='.\n')
    print(viz_info)

    system_prompt = '''
    ### Context
    I have a list of visualizations I have made.
    You are a chatbot to help users to find interesting visualizations.

    ### Objective
    You will be provided with the user's question and a list of visualizations that are relevant.
    Your goal is to summarise the information of relevant visualizations to answer the question.

    ### Tone
    Answer the question in a friendly tone.

    ### Response
    Please answer in a paragraph with high-level summary of the relevant visualizations.
    '''

    retrieved_results = f'''
    ### User question
    {query}

    ### Relevant visualization info
    {viz_info}
    '''

    openai.api_key = api_key
    messages = [{"role": "system", "content": system_prompt}, {"role":"user", "content": retrieved_results}]

    response = openai.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages = messages
    )

    #print(response)

    answer = response.choices[0].message.content
    #print(answer)

    viz_lists = []
    tableau_code_lists = []

    i = 1
    for index, row in search_results.iterrows():
        title = row["title"]
        url = row["url"]
        date = row["date"]
        tableau_code = row["tableau_code"]
        dist = row["distance"]
        if i > 3:
            break
        viz_lists.append(f"\n\n {i}. [{title}]({url}) on {date}")
        tableau_code_lists.append(tableau_code)
        i += 1

    return answer, viz_lists, tableau_code_lists
