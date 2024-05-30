#Import libraries and depenc
import streamlit as st
import replicate
import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
from datetime import datetime

llm = None

#Function to send data to HIFIS (still need to link to Snowflake)
def display_data_analytics():
    st.title("Data Analytics")
    if st.button("Send Data to HIFIS"):
        send_data_to_hifis()

    st.header('Network Graph of Co-Occurring Keywords')
    create_network_graph(st.session_state.messages)

    st.header('Sentiment Analysis Over Time')
    perform_sentiment_analysis(st.session_state.messages)
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def send_data_to_hifis():
    st.write("Sending data to HIFIS...")
    
#Data to generate response based on prompt
def generate_llama2_response(prompt_input):
    global llm  
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    try:
        output = replicate.run(
            llm, 
            input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
            "temperature": 0.1, "top_p": 0.9, "max_length": 150, "repetition_penalty": 1}
        )
        return output
    except Exception as e:
        st.error(f"Error during API call: {e}")
        return [""]

#List of key words for socioeconomic, geospatial, and behavioral keywords. Keywords are collected through LLM generated responses.
def detect_keywords_and_collect_data(message):
    socioeconomic_keywords = [
    "job", "work", "career", "unemployed", "school", "education", "degree", 
    "income", "earnings", "salary", "wage", "family", "finance", "assets", "debt", 
    "housing", "shelter", "accommodation", "services", "social services", "legal", "support",
    "jobless", "employed", "student", "graduate", "pay", "earn", "livelihood", "rent", 
    "mortgage", "landlord", "rental", "utilities", "bills", "welfare", "benefits", "aid",
    "food", "nutrition", "groceries", "medical bills", "insurance", "pension", "savings",
    "eviction", "foreclosure", "homeless", "homelessness", "crisis", "struggle", "emergency"
    ]

    geospatial_keywords = [
    "shelter", "population", "people", "community", "assistance", "aid", 
    "hospital", "clinic", "facility", "services", "food", "pantry", "store", 
    "crime", "safety", "security", "affordable housing", "location", "proximity", 
    "nearby", "surroundings", "neighborhood", "district", "zone", "town", "city",
    "vulnerable", "at-risk", "unsafe", "high-risk", "rough", "poverty", "urban", "rural"
    ]   

    behavioral_keywords = [
    "addiction", "substance", "drug", "alcohol", "mental health", "anxiety", "depression", 
    "psychological", "therapy", "counseling", "support", "interaction", "relationship", 
    "employment", "job", "career", "seeking", "transportation", "routine", "habit", 
    "risk", "relapse", "coping", "mechanism", "engagement", "activity", "exercise", 
    "leisure", "hobby", "pastime", "friendship", "socializing", "networking",
    "stress", "trauma", "isolation", "loneliness", "challenges", "struggle", "survival"
    ]

#Notification on when keywords are detected. 
    if any(keyword in message.lower() for keyword in socioeconomic_keywords):
        st.sidebar.info("Detected keywords related to Socioeconomic Data.")


    if any(keyword in message.lower() for keyword in geospatial_keywords):
        st.sidebar.info("Detected keywords related to Geospatial Data.")

    if any(keyword in message.lower() for keyword in behavioral_keywords):
        st.sidebar.info("Detected keywords related to Behavioral Insights.")

#Network graph of keywords collected will generate on a separate Streamlit tab, for users to have full transparency on what data is being collected
def create_network_graph(data):
    G = nx.Graph()

    keywords = set()
    for entry in data:
        message = entry["content"].lower()
        for word in message.split():
            keywords.add(word)
    for keyword in keywords:
        G.add_node(keyword)

    for entry in data:
        message = entry["content"].lower()
        words = message.split()
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                word1 = words[i]
                word2 = words[j]
                if not G.has_edge(word1, word2):
                    G.add_edge(word1, word2, weight=1)
                else:
                    G[word1][word2]['weight'] += 1

#Sentiment analysis graph generated on separate Streamlit tab, for users to have full transparency on what data is being collected
def perform_sentiment_analysis(messages):
    df = pd.DataFrame(messages)

    df['timestamp'] = datetime.now()

    sentiments = []
    for content in df['content']:
        blob = TextBlob(content)
        sentiments.append(blob.sentiment.polarity)

    df['sentiment'] = sentiments

    df['date'] = df['timestamp'].dt.date
    avg_sentiments = df.groupby('date')['sentiment'].mean()

    st.line_chart(avg_sentiments)

#Main page of application. Users will chat with Bella here, and Llama API key is confirmed here.
def main():
    st.set_page_config(page_title="BELLA")

    with st.sidebar:
        st.title("BELLAChat")
        st.write('BELLAChat is an open-source, domain-specific LLM trained on various documentation on housing, bylaws, societal standards, and more.')
    
    #Collection and confirmation of Llama API key
        replicate_api = None
        if 'REPLICATE_API_TOKEN' in st.secrets:
            st.success('API key already provided!', icon='✅')
            replicate_api = st.secrets['REPLICATE_API_TOKEN']
        else:
            replicate_api = st.text_input('Enter Replicate API token:', type='password')
            if replicate_api:
                if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
                    st.warning('Please enter a valid API token!', icon='⚠️')
                else:
                    st.success('Proceed to entering your prompt message!', icon='👉')
                    os.environ['REPLICATE_API_TOKEN'] = replicate_api

        st.subheader('Models and parameters')
        selected_model = st.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
        if selected_model == 'Llama2-7B':
            llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
        elif selected_model == 'Llama2-13B':
            llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    #If chat history needs to be cleared, users can click this button
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if replicate_api:
        if prompt := st.chat_input(disabled=not replicate_api):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_llama2_response(st.session_state.messages[-1]["content"])
                    placeholder = st.empty()
                    full_response = ''
                    for item in response:
                        full_response += item
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)
    else:
        st.warning("Please enter a valid API token to use the chat functionality.")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            if message["role"] == "user":
                detect_keywords_and_collect_data(message["content"])

    #Navigation sidebar
    main_page = st.sidebar.radio("Navigation", ["Chat", "Data Analytics"])
    if main_page == "Data Analytics":
        display_data_analytics()

if __name__ == "__main__":
    main()
