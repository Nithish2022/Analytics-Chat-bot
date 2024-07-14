import streamlit as st
import pandas as pd
from pymongo import MongoClient
from bson.codec_options import CodecOptions, DatetimeConversion
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pandasai import Agent
from streamlit_chat import message
from pandasai.pipelines.chat.generate_chat_pipeline import GenerateChatPipeline


# Load environment variables from .env file
load_dotenv()

# Function to chat with data
def chat_with_csv(df, query):
    groq_api_key = os.environ['GROQ_API_KEY']

    llm = ChatGroq(
        groq_api_key=groq_api_key, model_name="llama3-70b-8192",
        temperature=0.6
    )

    pandas_ai = Agent(df, config={"llm": llm,"conversational": True,"use_error_correction_framework": True},memory_size=20,pipeline=GenerateChatPipeline)
    
    result = pandas_ai.chat(query)
    return result

# Set layout configuration for the Streamlit page
st.set_page_config(layout='centered',initial_sidebar_state='collapsed')

# Connect to MongoDB database
mongodb_url = os.environ['MONGODB_URL']
client = MongoClient(mongodb_url, datetime_conversion=DatetimeConversion.DATETIME_AUTO)

# Lock the database name to "LPT"
selected_db = "LPT"

# Fetch collections from the selected database
collection_names = client[selected_db].list_collection_names()

# Sidebar for Collection selection
with st.sidebar:
    selected_collection = st.selectbox("Select a collection", collection_names)

# Function to fetch data from a collection
def fetch_data(db_name, collection_name):
    db = client[db_name]
    collection = db.get_collection(collection_name, codec_options=CodecOptions(tz_aware=True, datetime_conversion=DatetimeConversion.DATETIME_AUTO))
    cursor = collection.find()
    data = []
    for document in cursor:
        document['_id'] = str(document['_id'])  # Convert ObjectId to string
        data.append(document)
    return pd.DataFrame(data)

# Fetch data from the selected collection
if selected_db and selected_collection:
    df = fetch_data(selected_db, selected_collection)

    # Initialize chat history in session state
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! I'm your virtual assistant. I'm here to provide information about your shop floor !🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    # Container for the chat history
    response_container = st.container()
    # User's text input
    user_input = st.chat_input("Send a message", key='input')

    # Process user input and generate response
    if user_input:
        st.session_state['past'].append(user_input)
        output = chat_with_csv(df, user_input)

        # Ensure output is JSON serializable
        if isinstance(output, pd.DataFrame):
            output = output.applymap(lambda x: int(x) if isinstance(x, pd.Int64Dtype) else x)
            st.session_state['generated'].append(output)
        elif isinstance(output, str):
            st.session_state['generated'].append(output)
        elif isinstance(output, dict):
            if output.get('type') == 'plot':
                st.session_state['generated'].append(output['value'])
            elif output.get('type') == 'number':
                st.session_state['generated'].append(str(output['value']))

    # Display chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="no-avatar")
                if isinstance(st.session_state["generated"][i], pd.DataFrame):
                    st.dataframe(st.session_state["generated"][i])
                elif os.path.exists(st.session_state["generated"][i]):
                    st.image(st.session_state["generated"][i])
                else:
                    message(st.session_state["generated"][i], key=str(i), avatar_style="no-avatar", logo='https://is4-ssl.mzstatic.com/image/thumb/Purple113/v4/cb/49/00/cb4900e0-e38b-474c-dd29-f770b307a7f7/source/512x512bb.jpg')


# Close MongoDB client
client.close()
