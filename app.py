# app.py

import streamlit as st
from dotenv import load_dotenv
from config import FREE_MODELS
from utils import load_csv
from agent import initialize_llm, set_agent_context, run_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

load_dotenv()

st.set_page_config(
    page_title="AI Data Analysis Agent V1",
    #page_icon="resources/logo.jpg",
    layout="wide"
)

#st.image("resources/logo.jpg", width=150)
st.title("AI Data Analysis Agent")
st.markdown("""
Upload a dataset and ask natural language questions. The AI will analyze your data and return insights.
""")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("OpenRouter API Key", type="password")
selected_model = st.sidebar.selectbox("Choose Model", options=list(FREE_MODELS.keys()))

if not api_key:
    st.error("Please enter your OpenRouter API Key.")
else:
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = load_csv(uploaded_file)
            st.success("CSV loaded successfully!")
            st.dataframe(df.head(5))

            llm = initialize_llm(selected_model, api_key)
            

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if user_input := st.chat_input("Ask a question about your data:"):
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    st_callback = StreamlitCallbackHandler(st.container())
                    try:
                       prompt_out = set_agent_context(df,user_input)
                       prompt_out=prompt_out.strip()
                       #print(prompt_out)
                       output = run_agent(prompt_out,llm,df)
                    except Exception as e:
                        output = f"Error: {e}. Try rephrasing your question."
                        st.error(output)

                st.session_state.messages.append({"role": "assistant", "content": output})
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Please upload a CSV file to get started.")

st.sidebar.subheader("Help")
st.sidebar.markdown("""
**Example Questions**:
- What is the total number of women in the data?
- What's the average age grouped by gender?
- Create a histogram of salary distribution.
- Find correlations between all numeric columns.
- Which products have the highest sales?
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Powered by Langchain and state-of-the-art LLMs")
