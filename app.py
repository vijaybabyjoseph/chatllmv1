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
Upload a dataset and ask natural language questions.\n 
Please provide clear and precise questions or statments as shown on the left.\n
The agent will generate the code to answer your question and display the results.
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
            df = pd.read_csv(uploaded_file)
            st.success("CSV loaded successfully!")
            st.dataframe(df.head(5))
            user_input = st.text_area("Enter your prompt:")

            # Generate output
            if st.button("Generate"):
                if user_input:
                    # call pandas_ai.run(), passing dataframe and prompt
                    with st.spinner("Generating response..."):
                        llm = initialize_llm(selected_model, api_key)
                        try:
                            prompt_out = set_agent_context(df,user_input)
                            prompt_out=prompt_out.strip()
                            #print(prompt_out)
                            print(f"Prompt being sent to agent: {prompt_out}")
                            output = run_agent(prompt_out,llm,df)
                            
                        except Exception as e:
                            output = f"Error: {e}. Try rephrasing your question."
                            st.error(output)
                else:
                    st.warning("Please enter a prompt.")           
                            
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Please upload a CSV file to get started.")

st.sidebar.subheader("Sample Questions:")
st.sidebar.markdown("""
- What is the total number of women in the data?
- What's the average age grouped by gender?
- Create a histogram using the income columns distribution.
- Create a corrplot using Age and Income columns.
- Create a dataframe that shows how many rows have Gender = M and how many rows have Gender =F.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Powered by OpenRouter and LangChain")
