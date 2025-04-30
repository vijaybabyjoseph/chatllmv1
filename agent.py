# agent.py

from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.output_parsers.fix import OutputFixingParser
from langchain.prompts import PromptTemplate
from config import FREE_MODELS
from utils import extract_python_code
import textwrap
import streamlit as st
import matplotlib.pyplot as plt

def initialize_llm(model_name, api_key):
    return OpenAI(
        model_name=FREE_MODELS[model_name],
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        streaming=True,
        temperature=0
    )

def set_agent_context(df,query):
    column_headers_list = df.columns.tolist()
    column_headers = ", ".join(column_headers_list) 

    sample_df = df.head(2)
    sample_df_string = sample_df.to_string()


    # query="""
    # what is the average of the 'Age' column?
    # """


    # query="""
    # Create a histogram chart with the 'Age' column?
    # """

    # query="""
    # Create a pie chart with the 'Gender' column
    # """

    # query="""
    # Create a corrplot chart with the 'Income' and 'Age' column
    # """


    prompt = f"""
    You are a data analysis assistant.
    You will be provided with a pandas dataframe and 
    you will be asked to perform various operations on it.
    Do not provide code to load the dataframe like df = pd.read_csv('data.csv'). Refer to the dataframe as 'df'.
    If you are asked to plot a chart, provide the code to plot the chart using matplotlib. And do not use plotly

    Answer the following question with python code based on the dataframe provided below:
    {query}

    The dataframe has the following columns: {column_headers}.

    Here are a few rows of the dataframe:
    {sample_df_string}

    Provide only python code as output. Do not include any explanation or comments.
    
    If plt.show() appears in the code, replace it with st.pyplot(plt.gcf())
    If there is a chart or plot in the code display it using st.pyplot(plt.gcf()).
    If the code produces a string or a number and assigns it to a variable, display it using st.write().
    If the code produces a dataframe, assign it to a out_df and display it using st.dataframe(df).

    Ensure the code is executable in a Streamlit app.
    Use only matplotlib.pyplot for plotting.
    Return the only code <code> in the following format ```python <code>```
    """
    return prompt
    #prompt_template =p PromptTemplate.from_template(prompt_context)
    #prompt = prompt_template.format(data_sample=sample_df_string)



    
def run_agent(prompt, llm,df):
    output = llm.invoke(prompt)
    #print(output)
    df = df.copy()
    output = extract_python_code(output)
    output = str(output)
    
    lib_imports= """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
    """
    output = lib_imports + "\n" + output
    output=output.strip()
    
    phrase_to_check = "plt.show()"
    if phrase_to_check in output:
        output = output.replace("plt.show()", "st.pyplot(plt.gcf())")
        
    output = textwrap.dedent(output)
    st.write("Code being executed:")
    st.code(output, language='python')
    st.write("Output:")
    print(output)    
    exec(output)
    
    
def build_agent2(llm, df):
    prompt_template = PromptTemplate(
        input_variables=["input"],
        template="You are analyzing a pandas dataframe. Provide your answer in JSON format with keys 'thought', 'action', and 'result'. If your response includes code, ensure it is clearly marked as an 'action'. Question: {input}"
    )

    output_parser = OutputFixingParser.from_llm(
        llm=llm,
        parser=None,  # Optional: plug in a parser like StrOutputParser()
        prompt=prompt_template,
        max_retries=2
    )

    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        handle_parsing_errors=True,
        allow_dangerous_code=True,
        output_parser=output_parser,
        max_iterations=5,
        early_stopping_method="generate",
        return_intermediate_steps=True
    )
