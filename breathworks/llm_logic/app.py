# Standard Library
import os
import requests
from pathlib import Path

# Local Files
from helpers import get_index
from preprocessing import clean
from param import ANTHROPIC_API_KEY,file_path_new_final
from prompts import instruction_str_pandas, instruction_str_llm, context, prompt_template_pandas, prompt_template_llm
from breathworks.utils import get_data

# Third part packages
import pandas as pd
import streamlit as st
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.llms.anthropic import Anthropic
from llama_index.readers.file import  CSVReader
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import matplotlib.pyplot as plt

# TODO
# Review the prompts - we saw some of the prompts should be formatting and adding in an actual query but they aren't = new_prompt
# Looking at the get_index function - doesn't always find the local index


@st.cache_resource()
def initialise():
    df = get_data('final_set')
    df.CustomerPurpose  = df.CustomerPurpose.fillna('')
    llm = Anthropic(model="claude-3-opus-20240229")
    Settings.llm = Anthropic(model="claude-3-opus-20240229",api_key=ANTHROPIC_API_KEY)
    # Initial huggingface as embedding model

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    tokenizer = Anthropic().tokenizer
    Settings.tokenizer = tokenizer

    # Output and reload Load documents and create indexes
    df.to_csv(file_path_new_final)
    documents = CSVReader().load_data(Path(file_path_new_final))


    df_index = get_index(documents,'df_index',llm=llm)

    # Create query engines
    pandas_query_engine = PandasQueryEngine(df=df,verbose=True,instruction_str=instruction_str_pandas)
    query_engine_llm = df_index.as_query_engine(llm=llm,instruction_str=instruction_str_llm)

    # Create prompt templates with updated prompts

    # pandas_query_engine.update_prompts({
    # "pandas_prompt": prompt_template_pandas
    # })

    # Update prompts for query_engine_llm
    query_engine_llm.update_prompts({
        "llm_prompt": prompt_template_llm
    })


    tools = [
        QueryEngineTool(
            query_engine=query_engine_llm,
            metadata=ToolMetadata(
                name="llm_query_engine",
                description="This queries the llm to answer text style questions",
            ),
        ),
        QueryEngineTool(
            query_engine=pandas_query_engine,
            metadata=ToolMetadata(
                name="pandas_query_engine",
                description="This creates plots using pandas",
            ),
        )
    ]

    llm = Anthropic(model="claude-3-opus-20240229")
    agent = ReActAgent.from_tools(tools, llm=llm, verbose=True,context=context,max_iterations=5)

    '''
    # Breathworks Website
    '''
    return agent, df


def main (agent,df):
    prompt = st.text_input('Prompt')
    # prompt = 'plot a chart of clients by gender'
    df=df
    if prompt:
        print (prompt)
        response = agent.chat(prompt)
        print(response)
        for source in response.sources:
            # Access the tool_name attribute of each ToolOutput object
            tool_name = source.tool_name
            print("Tool Name:", tool_name)
            if tool_name=='pandas_query_engine':
                raw_output = source.raw_output
                pandas_instruction_str = raw_output.metadata['pandas_instruction_str']
                fig, ax = plt.subplots()
                fig = exec(pandas_instruction_str)
                fig = plt.gcf()
                st.write(response.response)
                st.pyplot(fig)
                agent.reset()
            else:
                st.write(response.response)
                agent.reset()


if __name__ == "__main__":
    agent,df = initialise()
    main(agent,df)
