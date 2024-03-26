# Standard Library
import os
import requests
from pathlib import Path

# Local Files
from helpers import get_index
from preprocessing import clean
from param import file_local, ANTHROPIC_API_KEY
from prompts import instruction_str_pandas, instruction_str_llm, prompt_template_df, prompt_template_llm

# Third part packages
import pandas as pd
import streamlit as st
from llama_index.core.agent import ReActAgent
from llama_index.llms.anthropic import Anthropic
from llama_index.readers.file import CSVReader
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# TODO
# Review the prompts - we saw some of the prompts should be formatting and adding in an actual query but they aren't = new_prompt
# Looking at the get_index function - doesn't always find the local index

df = pd.read_csv(file_local)
st.sidebar.image('logo.jpg', width=300)
st.title("Breathworks Agent")
prompt = st.text_input('Prompt')

# @st.cache_resource()
def initialise():

    # df.Motivation = df.Motivation.fillna('')
    # df['clean_text'] = df.Motivation
    def model():
        llm = Anthropic(model="claude-3-opus-20240229")
        Settings.llm = Anthropic(model="claude-3-opus-20240229",api_key=ANTHROPIC_API_KEY)
        # Initial huggingface as embedding model

        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        tokenizer = Anthropic().tokenizer
        Settings.tokenizer = tokenizer

        # Load documents and create indexes
        documents = CSVReader().load_data(Path('file.csv'))
        # pdf_documents = PDFReader().load_data(Path(pdf_file_path))


        df_index = get_index(documents,'df_index',llm=llm)
        # pdf_index = get_index(pdf_documents,'pdf_index',llm=llm)

        # Create query engines
        df_query_engine = PandasQueryEngine(df=df.head(),verbose=True,instruction_str=instruction_str_pandas)
        query_engine_llm = df_index.as_query_engine(llm=llm,instruction_str=instruction_str_llm)
        # query_engine_pdf = pdf_index.as_query_engine(llm=llm, instruction_str=instruction_str_pdf)

        # Create prompt templates with updated prompts

        df_query_engine.update_prompts({
        "pandas_prompt": prompt_template_df
        })

        # Update prompts for query_engine_llm
        query_engine_llm.update_prompts({
            "llm_prompt": prompt_template_llm
        })

        # # Update prompts for query_engine_pdf
        # query_engine_pdf.update_prompts({
        #     "pdf_prompt": prompt_template_pdf
        # })

        # Update prompts for df_query_engine
        df_query_engine.update_prompts({
        "pandas_prompt": prompt_template_df
        })

        # Update prompts for query_engine_llm
        query_engine_llm.update_prompts({
            "llm_prompt": prompt_template_llm
        })

        # # Update prompts for query_engine_pdf
        # query_engine_pdf.update_prompts({
        #     "pdf_prompt": prompt_template_pdf
        # })


        tools = [
            QueryEngineTool(
                query_engine=query_engine_llm,
                metadata=ToolMetadata(
                    name="llm_data",
                    description="This queries the llm to answer text style questions",
                ),
            )]

        llm = Anthropic(model="claude-3-opus-20240229")
        agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

        '''
        # Breathworks Website
        '''
        return agent

    if st.button('Show Data'):
    # Display the DataFrame when the button is clicked
        st.write(model())



def main (agent):
    # prompt = 'what is the number 1 reason in 1 word why people join the class?'
    if prompt:
        response = agent.query(prompt)
        print(response)
        # for source in response.sources:
        #     content = source.content
        #     tool_name = source.tool_name

        #     raw_input = source.raw_input['input']  # Assuming raw_input is a dictionary attribute
        #     raw_output_response = source.raw_output.response  # As

        st.write(response.response)


if __name__ == "__main__":
    agent = initialise()
    main(agent)
