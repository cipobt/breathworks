from llama_index.llms.anthropic import Anthropic
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core import Settings
import os
import pandas as pd
from llama_index.readers.file import CSVReader
from pathlib import Path
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage, SimpleDirectoryReader, ServiceContext
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core import Settings
tokenizer = Anthropic().tokenizer
from prompts import instruction_str, new_prompt, context
from preprocessing import clean
from param import ANTHROPIC_API_KEY
from helpers import get_index
import streamlit as st
import requests

file_path = os.path.join('data','omfh_backup.csv')
df = pd.read_csv(file_path)
df.Motivation.fillna('',inplace=True)

# Apply to all texts
df['clean_text'] = df.Motivation
llm = Anthropic(model="claude-3-opus-20240229")
Settings.tokenizer = tokenizer
Settings.llm = Anthropic(model="claude-3-opus-20240229",api_key=ANTHROPIC_API_KEY)

df_query_engine = PandasQueryEngine(df=df.head(),verbose=True,instruction_str=instruction_str)
df_query_engine.update_prompts({"pandas_prompt": new_prompt})
documents = CSVReader().load_data(Path('data/df_head.csv'))
service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm, embed_model="local")
df_index = get_index(documents,'df_index',llm=llm,service_context=service_context)

df_query_engine = PandasQueryEngine(df=df.head(),verbose=True,instruction_str=instruction_str)
df_query_engine.update_prompts({"pandas_prompt": new_prompt})
query_engine_llm = df_index.as_query_engine(llm=llm)

tools = [
    QueryEngineTool(
        query_engine=df_query_engine,
        metadata=ToolMetadata(
            name="df_data",
            description="this translates human lanuage into a pandas query",
        ),
    ),
    QueryEngineTool(
            query_engine=query_engine_llm,
        metadata=ToolMetadata(
            name="llm_data",
            description="This queries the llm to answer text style questions",
        ),
    ),
]

llm = Anthropic(model="claude-3-opus-20240229")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

'''
# Breathworks Website
'''

prompt = st.text_input('Prompt')
if prompt:
    response = agent.chat(prompt)

    for source in response.sources:
        # Assuming source is an instance of ToolOutput or a similar class
        content = source.content
        tool_name = source.tool_name

        # Accessing attributes of nested custom objects
        raw_input = source.raw_input['input']  # Assuming raw_input is a dictionary attribute
        raw_output_response = source.raw_output.response  # As

        st.write(raw_output_response)
