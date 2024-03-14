from llama_index.llms.anthropic import Anthropic
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core import Settings
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from llama_index.readers.file import CSVReader
from pathlib import Path
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage, SimpleDirectoryReader, ServiceContext
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core import Settings
tokenizer = Anthropic().tokenizer
Settings.tokenizer = tokenizer
Settings.llm = Anthropic(model="claude-3-opus-20240229",api_key=ANTHROPIC_API_KEY)
from prompts import instruction_str, new_prompt
from preprocessing import clean
load_dotenv()

file_path = os.path.join('..','data','omfh_backup.csv')
df = pd.read_csv(file_path)
df.Motivation.fillna('',inplace=True)

# Apply to all texts
df['clean_text'] = df.Motivation.apply(clean)
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
llm = Anthropic(model="claude-3-opus-20240229")

df_query_engine = PandasQueryEngine(df=df.head(),verbose=True,instruction_str=instruction_str)
df_query_engine.update_prompts({"pandas_prompt": new_prompt})
documents = CSVReader().load_data(Path('../data/df_head.csv'))
service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm, embed_model="local")
