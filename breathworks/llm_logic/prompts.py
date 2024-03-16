from llama_index.core import PromptTemplate

# Define different sets of instructions
instruction_str_pandas = """\
    1. Convert the query to executable Python code using Pandas.
    2. The final line of code should be a Python expression that can be called with the `eval()` function.
    3. The code should represent a solution to the query.
    4. PRINT ONLY THE EXPRESSION.
    5. Do not quote the expression.
    6. If the prompt asks for a plot or chart, create the chart with an appropriate header, colour the chart in red
       and give the x-axis and y-axis a correct title if applicable.
    """

instruction_str_llm = """\
    1. Check the applicable columns of the df which have the data type text.
    2. Return the answer short and consise.
    """

instruction_str_pdf = """\
    1. Go through the pdf document which features meditation techniques.
    2. Return the answer short and concise.
    """

# Create a dictionary to map instruction strings to prompt templates
prompt_templates = {
    "instruction_str_pandas":
    PromptTemplate("""\
        You are working with a pandas dataframe in Python.
        The name of the dataframe is `df`.
        This is the result of `print(df.head())`:
        {df_str}

        Follow these instructions:
        {instruction_str}
        Query: {query_str}

        Expression:
        """),
    "instruction_str_llm":
    PromptTemplate("""\
        You are working with a pandas dataframe in Python.
        The name of the dataframe is `df`.
        This is the result of `print(df.head())`:
        {df_str}

        Follow these instructions:
        {instruction_str_llm}
        Query: {query_str}

        Expression:
        """),
    "instruction_str_pdf":
    PromptTemplate("""\
        You are working with a pandas dataframe in Python.
        The name of the dataframe is `df`.
        This is the result of `print(df.head())`:
        {df_str}

        Follow these instructions:
        {instruction_str_pdf}
        Query: {query_str}

        Expression:
        """)
}

context = """Purpose: The primary role of this agent is to assist users by providing accurate
information about the clients in this data set who have signed up for a breathing class. The agent has
the choice to query the dataframe usings pandas, using the breathworks meditation pdf which features
explanations about different types of meditations or using it's llm capabilities to answer the question.
When asked about about data from the data frame which refers to the history or motivation column, use the `query_engine_llm` query engine.
"""

prompt_template_df = prompt_templates.get('instruction_str_pandas')
prompt_template_llm = prompt_templates.get('instruction_str_llm')
prompt_template_pdf = prompt_templates.get('instruction_str_pdf')
