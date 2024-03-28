from llama_index.core import PromptTemplate

# Define different sets of instructions
instruction_str_pandas = """\
    1. Convert the query to executable Python code using Pandas.
    3. The code should represent a solution to the query.
    4. PRINT ONLY THE EXPRESSION.
    5. Do not quote the expression.
    6. You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
    7. This is the result of `print(df.head())`: {df_str}
    8. The prompt will ask for a plot or chart, create the chart with an appropriate header, colour the chart in blue
       and give the x-axis and y-axis a correct title if applicable.
    9. Make sure you use 'import matplotlib.pyplot as plt' and 'import pandas as plt' to create the plots.
    """

instruction_str_llm = """\
    1. Check the applicable columns of the df which have the data type text.
    2. Return the answer short and consise.
    """


# Create a dictionary to map instruction strings to prompt templates
prompt_templates = {
    "prompt_templates_pandas":
    PromptTemplate("""\
        You are working with a pandas dataframe in Python.
        The name of the dataframe is `df`.
        This is the result of `print(df.head())`:
        {df_str}

        Follow these instructions:
        {instruction_str_pandas}
        Query: {query_str}

        Expression:
        """),
    "prompt_templates_llm":
    PromptTemplate("""\
        You are working with a pandas dataframe in Python.
        The name of the dataframe is `df`.
        This is the result of `print(df.head())`:
        {df_str}

        Follow these instructions:
        {instruction_str_llm}
        Query: {query_str}

        Expression:
        """)

}

context = """Purpose: The primary role of this agent is to assist users by providing accurate
information about the clients in this data set who have signed up for a breathing class. The agent has
the choice to use it's llm capabilities to answer question in conversation style about the dataset.
When asked to return a plot, use the pandas_query_engine to return a matplotly object.
"""

prompt_template_pandas = prompt_templates.get('instruction_str_pandas')
prompt_template_llm = prompt_templates.get('instruction_str_llm')
