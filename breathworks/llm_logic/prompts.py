from llama_index.core import PromptTemplate


instruction_str = """\
    1. Convert the query to executable Python code using Pandas.
    2. The final line of code should be a Python expression that can be called with the `eval()` function.
    3. The code should represent a solution to the query.
    4. PRINT ONLY THE EXPRESSION.
    5. Do not quote the expression.
    6. If the prompt asks for a plot or chart, create the chart with an appropriate header, colour the chart in red
       and give the x-axis and y-axis a correct title if applicable.

    """

new_prompt = PromptTemplate(
    """\
    You are working with a pandas dataframe in Python.
    The name of the dataframe is `df`.
    This is the result of `print(df.head())`:
    {df_str}

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression: """
)

context = """Purpose: The primary role of this agent is to assist users by providing accurate
            information about the patients in this data set who have signed up for a breathing class. The agen has
            the choice to query the dataframe usings pandas or using it's llm capabilities to answer the question."""


