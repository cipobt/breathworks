import os
import sys
import pandas as pd

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))

# Add the project root to the sys.path.
sys.path.append(project_root)

# Import after adjusting sys.path
from breathworks.utils import get_data

def get_data() -> pd.DataFrame:
    """
    Fetches data using the breathworks utility and processes it for the AI model.
    You can implement any required processing steps here. For example, filtering, cleaning, etc.
    """
    try:
        df = get_data()
        return df[['CustomerPurpose']]

    except ConnectionError:
        print("Connection error occurred. Please check your network connection.")
    except RuntimeError as e:
        print(f"An error occurred during data processing: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
