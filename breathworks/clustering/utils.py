import os
from google.cloud import bigquery
from dotenv import load_dotenv
import pandas as pd

def get_data():
    load_dotenv()
    PROJECT_ID = os.environ['GCP_PROJECT']
    DATASET_ID = os.environ['BQ_DATASET']
    TABLE_DATA_ID = os.environ['BQ_TABLE_DATA']

    client = bigquery.Client()

    query = f'SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_DATA_ID}`'
    query_job = client.query(query)
    df = query_job.to_dataframe()

    return df

def go_to_data(file_name):
    current_dir = os.getcwd()
    # Go up one level from the current directory
    parent_dir = os.path.dirname(current_dir)

    # Construct the path to the target directory 'raw_data'
    target_dir = os.path.join(parent_dir, 'breathworks/raw_data')

    # create the dataframe
    file_path = os.path.join(target_dir, file_name)
    df = pd.read_csv(file_path)
    return df

if __name__ == "__main__":
    dataframe = get_data()
    print(dataframe)
