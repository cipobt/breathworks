import os
from google.cloud import bigquery
from dotenv import load_dotenv

def get_data(table_id):
    load_dotenv()
    PROJECT_ID = os.environ['GCP_PROJECT']
    DATASET_ID = os.environ['BQ_DATASET']

    client = bigquery.Client()

    query = f'SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{table_id}`'
    query_job = client.query(query)
    df = query_job.to_dataframe()

    return df

if __name__ == "__main__":
    dataframe = get_data()
    print(dataframe)
