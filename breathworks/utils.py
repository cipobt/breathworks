import os
from google.cloud import bigquery
from dotenv import load_dotenv

def get_data():
    load_dotenv()
    PROJECT_ID = os.environ['GCP_PROJECT']
    DATASET_ID = os.environ['BQ_DATASET']
    TABLE_DATA_ID = os.environ['BQ_TABLE_DATA']

    client = bigquery.Client()

    query = f'SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_DATA_ID}`'
    print(query)
    query_job = client.query(query)
    df = query_job.to_dataframe()
    # print(query)
    return df

if __name__ == "__main__":
    dataframe = get_data()
    print(dataframe)
