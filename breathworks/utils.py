import os
from google.cloud import bigquery
from dotenv import load_dotenv

def get_data(table_id):
    load_dotenv()
    PROJECT_ID = os.environ['GCP_PROJECT']
    DATASET_ID = os.environ['BQ_DATASET']

    client = bigquery.Client()

<<<<<<< HEAD
    query = f'SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_DATA_ID}`'
    print(query)
=======
    query = f'SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{table_id}`'
>>>>>>> 8d464e49bbd300bc51488aeb70bf3a96eb6d84ef
    query_job = client.query(query)
    df = query_job.to_dataframe()
    # print(query)
    return df

if __name__ == "__main__":
    dataframe = get_data()
    print(dataframe)
