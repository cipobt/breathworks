from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
pdf_file_path = os.path.join('documents','breathworks_guide.pdf')
file_path = os.path.join('data','omfh_backup.csv')
file_path_new = os.path.join('data','combined_courses3.csv')

file_local = os.path.join('file.csv')
file_local

# def find_data():
#     if
