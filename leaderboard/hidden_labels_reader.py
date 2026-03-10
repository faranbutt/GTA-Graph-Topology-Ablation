import os
import pandas as pd
import io
from dotenv import load_dotenv

load_dotenv()

def read_hidden_labels():
    csv_string = os.environ.get("TEST_LABELS_CSV")
    
    if not csv_string:
         print("Error: TEST_LABELS_CSV not found. Have you added it to your .env file or secrets?")
         return None
     
    csv_string = csv_string.replace(r'\n', '\n')
    
    csv_buffer = io.StringIO(csv_string)
    df = pd.read_csv(csv_buffer)
    return df