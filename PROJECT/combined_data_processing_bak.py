import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Define paths
DATASET_PATH = "./datasets"
RAW_DATA_PATH = DATASET_PATH
PROCESSED_DATA_PATH = os.path.join(DATASET_PATH, "processed10percent")
# Set up logging
if not os.path.exists(PROCESSED_DATA_PATH):
    os.makedirs(PROCESSED_DATA_PATH)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=os.path.join(PROCESSED_DATA_PATH, 'combined_data_processing.log'),
    filemode='a'
)

# Create a separate logger for errors, warnings, and failures
error_logger = logging.getLogger('error_logger')
error_logger.setLevel(logging.WARNING)

# Create a file handler for the error logger
error_file_handler = logging.FileHandler('error_and_warnings.log', mode='w')
error_file_handler.setLevel(logging.WARNING)

# Create a formatter for the error logger
error_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
error_file_handler.setFormatter(error_formatter)

# Add the file handler to the error logger
error_logger.addHandler(error_file_handler)



def remove_username_string(username):
    return username[3:]

def load_csv(file_path, dataset_name, start_row, batch_size):
    columns_to_read = {
        "email"  : ["date", "user", "pc", "from", "to", "attachments"],
        "file"   : ["date", "user", "pc", "filename", "file_activity"],
        "device" : ["date", "user", "pc", "device_activity"],
        "logon"  : ["date", "user", "pc", "logon_activity"]
    }
    
    try:
        return pd.read_csv(file_path, usecols=columns_to_read[dataset_name], skiprows=range(1, start_row), nrows=batch_size)
    except Exception as e:
        logging.error(f"Error loading {dataset_name} data: {str(e)}")
        return None

def preprocess_data(df, dataset_name):
    if df is None:
        return None
    
    try:
        df['pc']      = df['pc'].str.replace('PC-', '', regex=False)
        df['user']    = df['user'].apply(remove_username_string).astype('int')
        if dataset_name == "email":
            df['to_outside']     = ((df['from'].str.split('@').str[1] == 'dtaa.com') & (df['to'].str.split('@').str[1] != 'dtaa.com'))
            df['from_outside']   = ((df['from'].str.split('@').str[1] != 'dtaa.com') & (df['to'].str.split('@').str[1] == 'dtaa.com'))
            df                   = df.drop(['from', 'to'], axis=1)
            df['attachments']    = df['attachments'].fillna(False)
            df['has_attachment'] = df['attachments'].apply(lambda x: True if x else False)
            df                   = df.drop(columns=['attachments'])
        
        elif dataset_name == "file":
            df['file_ext']       = df['filename'].str.extract(r'(\.[^.]+$)', expand=False).str[1:]
            df['is_exe']         = (df['file_ext'] == 'exe')
            # df['not_exe'] = (df['file_ext'] != 'exe')
            df['file_open']      = (df['file_activity'] == 'File Open')
            df['file_write']     = (df['file_activity'] == 'File Write')
            df['file_copy']      = (df['file_activity'] == 'File Copy')
            df['file_delete']    = (df['file_activity'] == 'File Delete')
            df                   = df.drop(columns=['filename', 'file_activity', 'file_ext'])
        
        elif dataset_name == "device":
            df                   = pd.get_dummies(df, columns=['device_activity'], prefix='', prefix_sep='')
            df.rename(columns={'Connect': 'connected', 'Disconnect': 'disconnected'}, inplace=True)
            df.drop('disconnected', axis=1, inplace=True)
        
        elif dataset_name == "logon":
            df                   = pd.get_dummies(df, columns=['logon_activity'], prefix='', prefix_sep='')
            df.rename(columns={'Logon': 'logon', 'Logoff': 'logoff'}, inplace=True)
            df.drop('logoff', axis=1, inplace=True)
        
        return df
    except Exception as e:
        logging.error(f"Error preprocessing {dataset_name} data: {str(e)}")
        return None

def merge_datasets(datasets):
    try:
        for df in datasets:
            if df is not None:
                df['date']       = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
        
        merged_df      = pd.concat(datasets, axis=1)
        merged_df      = merged_df.loc[:, ~merged_df.columns.duplicated()]
        merged_df      = merged_df.sort_values('date').reset_index(drop=True)
        
        return merged_df
    except Exception as e:
        logging.error(f"Error merging datasets: {str(e)}")
        return None

def save_dataset(df, filename):
    if df is None:
        logging.error("Cannot save None dataset.")
        return
    
    try:
        output_path      = os.path.join(PROCESSED_DATA_PATH, filename)
        df.to_csv(output_path, index=False)
        logging.info(f"Saved processed dataset to {output_path}")
    except Exception as e:
        logging.error(f"Error saving dataset: {str(e)}")

def process_batch(start_row, batch_size, batch_number):
    datasets = {
        "email"  : load_csv(os.path.join(RAW_DATA_PATH, "normalizedEmail.csv"), "email", start_row, batch_size),
        "file"   : load_csv(os.path.join(RAW_DATA_PATH, "normalizedFile.csv"), "file", start_row, batch_size),
        "device" : load_csv(os.path.join(RAW_DATA_PATH, "device.csv"), "device", start_row, batch_size),
        "logon"  : load_csv(os.path.join(RAW_DATA_PATH, "logon.csv"), "logon", start_row, batch_size)
    }
    
    processed_datasets        = {name: preprocess_data(df, name) for name, df in datasets.items()}
    
    merged_dataset            = merge_datasets([df for df in processed_datasets.values() if df is not None])
    
    if merged_dataset is not None:
        filename              = f"merged_dataset_batch_{batch_number}.csv"
        save_dataset(merged_dataset, filename)
    else:
        logging.error(f"Failed to create merged dataset for batch {batch_number}.")

def main():
    total_rows      = max(
        sum(1 for line in open(os.path.join(RAW_DATA_PATH, file))) - 1
        for file in ["normalizedEmail.csv", "normalizedFile.csv", "device.csv", "logon.csv"]
    )
    batch_size      = max(100000, int(total_rows * 0.1))  # 10% of total rows or 100,000, whichever is larger

    for batch_number, start_row in enumerate(range(0, total_rows, batch_size), 1):
        logging.info(f"Processing batch {batch_number}")
        process_batch(start_row, batch_size, batch_number)

    logging.info("Batch processing completed.")

if __name__ == "__main__":
    main()
