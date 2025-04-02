import pandas as pd
from   tensorflow.keras.models import Model                             #type: ignore
from   tensorflow.keras.layers import LSTM, Dense, Input, Reshape       #type: ignore
from   tensorflow.keras        import models                            #type: ignore
from   keras.optimizers        import Adam                              #type: ignore
from   keras.callbacks         import ReduceLROnPlateau                 #type: ignore
import logging
import numpy as np
import tensorflow as tf
import os
import gc

# ---- INITIALIZATION STARTS ------
BASE_PATH     = "./"
MODEL_PATH    = os.path.join(BASE_PATH, "saved_models")
DATASET       = "../datasets/processed10percent/"
path_list     = [MODEL_PATH]
EPOCHS        = 11 # NUM of epoch for each dataset i.e 8 dataset which is 8 x 50 = 400 EPOCHS
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= os.path.join(BASE_PATH, "lstm_model_training.log"), filemode='a')
for path in path_list:
    if not os.path.exists(path=path):
        os.makedirs(path)
        logging.info(f'Created Folder {path.split("/")[-1]} at {path}')

tf.config.run_functions_eagerly(True)  # THIS LINE I ADDED, THAT SOLVED MY ISSUE ...
tf.data.experimental.enable_debug_mode()
"""
XLA requires ptxas version 11.8 or higher
	 [[{{node StatefulPartitionedCall}}]] [Op:__inference_one_step_on_iterator_3663]
"""
# ---- INITIALIZATION ENDS ------

def generate_sequence(dataset, sequence_length):
    logging.info(f"Generating sequences of length {sequence_length} for dataset")
    train_sequences     = []
    for i in range(len(dataset) - sequence_length):
        sequence    = dataset.iloc[i:i+sequence_length]
        train_sequences.append(sequence)
    # logging.info(f"Sequence List in generate sequence function \n{train_sequences[0:sequence_length]}")
    logging.info(f"Generated sequences")
    # sequences_array     = np.asanyarray(train_sequences)
    # sequences_array     = np.where(sequences_array, 1.0, 0.0).astype(np.float32)
    return np.asarray(train_sequences).astype(np.float32)

def create_lstm_anomaly_model(input_shape):
    # Define the input layer
    inputs = Input(shape=input_shape)
    logging.info(f"Input shape used for LSTM input {input_shape}")
    # Create the LSTM layers
    lstm_outputs = LSTM(50, activation='relu', return_sequences=True)(inputs)
    lstm_outputs = LSTM(25, activation='relu')(lstm_outputs)

    # Create the output layer
    # Create the Dense layer with the exact output size
    dense_outputs = Dense(input_shape[0] * input_shape[1])(lstm_outputs)
    
    # Reshape to match the original input shape
    outputs    = Reshape((input_shape[0], input_shape[1]))(dense_outputs)
    logging.info(f"Dense used input as input_shape[1] is {input_shape[1]}")
    # Create the model
    model      = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    optimizer  = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    logging.info("LSTM Model Compiled")
    return model

def get_callbacks():
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    return [reduce_lr]

def start_training(model_instance, to_just_compute_anomaly=False):
    logging.info("Starting training process")
    intermediate_model = os.path.join(MODEL_PATH, "intermediate_model.keras")
    callbacks          = get_callbacks()
    if to_just_compute_anomaly:
        logging.info(f"Skipping the model training step ...")
        final_model      = os.path.join(MODEL_PATH, "LSTM_Final_model.keras")
        logging.info(f"Loaded the model from path {final_model}")
        model_instance   = model_instance = models.load_model(final_model)
        return model_instance
    
    if not os.path.exists(intermediate_model):
        train_data       = pd.read_csv(os.path.join(DATASET, "merged_dataset_batch_1.csv"))

        train_data       = preprocess(train_data)
        logging.info(f"Loaded first batch dataset with shape {train_data.shape}")

        sequences_list   = generate_sequence(train_data, sequence_length=10)

        # logging.info(f"This is the sequence data i see {sequences_list}")

        model_instance.fit(
            sequences_list,
            epochs      = EPOCHS,
            batch_size  = 32,
            callbacks   = callbacks
        )

        logging.info("First batch training completed")
        save_model(model=model_instance, model_name="intermediate_model")
        memory_cleanup()
        logging.info(f"Batch 1 Intermediate model saved and memory cleaned")

    for i in range(2, 9):
        logging.info(f"Loading model for batch {i}")
        model_instance = models.load_model(intermediate_model)
        file_name      = f"merged_dataset_batch_{i}.csv"
        file_path      = os.path.join(DATASET, file_name)
        batch_data     = pd.read_csv(file_path)
        batch_data     = preprocess(batch_data)
        logging.info(f"Loaded batch {i} dataset with shape {batch_data.shape}")

        if i > 1:
            prev_file_name       = f"merged_dataset_batch_{i-1}.csv"
            prev_file_path       = os.path.join(DATASET, prev_file_name)
            total_rows_to_read   = 10_000
            prev_batch_data      = pd.read_csv(prev_file_path, nrows=total_rows_to_read)
            prev_batch_data      = preprocess(prev_batch_data)
            batch_data           = pd.concat([prev_batch_data, batch_data])
            logging.info(f"Added 10% of Previous dataset to the current dataset")
            logging.info(f"batch_data shape now became {batch_data.shape}")

        sequences_list = generate_sequence(batch_data, sequence_length=10)

        model_instance.fit(
            sequences_list,
            epochs      = EPOCHS,
            batch_size  = 32,
            callbacks   = callbacks
        )

        logging.info(f"Batch {i} training completed")
        save_model(model=model_instance, model_name="intermediate_model")
        memory_cleanup()
        logging.info(f"Batch {i} Intermediate model saved and memory cleaned")

    logging.info("Training process completed")
    return model_instance

def memory_cleanup(model=None):
    # Clear Keras backend session
    tf.keras.backend.clear_session()
    
    # Force garbage collection
    gc.collect()
    
    # # Optional: Clear specific model if provided
    # if model is not None:
    #     del model
    
    # Clear any large numpy arrays or tensors
    try:
        # Clear any existing large numpy arrays
        for var in list(globals().keys()):
            if isinstance(globals()[var], (np.ndarray, tf.Tensor)):
                del globals()[var]
    except Exception as e:
        print(f"Error during cleanup: {e}")

def save_model(model, model_name):
    try:
        model_path = os.path.join(MODEL_PATH, model_name + '.keras')
        tf.keras.models.save_model(model, model_path)
        logging.info(f"Model '{model_name}' sucessfully saved to {model_path}")

        try:
            tf.keras.models.load_model(model_path)
            logging.info(f"Model '{model_name}' sucessfully verified after saving.")
        except Exception as verify_error:
            logging.error(f"Model '{model_name}' saved but failed verification: {verify_error}")
    
    except tf.errors.OpError as op_error:
        logging.error(f"TensorFlow Operation error while saving model '{model_name}': {op_error}")
    except PermissionError:
        logging.error(f"Permission denied when attempting to save model '{model_name}'. Check directory permissions.")
    except IOError as io_error:
        logging.error(f"I/O error occurred while saving model '{model_name}': {io_error}")
    except Exception as e:
        logging.error(f"Unexpected error occurred while saving model '{model_name}': {e}")
        logging.debug("Error details:", exc_info=True)

def compute_anomaly_threshold(model, sequence_length):
    filename       = "merged_dataset_batch_10.csv"
    filepath       = os.path.join(DATASET, filename)
    val_data       = pd.read_csv(filepath)
    val_data       = preprocess(val_data)
    val_seqs       = generate_sequence(val_data, sequence_length=10)

    predictions    = model.predict(val_seqs)

    mse            = np.mean(np.power(val_seqs - predictions, 2), axis=1)

    threshold      = np.mean(mse) + 2 * np.std(mse)

    logging.info("Computed and cached new threshold value")
    logging.info(f"""Anomaly Threshold Computation:
                Mean MSE  : {np.mean(mse)}
                Std MSE   : {np.std(mse)}
                Threshold : {threshold}""")
    
    return threshold

def detect_anomaly(model, data, threshold):
    # Predict values for data
    predictions = model.predict(data)
    
    # Compute MSE between predicted and actual values
    mse = np.mean(np.power(data - predictions, 2), axis=1)
    
    # Check if MSE is above threshold
    if mse > threshold:
        return True  # Anomaly detected
    else:
        return False  # No anomaly detected

def preprocess(dataset):
    # logging.info(f"Original dataset shape: {dataset.shape}")
    # logging.info(f"NA values in original dataset:\n {dataset.isnull().sum()}")
    processed_dataset   = dataset.drop(columns="date", axis=1)

    return processed_dataset

def get_input():
    filepath       = os.path.join(DATASET, "merged_dataset_batch_1.csv")
    df             = pd.read_csv(filepath, nrows=10)
    df             = preprocess(df)
    logging.info(f"Shape of the input {df.shape}")
    return df.shape

def pipeline():
    logging.info("Started Pipeline ...")
    sequence       = 10
    input_shape    = get_input()
    lstm_model     = create_lstm_anomaly_model(input_shape=input_shape)
    model_state    = start_training(lstm_model, to_just_compute_anomaly=False)
    save_model(model=model_state, model_name="LSTM_Final_model")
    compute_anomaly_threshold(model=model_state, sequence_length=sequence)
    logging.info("Pipeline ended.")

if __name__ == "__main__":
    pipeline()
