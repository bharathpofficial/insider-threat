from   tensorflow.keras.models import Model #type: ignore
from   tensorflow.keras.layers import Input, Dense #type: ignore
import tensorflow as tf
import os
import logging
import numpy as np
import pandas as pd

BATCH_SIZE   = 2048
INPUT_DIM    = 12
ENCODING_DIM = 8

DATASET_PATH         = "../datasets/processed10percent"
BASE_FOLDER          = "./"
PROCESSED_DATA_PATH  = os.path.join(BASE_FOLDER,"processed_combined_6_0")
MODEL_PATH           = os.path.join(PROCESSED_DATA_PATH, "saved_models")
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(BASE_FOLDER,"autoencoder_model_training_classic.log"), filemode='a')

path_list = [PROCESSED_DATA_PATH, MODEL_PATH]
for path in path_list:
    if not os.path.exists(path=path):
        os.makedirs(path)
        logging.info(f'Created Folder {path.split("/")[-1]} at {path}')

logging.info(f"""
                Initiated Project Successfully  
                DATASET_PATH        : {DATASET_PATH},
                PROCESSED_DATA_PATH : {PROCESSED_DATA_PATH},
                MODEL_PATH          : {MODEL_PATH},
                BASE_FOLDER         : {BASE_FOLDER}
                """)
tf.config.run_functions_eagerly(True)  # THIS LINE I ADDED, THAT SOLVED MY ISSUE ...
def create_autoencoder(input_dimension, encoding_dim):
    logging.info("Control was at create_autoencoder Function")

    GATE = 64 
    input_layer = Input(shape=(input_dimension,))

    # Add batch normalization and dropout for better training
    encoded = Dense(GATE, activation='relu')(input_layer)
    encoded = tf.keras.layers.BatchNormalization()(encoded)
    encoded = tf.keras.layers.Dropout(0.2)(encoded)

    encoded = Dense(GATE // 2, activation='relu')(encoded)
    encoded = Dense(GATE // 4, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    decoded = Dense(GATE // 4, activation='relu')(encoded)
    decoded = Dense(GATE // 2, activation='relu')(decoded)
    decoded = Dense(GATE, activation='relu')(decoded)
    decoded = Dense(input_dimension, activation='sigmoid')(decoded)

    autoencoder      = Model(input_layer, decoded)

    logging.info(f"""autoencoder Model:
                 Hyperparameters    : {GATE}, {GATE // 2}, {GATE // 4} 
                 Input Layer        : {input_layer.shape},
                 Input Dimension    : {input_dimension},
                 Encoding Dimension : {encoding_dim}
                 """)
    logging.info(f""" ,
                 {autoencoder.summary()}
                  """)
    
    # Use a different optimizer with a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

    logging.info(f"Autoencoder Model compiled Sucessfully!")
    return autoencoder

def train_model_on_batch(model, processed_data):
    logging.info("Model Training On Batch Function ...")
    dataset     = tf.data.Dataset.from_tensor_slices(processed_data).batch(BATCH_SIZE)

    epoch_loss  = []
    for mini_batch in dataset:
        loss    = model.train_on_batch(mini_batch, mini_batch)
        epoch_loss.append(loss)
    
    avg_loss    = np.mean(epoch_loss)
    logging.info(f"Average loss for processing file :  {avg_loss}")
    return model, avg_loss

def preprocess_batch_data(batch_data):
    # Drop date column and hash user column
    batch_data          = batch_data.drop('date', axis=1)
    # batch_data['user']  = batch_data['user'].apply(hash_categorical)
    
    # # Convert to numpy and normalize
    # bool_columns        = batch_data.select_dtypes(inlcude=['bool']).colums
    # for col in bool_columns:
    #     batch_data[col] = batch_data[col].astype(int)

    data                = batch_data.values.astype('float32')
    # data                = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0) + 1e-10)
    return data

def train_model_incrementally(model, batch_files, val_file, test_file, epochs=31):
    all_losses      = []
    checkpoint_path = os.path.join(MODEL_PATH, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    for epoch in range(epochs):
        epoch_losses    = []
        logging.info(f"Epoch {epoch + 1} / {epochs}")

        for index, file in enumerate(batch_files, start=1):
            # Save current model state
            temp_model_path = os.path.join("checkpoints", f"temp_model_epoch{epoch}_batch{index}")

            # model.save(temp_model_path)
            save_model(model=model, model_name=temp_model_path)

            # Clear memory
            tf.keras.backend.clear_session()

            # Load fresh model instance
            model = tf.keras.models.load_model(os.path.join("./processed_combined_6_0/saved_models", temp_model_path) + ".keras")

            batch_data         = pd.read_csv(file)
            logging.info(f"Processing file: {file}")
            
            # Preprocessing The Batch
            preprocessed_data  = preprocess_batch_data(batch_data)
            logging.info(f"Shape Of the Dataset   : {preprocessed_data.shape}")
            # logging.info(f"Summary Of the Dataset : \n{preprocessed_data.view()}")
            model, batch_loss  = train_model_on_batch(model, preprocessed_data)
            epoch_losses.append(batch_loss)

            # Clean up temporary file
            if os.path.exists(os.path.join(MODEL_PATH, temp_model_path) + ".keras"):
                os.remove(os.path.join(MODEL_PATH, temp_model_path) + ".keras")

        save_model(model,"AutoEncoder_Intermediate_trained")
        logging.info(f"Evaluating model after epoch {epoch + 1}")
        mse, anomalies = evaluate_model(model=model, val_dataset_for_threshold=val_file, test_dataset=test_file)
        all_losses.append(np.mean(epoch_losses))
        logging.info(f"Epoch {epoch + 1} average loss: {all_losses[-1]}\n")

    save_model(model,"AutoEncoder_Trained_Final_Model")
    return model, all_losses

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

def compute_anomaly_threshold(model, val_dataset):
    
    # Compute threshold as before
    val_data = pd.read_csv(val_dataset)
    val_data = preprocess_batch_data(val_data)
    
    predictions = model.predict(val_data)
    mse = np.mean(np.power(val_data - predictions, 2), axis=1)
    threshold = np.mean(mse) + 2 * np.std(mse)

    logging.info("Computed and cached new threshold value")
    logging.info(f"""Anomaly Threshold Computation:
                Mean MSE  : {np.mean(mse)}
                Std MSE   : {np.std(mse)}
                Threshold : {threshold}""")
    return threshold

# Add this after computing threshold in compute_anomaly_threshold function
def save_threshold(threshold, path):
    np.save(os.path.join(path, 'anomaly_threshold.npy'), threshold)
    logging.info(f"Saved threshold value: {threshold}")


def evaluate_model(model, test_dataset, val_dataset_for_threshold):
    # Compute threshold using validation data
    threshold = compute_anomaly_threshold(model, val_dataset_for_threshold)
    
    # In compute_anomaly_threshold function, add:
    save_threshold(threshold, MODEL_PATH)

    # Load and preprocess test data
    test_data = pd.read_csv(test_dataset)
    test_data = preprocess_batch_data(test_data)
    
    # Get reconstruction errors
    predictions = model.predict(test_data)
    mse = np.mean(np.power(test_data - predictions, 2), axis=1)
    
    # Identify anomalies
    anomalies = mse > threshold
    
    logging.info(f"""Model Evaluation:
                 Average MSE         : {np.mean(mse)}
                 Number of Anomalies : {np.sum(anomalies)}
                 Anomaly Percentage  : {(np.sum(anomalies)/len(anomalies))*100}%""")
    
    return mse, anomalies

if __name__ == "__main__":
    logging.info("STARTED AUTOENCODER TRAINING")
    
    autoencoder_model = create_autoencoder(INPUT_DIM, ENCODING_DIM)

    train_batch_files = [f"{DATASET_PATH}/merged_dataset_batch_{i}.csv" for i in range(1, 9)]
    val_batch_file    = f"{DATASET_PATH}/merged_dataset_batch_9.csv"
    test_batch_file   = f"{DATASET_PATH}/merged_dataset_batch_10.csv"

    model ,all_losses = train_model_incrementally(model=autoencoder_model,
                                      batch_files=train_batch_files,
                                      val_file=val_batch_file,
                                      test_file=test_batch_file)
    logging.info("ENDED AUTOENCODER TRAINING")
