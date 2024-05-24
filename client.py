import flwr as fl
import pandas as pd
import numpy as np
import tensorflow as tf

# Load training features and labels from an Excel file
def load_data_from_excel(file_path):
    df = pd.read_excel(file_path)
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    return features, labels

# Create an RNN model
def create_rnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.SimpleRNN(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, excel_file):
        self.x_train, self.y_train = load_data_from_excel(excel_file)
        input_shape = (self.x_train.shape[1], 1)
        self.x_train = self.x_train.reshape(-1, input_shape[0], input_shape[1])
        self.model = create_rnn_model(input_shape)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_train, self.y_train)
        return loss, len(self.x_train), {"accuracy": accuracy}

def start_flower_client(excel_file):
    client = FlowerClient(excel_file)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python client.py <excel_file>")
    else:
        excel_file = sys.argv[1]
        start_flower_client(excel_file)
