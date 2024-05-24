


import flwr as fl

def start_flower_server():
    # Define the federated averaging strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Fraction of clients used during training
        min_fit_clients=3,  # Minimum number of clients to be used during training
        min_available_clients=3,  # Minimum number of clients that need to be connected to the server
    )

    # Start the Flower server
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),  # Number of federated learning rounds
        strategy=strategy,
    )

if __name__ == "__main__":
    start_flower_server()
