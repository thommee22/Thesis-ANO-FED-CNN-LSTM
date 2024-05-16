# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:43:52 2024

@author: Administrator
"""

import flwr as fl
import numpy as np
from flwr.common import Metrics
from typing import List, Tuple
import matplotlib.pyplot as plt
from logging import INFO
import pickle
from pathlib import Path
from flwr.common.logger import log
from flwr.common import parameters_to_ndarrays


metrics_history = {
    "binary_accuracy": [],
    "precision": [],
    "recall": []
}


def average_metrics(metrics):

    # Here num_examples are not taken into account by using _
    accuracies = np.mean([metric["binary_accuracy"] for _, metric in metrics])
    recalls = np.mean([metric["recall"] for _, metric in metrics])
    precisions = np.mean([metric["precision"] for _, metric in metrics])


    metrics_history["binary_accuracy"].append(accuracies)
    metrics_history["precision"].append(precisions)
    metrics_history["recall"].append(recalls)

    return {
        "binary_accuracy": accuracies,
        "recall": recalls,
        "precision": precisions,
    }


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_sum = { "binary_accuracy": 0, "precision": 0, "recall": 0 }

    for num_examples, m in metrics:
        weight = num_examples / total_examples
        weighted_sum["binary_accuracy"] += m["binary_accuracy"] * weight
        weighted_sum["precision"] += m["precision"] * weight
        weighted_sum["recall"] += m["recall"] * weight

    metrics_history["binary_accuracy"].append(weighted_sum["binary_accuracy"])
    metrics_history["precision"].append(weighted_sum["precision"])
    metrics_history["recall"].append(weighted_sum["recall"])

    return weighted_sum

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    return {}

def plot_metrics(metrics_history: dict):
    """Plot the metrics history."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(metrics_history['binary_accuracy'], label='Binary Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(metrics_history['precision'], label='Precision')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(metrics_history['recall'], label='Recall')
    plt.legend()

    plt.savefig('metrics_plot.png')  # Save the plot to a file



class FedAvgWithModelSaving(fl.server.strategy.FedAvg):
    """This is a custom strategy that behaves exactly like
    FedAvg with the difference of storing of the state of
    the global model to disk after each round.
    """
    def __init__(self, save_path: str, *args, **kwargs):
        self.save_path = Path(save_path)
        # Create directory if needed
        self.save_path.mkdir(exist_ok=True, parents=True)
        super().__init__(*args, **kwargs)

    def _save_global_model(self, server_round: int, parameters):
        """A new method to save the parameters to disk."""

        # convert parameters to list of NumPy arrays
        # this will make things easy if you want to load them into a
        # PyTorch or TensorFlow model later
        ndarrays = parameters_to_ndarrays(parameters)
        data = {'globa_parameters': ndarrays}
        filename = str(self.save_path/f"parameters_round_{server_round}.pkl")
        with open(filename, 'wb') as h:
            pickle.dump(data, h, protocol=pickle.HIGHEST_PROTOCOL)
        log(INFO, f"Checkpoint saved to: {filename}")    

    def evaluate(self, server_round: int, parameters):
        """Evaluate model parameters using an evaluation function."""
        # save the parameters to disk using a custom method
        self._save_global_model(server_round, parameters)

        # call the parent method so evaluation is performed as
        # FedAvg normally does.
        return super().evaluate(server_round, parameters)

class MyStrategy3(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round = 0

    def aggregate_fit(self, rnd, results, failures, config):
        # Perform the usual FedAvg fitting
        weights = super().aggregate_fit(rnd, results, failures, config)

        # After fitting, save the global model
        self.round += 1
        global_model = self.aggregator.aggregated()
        global_model.save(f'global_model_round_{self.round}.h5')

        # Return the aggregated weights
        return weights


class MyStrategy2(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round = 0

    def aggregate_fit(self, rnd, results, failures, config):
        # Perform the usual FedAvg fitting
        super().aggregate_fit(rnd, results, failures, config)

        # After fitting, save the global model
        self.round += 1
        global_model = self.aggregator.aggregated()
        global_model.save(f'global_model_round_{self.round}.h5')





class MyStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round = 0

    def fit(self, rnd, parameters, config):
        # Perform the usual FedAvg fitting
        results, failures = super().fit(rnd, parameters, config)

        # After fitting, save the global model
        self.round += 1
        global_model = load_model_from_parameters(results[0])
        global_model.save(f'global_model_round_{self.round}.h5')

        return results, failures

if __name__ == "__main__":

    strategy = FedAvgWithModelSaving(
             save_path='my_checkpoints',
             fraction_fit =1,
             fraction_evaluate=1,
             min_fit_clients = 3,
             min_evaluate_clients=3,
             min_available_clients=3,
             on_evaluate_config_fn=evaluate_config,
             evaluate_metrics_aggregation_fn=weighted_average
    )

    # Start Flower server with the strategy
    fl.server.start_server(
        server_address="0.0.0.0:8080", 
        config=fl.server.ServerConfig(num_rounds=20),  # Define number of training rounds
        strategy=strategy
    )

    plot_metrics(metrics_history)
