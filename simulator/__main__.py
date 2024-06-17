"""This file contains the main executable file within the project."""

# Importing Python libraries
from random import seed
import time
import argparse

# Importing EdgeSimPy components
from edge_sim_py import *

# Importing helper functions
from simulator.helper_functions import *
from simulator.edgesimpy_extensions import *

# Importing Resource management policies
from simulator.algorithms import *


def load_edgesimpy_extensions():
    """Loads EdgeSimPy extensions"""
    # Overriding metric collection methods
    EdgeServer.collect = edge_server_collect
    Service.collect = service_collect
    ContainerRegistry.collect = container_registry_collect
    Simulator.monitor = simulator_monitor
    Simulator.consolidate_metrics = simulator_consolidate_metrics
    Simulator.consolidate_edge_server_metrics = simulator_consolidate_edge_server_metrics
    Simulator.consolidate_container_registry_metrics = simulator_consolidate_container_registry_metrics
    Simulator.consolidate_service_metrics = simulator_consolidate_service_metrics

    # Overriding entity update methods
    NetworkFlow.step = network_flow_step
    Service.step = service_step
    EdgeServer.step = edge_server_step
    ContainerRegistry.step = container_registry_step
    ContainerRegistry.deprovision = container_registry_deprovision

    # Creating entity attributes
    for container_registry in ContainerRegistry.all():
        container_registry.provisioning_initial_time_step = 0
        container_registry.provisioning_final_time_step = 0
        container_registry.provisioning_time = 0
        container_registry.first_usage_time_step = None
    for service in Service.all():
        service.provisioning_time_sla_violations = 0


def main(parameters: dict):
    # Defining a seed value to enable reproducible results
    seed(parameters["seed_value"])

    # Creating a Simulator object
    simulator = Simulator(
        tick_duration=1,
        tick_unit="seconds",
        stopping_criterion=lambda model: model.schedule.steps == parameters["time_steps"],
        resource_management_algorithm=eval(parameters["algorithm"]),
        resource_management_algorithm_parameters=parameters,
        dump_interval=float("inf"),
        logs_directory=f"logs",
    )

    # Parsing simulation parameters
    parameters_string = f"timestamp={int(time.time())};dataset={parameters['dataset'].split('datasets/')[1].split('.json')[0]};"
    for key, value in parameters.items():
        if key != "dataset":
            parameter_divisor = "" if key == list(parameters.keys())[-1] else ";"
            parameters_string += f"{key}={value}{parameter_divisor}"

    simulator.output_file_name = parameters_string
    simulator.overall_metrics = {}
    simulator.per_step_metrics = []

    # Initializing the simulated scenario
    simulator.initialize(input_file=parameters["dataset"])

    # Applying EdgeSimPy extensions
    load_edgesimpy_extensions()

    for container_image in ContainerImage.all():
        for index, layer_digest in enumerate(container_image.layers_digests, 1):
            container_layer = ContainerLayer.find_by("digest", layer_digest)
            container_layer.instruction = f"{container_image.name}_layer_{index}"

    print("======================")
    print("====== SCENARIO ======")
    print("======================")
    print(f"Edge Servers: {EdgeServer.count()}")
    print(f"Applications: {Application.count()}")
    print(f"Services: {Service.count()}")
    print(f"Users: {User.count()}")

    # Starting the simulation's execution time counter
    start_time = time.time()

    # Executing the simulation
    simulator.run_model()

    # Finishing the simulation's execution time counter
    final_time = time.time()

    # Consolidating simulation parameters
    simulator.resource_management_algorithm_parameters["elapsed_time"] = final_time - start_time
    simulator.consolidate_metrics(export_file=True, hide_detailed_metrics=False)


if __name__ == "__main__":
    # Parsing named arguments from the command line
    parser = argparse.ArgumentParser()

    # Generic arguments
    parser.add_argument("--seed", "-s", help="Seed value for EdgeSimPy", default="1")
    parser.add_argument("--input", "-i", help="Input dataset file", default="datasets/dataset1.json")
    parser.add_argument("--algorithm", "-a", help="Algorithm that will be executed", required=True)
    parser.add_argument("--time-steps", "-t", help="Number of time steps (seconds) to be simulated", required=True)
    parser.add_argument("--delay-threshold", "-d", help="Delay threshold used by the resource management algorithm", default="1")
    parser.add_argument("--prov-time-threshold", "-p", help="Prov. time threshold used by the resource management algorithm", default="1")

    args = parser.parse_args()

    parameters = {
        "seed_value": int(args.seed),
        "dataset": args.input,
        "algorithm": args.algorithm,
        "time_steps": int(args.time_steps),
        "delay_threshold": float(args.delay_threshold),
        "prov_time_threshold": float(args.prov_time_threshold),
    }

    main(parameters=parameters)
