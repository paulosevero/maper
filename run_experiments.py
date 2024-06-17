"""This script is responsible for orchestrating the batch-execution of algorithms."""

# Importing Python libraries
from subprocess import Popen, DEVNULL, TimeoutExpired
import itertools
from time import sleep
import os


# NUMBER_OF_PARALLEL_PROCESSES = max(1, os.cpu_count() - 2)
NUMBER_OF_PARALLEL_PROCESSES = 12


def run_simulation(
    seed: int,
    input: str,
    time_steps: int,
    algorithm: str,
    delay_threshold: float,
    prov_time_threshold: float,
):
    """Executes the simulation with the specified parameters.

    Args:
        seed (int): Seed value defined to enabling reproducible results.
        input (str): Input dataset file.
        time_steps (int): Number of time steps (seconds) to be simulated.
        algorithm (str): Algorithm being executed.
        delay_threshold (float): Delay threshold used by the resource management algorithm.
        prov_time_threshold (float): Prov. time threshold used by the resource management algorithm
    """
    # Running the simulation based on the parameters and gathering its execution time
    cmd = f"python3 -B -m simulator -s {seed} -i {input} -t {time_steps} -a {algorithm} -d {delay_threshold} -p {prov_time_threshold}"
    return Popen(cmd.split(" "), stdout=DEVNULL, stderr=DEVNULL)


# Parameters
datasets = ["datasets/dataset1.json"]
algorithms = ["temp_et_al"]
time_steps = 3600  # One hour in seconds

# delay_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# prov_time_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
delay_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
prov_time_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

print(f"datasets: {datasets}")
print(f"algorithms: {algorithms}")
print(f"time_steps: {time_steps}")
print(f"delay_thresholds: {delay_thresholds}")
print(f"prov_time_thresholds: {prov_time_thresholds}")
print()

# Generating list of combinations with the parameters specified
combinations = list(
    itertools.product(
        datasets,
        algorithms,
        delay_thresholds,
        prov_time_thresholds,
    )
)

# Executing simulations and collecting results
processes = []

print(f"EXECUTING {len(combinations)} COMBINATIONS")
for i, parameters in enumerate(combinations, 1):
    # Parsing parameters
    dataset = parameters[0]
    algorithm = parameters[1]
    delay_threshold = parameters[2]
    prov_time_threshold = parameters[3]

    print(f"\t[Execution {i}]")
    print(f"\t\t[{algorithm}] dataset={dataset}. delay_threshold={delay_threshold}. prov_time_threshold={prov_time_threshold}")

    # Executing algorithm
    proc = run_simulation(
        seed=1,
        input=dataset,
        time_steps=time_steps,
        algorithm=algorithm,
        delay_threshold=delay_threshold,
        prov_time_threshold=prov_time_threshold,
    )

    sleep(2)

    processes.append(proc)

    while len(processes) >= NUMBER_OF_PARALLEL_PROCESSES:
        for proc in processes:
            try:
                proc.wait(timeout=1)

            except TimeoutExpired:
                pass

            else:
                processes.remove(proc)
                print(f"PID {proc.pid} finished")

    print(f"{len(processes)} processes running in parallel")
