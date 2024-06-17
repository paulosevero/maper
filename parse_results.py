# Python libraries
import csv
import pandas as pd


def min_max_norm(x, min, max):
    """Normalizes a given value (x) using the Min-Max Normalization method.

    Args:
        x (any): Value that must be normalized.
        min (any): Minimum value known.
        max (any): Maximum value known.

    Returns:
        (any): Normalized value.
    """
    if min == max:
        return 1
    return (x - min) / (max - min)


def read_csv(csv_file: str) -> list:
    """Reads a CSV file and returns a list of dictionaries containing the values.

    Args:
        csv_file (str): Input CSV file name.

    Returns:
        dataset_file (list): Parsed dataset file.
    """
    # Reading the CSV file
    data_frame = pd.read_csv(csv_file)

    # Converting the read data frame into a list of dictionaries
    dataset_file = data_frame.to_dict(orient="records")

    return dataset_file


# Passing the input CSV file name

dataset_file = "summarized_results_temp_et_al.csv"
# Reading the input CSV file
dataset = read_csv(csv_file=dataset_file)

# Finding minimum and maximum values
minimum = {
    "energy_consumption_kwh": float("inf"),
    "latency_sla_violations": float("inf"),
    "provisioning_time_sla_violations": float("inf"),
}
maximum = {
    "energy_consumption_kwh": float("-inf"),
    "latency_sla_violations": float("-inf"),
    "provisioning_time_sla_violations": float("-inf"),
}
for row in dataset:
    row["energy_consumption_kwh"] = row["sum_overall_power_consumption"] / (row["time_steps"] * 1000)

    for metric in minimum.keys():
        row[metric] = float(row[metric])
        if minimum[metric] > row[metric]:
            minimum[metric] = row[metric]
        if maximum[metric] < row[metric]:
            maximum[metric] = row[metric]

# Calculating normalized values of chosen metrics
print(f"====== {dataset_file} ======")
for row in dataset:
    cost = 0
    for metric in minimum.keys():
        row[f"norm_{metric}"] = min_max_norm(x=row[metric], min=minimum[metric], max=maximum[metric])
        cost += row[f"norm_{metric}"]

    row["norm_cost"] = cost

    print(f"\t{row}")


# Exporting parsed results to a CSV file
with open(f"parsed_{dataset_file}", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=dataset[0].keys())
    writer.writeheader()
    writer.writerows(dataset)
