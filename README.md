# MAPER: Mobility-Aware Power-Efficient Container Registry Migrations for Edge Computing Infrastructures

This repository presents MAPER, a container management strategy designed to migrate container registries in edge computing infrastructures while optimizing both the performance of containerized applications and power consumption.

## Table of Contents
- [Motivation](#motivation)
- [Repository Structure](#repository-structure)
- [Installation Guide](#installation-guide)
- [Usage Guide](#usage-guide)
- [Manuscript](#manuscript)

  
## Motivation: 

Containerization is a valuable strategy for Edge Computing infrastructures due to its small footprint and low overhead. However, its performance is significantly impacted by the effective management of container registries, as unoptimized registry placement can lead to increased application provisioning times and resource waste. While previous studies have proposed several registry positioning strategies to reduce application provisioning times, none of them have considered the impact of registry management decisions on the energy efficiency of edge infrastructure. To address this gap, MAPER dynamically relocates container registries in response to user mobility patterns, optimizing both application provisioning times and power consumption.

## Repository Structure

Within the repository, you'll find the following directories and files, logically grouping common assets used to simulate container registries migrations at the edge. You'll see something like this:

```
├── pyproject.toml
├── container_image_analysis.py
├── create_dataset.py
├── parse_results.py
├── percentile.py 
├── run_experiments.py
├── datasets/
└── simulator
    ├── __main__.py
    ├── edgesimpy_extensions.py
    ├── helper_functions.py
    └── algorithms
        ├── follow_user.py
        ├── maper.py
        ├── never_follow.py
        └── temp_et_al.py
```

In the root directory, the ```pyproject.toml``` file organizes all project dependencies, including the minimum required version of the Python language. This file guides the execution of Poetry, a Python library that installs the dependencies securely, avoiding conflicts with external packages.

 > Modifications made to the pyproject.toml file are automatically inserted into poetry.lock whenever Poetry is called.

The ```container_image_analysis.py``` and ```create_dataset.py``` files comprise the source code used for creating datasets. Created dataset files are stored in the datasets directory.

The ```parse_results.py``` file contains the code used to parse and compute the results obtained.

The ```percentile.py``` file calculates the 90th percentile of migration time values for the strategies used in the simulation.

The ```run_experiments.py``` file makes it easy to execute the implemented strategies. For instance, with a few instructions, we can conduct a complete sensitivity analysis of the algorithms using different sets of parameters.

The ```datasets``` directory contains JSON files describing the scenario and components that will be simulated during the experiments.

Finally, the ```simulator``` directory contains the ```algorithms``` subdirectory, which accommodates the source code for the strategies used in the simulator. It also contains the ```edgesimpy_extensions.py``` and the ```helper_functions.py``` files, which host methods that extend the standard functionality of the simulated components.

## Installation Guide

This section contains information about the prerequisites of the system and about how to configure the environment to execute the simulations.

Project dependencies are available for Linux, Windows, and macOS. However, we highly recommend using a recent version of a Debian-based Linux distribution. The installation below was validated on **Ubuntu 24.04 LTS**.

The first step needed to run the simulation is installing Python 3. We can do that by executing the following command:

``` 
sudo apt install python3 -y
```

We use a Python library called Poetry to manage project dependencies. In addition to selecting and downloading proper versions of project dependencies, Poetry automatically provisions virtual environments for the simulator, avoiding problems with external dependencies. On Linux and macOS, we can install Poetry with the following command:

```
curl -sSL https://install.python-poetry.org | python3 -
```

The command above installs Poetry executable inside Poetry’s bin directory. On Unix, it is located at ```$HOME/.local/bin```. We can get more information about Poetry installation from their [documentation page](https://python-poetry.org/docs/#installation).

Considering that we already downloaded the repository, we first need to install dependencies using Poetry. To do so, we access the command line in the root directory and type the following command:

```
poetry shell
```

The command we just ran creates a virtual Python environment that we will use to run the simulator. Notice that Poetry automatically sends us to the newly created virtual environment. Next, we need to install the project dependencies using the following command:

```
peotry install
```

After a few moments, Poetry will have installed all the dependencies needed by the simulator and we will be ready to run the experiments.

## Usage Guide

This section contains instructions about the execution of the simulator. Please notice that the commands below need to be run inside the virtual environment created by Poetry after the project's dependencies have been successfully installed.

A description of the custom parameters utilized in the simulation is given below:

- ```--seed```: Seed value for EdgeSimPy.
- ```--input```: Input dataset file.
- ```--algorithm```: Algorithm that will be executed.
- ```--time-steps```: Number of time steps (seconds) to be simulated.
- ```--delay-threshold```: Delay threshold used by the resource management algorithm.
- ```--prov-time-threshold```: Provisioning time threshold used by the resource management algorithm.

Below is an example of the command used to run the simulation:

```bash
python -B -m simulator --seed 1 --input "datasets/dataset1.json" --algorithm "maper" --time-steps 3600 --delay-threshold 0.9 --prov-time-threshold 1
```

If you want to run the experiments comparing MAPER and baseline strategies from the literature, you can do so by executing the ```run_experiments.py``` file using the command below:

```bash
python run_experiments.py
```

## Manuscript
TBD
