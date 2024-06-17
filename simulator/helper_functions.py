"""This file contains a set of helper functions that facilitate the simulation execution."""
# Importing Python libraries
from random import choice, shuffle, sample
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import operator
import csv
import os

# Importing EdgeSimPy components
from edge_sim_py import *


def display_topology(topology: object, output_filename: str = "topology"):
    """Prints the network topology to an output file.

    Args:
        topology (object): Topology object.
        output_filename (str, optional): Output file name. Defaults to "topology".
    """
    # Customizing visual representation of topology
    positions = {}
    labels = {}
    colors = []
    sizes = []

    for node in topology.nodes():
        positions[node] = node.coordinates
        labels[node] = node.id
        node_size = (
            500
            if any(user.coordinates == node.coordinates for user in User.all())
            else 100
        )
        sizes.append(node_size)

        if (
            len(node.base_station.edge_servers)
            and sum(len(s.container_registries) for s in node.base_station.edge_servers)
            == 0
        ):
            node_server = node.base_station.edge_servers[0]
            if node_server.model_name == "PowerEdge R620":
                colors.append("green")
            elif node_server.model_name == "SGI":
                colors.append("red")

        elif (
            len(node.base_station.edge_servers)
            and sum(len(s.container_registries) for s in node.base_station.edge_servers)
            > 0
        ):
            colors.append("blue")
        else:
            colors.append("black")

    # Configuring drawing scheme
    nx.draw(
        topology,
        pos=positions,
        node_color=colors,
        node_size=sizes,
        labels=labels,
        font_size=6,
        font_weight="bold",
        font_color="whitesmoke",
    )

    # Saving a topology image in the disk
    plt.savefig(f"{output_filename}.png", dpi=120)


def uniform(
    n_items: int, valid_values: list, shuffle_distribution: bool = True
) -> list:
    """Creates a list of size "n_items" with values from "valid_values" according to the uniform distribution.
    By default, the method shuffles the created list to avoid unbalanced spread of the distribution.

    Args:
        n_items (int): Number of items that will be created.
        valid_values (list): List of valid values for the list of values.
        shuffle_distribution (bool, optional): Defines whether the distribution is shuffled or not. Defaults to True.

    Raises:
        Exception: Invalid "valid_values" argument.

    Returns:
        uniform_distribution (list): List of values arranged according to the uniform distribution.
    """
    if (
        not isinstance(valid_values, list)
        or isinstance(valid_values, list)
        and len(valid_values) == 0
    ):
        raise Exception(
            "You must inform a list of valid values within the 'valid_values' attribute."
        )

    # Number of occurrences that will be created of each item in the "valid_values" list
    distribution = [
        int(n_items / len(valid_values)) for _ in range(0, len(valid_values))
    ]

    # List with size "n_items" that will be populated with "valid_values" according to the uniform distribution
    uniform_distribution = []

    for i, value in enumerate(valid_values):
        for _ in range(0, int(distribution[i])):
            uniform_distribution.append(value)

    # Computing leftover randomly to avoid disturbing the distribution
    leftover = n_items % len(valid_values)
    for i in range(leftover):
        random_valid_value = choice(valid_values)
        uniform_distribution.append(random_valid_value)

    # Shuffling distribution values in case 'shuffle_distribution' parameter is True
    if shuffle_distribution:
        shuffle(uniform_distribution)

    return uniform_distribution


def prettify_dictionary_for_dump(dictionary: dict, list_size_threshold: int = 10):
    """Helper serialization function that modifies an input dictionary to ensure that json.dump() will produce
    pretty outcomes when printing it. Specifically, the function modifies is to keep the indentation for
    nested dictionaries while displaying list elements inline.

    Args:
        dictionary (dict): Dictionary to be customized.
        list_size_threshold (int, optional): Maximum list size allowed without converting to string. Defaults to 10.

    Returns:
        dictionary (dict): Customized dictionary.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            prettify_dictionary_for_dump(value, list_size_threshold)
        elif isinstance(value, list) and len(value) > list_size_threshold:
            dictionary[key] = str(value)
    return dictionary


def save_dict_to_excel(dictionary: dict, output_filename: str):
    """Saves the collected metrics to an external spreadsheet file.

    Args:
        dictionary (dict): Dictionary containing the collected metrics.
        output_filename (str): Output file name.
    """
    # Creating the "logs" directory if it doesn't exists
    if not os.path.exists("logs/"):
        os.makedirs("logs")

    with pd.ExcelWriter(f"logs/{output_filename}.xlsx", engine="openpyxl") as writer:
        for key, value in dictionary.items():
            if isinstance(value, list):
                # Structure 1: List of dictionaries
                data_frame = pd.DataFrame(value)
            elif isinstance(value, dict):
                # Structure 2: Sub-dictionary
                data_frame = pd.DataFrame([value])
            else:
                continue

            data_frame.to_excel(writer, sheet_name=key, index=False)


def append_dict_to_csv(dictionary: dict, output_filename: str):
    """Appends the content of a dictionary to a specified output CSV file.

    Args:
        dictionary (dict): Dictionary whose content must be appended to the output CSV file.
        output_filename (str): Output CSV file name.
    """
    # Checking if the file already exists
    file_exists = os.path.isfile(f"{output_filename}.csv")

    # Opening the file in append mode ('a')
    with open(f"{output_filename}.csv", "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=dictionary.keys())

        # Writing the spreadsheet header if the file is new
        if not file_exists:
            writer.writeheader()

        # Appending the dictionary data as a new row within the spreadsheet
        writer.writerow(dictionary)


def display_system_state_at_current_step(current_step):
    registries_on_r620_hosts = len(
        [
            reg
            for reg in ContainerRegistry.all()
            if reg.server and reg.server.model_name == "PowerEdge R620"
        ]
    )
    registries_on_sgi_hosts = len(
        [
            reg
            for reg in ContainerRegistry.all()
            if reg.server and reg.server.model_name == "SGI"
        ]
    )

    services_on_r620_hosts = len(
        [
            service
            for service in Service.all()
            if service.server and service.server.model_name == "PowerEdge R620"
        ]
    )
    services_on_sgi_hosts = len(
        [
            service
            for service in Service.all()
            if service.server and service.server.model_name == "SGI"
        ]
    )

    pw_r620_hosts = sum(
        s.get_power_consumption()
        for s in EdgeServer.all()
        if s.model_name == "PowerEdge R620"
    )
    pw_sgi_hosts = sum(
        s.get_power_consumption() for s in EdgeServer.all() if s.model_name == "SGI"
    )
    overall_pw = pw_r620_hosts + pw_sgi_hosts

    current_step_metadata = {
        "Time Step": current_step,
        "Registries": f"{ContainerRegistry.count()} (R620: {registries_on_r620_hosts}. SGI: {registries_on_sgi_hosts})",
        "Services per Server Model": {
            "R620": services_on_r620_hosts,
            "SGI": services_on_sgi_hosts,
        },
        "Power Consumption": f"{overall_pw} (R620: {pw_r620_hosts}. SGI: {pw_sgi_hosts})",
    }
    print("\n")
    print(f"====== {current_step_metadata} ======")


def randomized_closest_fit():
    """Encapsulates a randomized closest-fit service placement algorithm."""
    services = sample(Service.all(), Service.count())
    for service in services:
        app = service.application
        user = app.users[0]
        user_switch = user.base_station.network_switch

        edge_servers = []
        for edge_server in EdgeServer.all():
            path = nx.shortest_path(
                G=Topology.first(),
                source=user_switch,
                target=edge_server.network_switch,
                weight="delay",
            )
            delay = Topology.first().calculate_path_delay(path=path)
            edge_servers.append(
                {
                    "object": edge_server,
                    "path": path,
                    "delay": delay,
                    "violates_sla": delay
                    > user.delay_slas[str(service.application.id)],
                    "free_capacity": get_normalized_capacity(object=edge_server)
                    - get_normalized_demand(object=edge_server),
                }
            )

        edge_servers = sorted(
            edge_servers, key=lambda s: (s["violates_sla"], -s["free_capacity"])
        )

        for edge_server_metadata in edge_servers:
            edge_server = edge_server_metadata["object"]

            # Checking if the host would have resources to host the service and its (additional) layers
            if edge_server.has_capacity_to_host(service=service):
                # Updating the host's resource usage
                edge_server.cpu_demand += service.cpu_demand
                edge_server.memory_demand += service.memory_demand

                # Creating relationship between the host and the registry
                service.server = edge_server
                edge_server.services.append(service)

                for layer_metadata in edge_server._get_uncached_layers(service=service):
                    layer = ContainerLayer(
                        digest=layer_metadata.digest,
                        size=layer_metadata.size,
                        instruction=layer_metadata.instruction,
                    )

                    # Updating host's resource usage based on the layer size
                    edge_server.disk_demand += layer.size

                    # Creating relationship between the host and the layer
                    layer.server = edge_server
                    edge_server.container_layers.append(layer)

                break

        # Creating an instance of the service image on its host if necessary
        if not any(
            hosted_image
            for hosted_image in service.server.container_images
            if hosted_image.digest == service.image_digest
        ):
            template_image = next(
                (
                    img
                    for img in ContainerImage.all()
                    if img.digest == service.image_digest
                ),
                None,
            )

            # Creating a ContainerImage object to represent the new image
            image = ContainerImage()
            image.name = template_image.name
            image.digest = template_image.digest
            image.tag = template_image.tag
            image.layers_digests = template_image.layers_digests

            # Connecting the new image to the target host
            image.server = service.server
            service.server.container_images.append(image)


def calculate_bandwidth(
    tick_duration: float, topology: object, path: list, minimum_link_bandwidth: float
) -> float:
    """Calculates the actual bandwidth available for transferring a data chunk based on
    the bandwidth available in the links and the network path's transmission delay.

    Args:
        tick_duration (float): Simulation's tick duration (seconds).
        topology (object): Network topology.
        path (list): Network path to be analyzed.
        minimum_link_bandwidth (float): Minimum link bandwidth available for transferring the data chunk.

    Returns:
        actual_bandwidth (float): Calculated bandwidth available for transferring the data chunk.
    """
    overall_transmission_delay = 0
    for i in range(0, len(path) - 1):
        link = topology[path[i]][path[i + 1]]
        overall_transmission_delay += link["transmission_delay"]

    actual_bandwidth = (
        tick_duration / (tick_duration + overall_transmission_delay)
    ) * minimum_link_bandwidth
    return actual_bandwidth


def min_max_norm(x, minimum, maximum):
    """Normalizes a given value (x) using the Min-Max Normalization method.

    Args:
        x (any): Value that must be normalized.
        min (any): Minimum value known.
        max (any): Maximum value known.

    Returns:
        (any): Normalized value.
    """
    if minimum == maximum:
        return 1
    return (x - minimum) / (maximum - minimum)


def find_minimum_and_maximum(metadata: list):
    """Finds the minimum and maximum values of a list of dictionaries.

    Args:
        metadata (list): List of dictionaries that contains the analyzed metadata.

    Returns:
        min_and_max (dict): Dictionary that contains the minimum and maximum values of the attributes.
    """
    min_and_max = {
        "minimum": {},
        "maximum": {},
    }

    for metadata_item in metadata:
        for attr_name, attr_value in metadata_item.items():
            if attr_name != "object" and type(attr_value) != list:
                # Updating the attribute's minimum value
                if (
                    attr_name not in min_and_max["minimum"]
                    or attr_name in min_and_max["minimum"]
                    and attr_value < min_and_max["minimum"][attr_name]
                ):
                    min_and_max["minimum"][attr_name] = attr_value

                # Updating the attribute's maximum value
                if (
                    attr_name not in min_and_max["maximum"]
                    or attr_name in min_and_max["maximum"]
                    and attr_value > min_and_max["maximum"][attr_name]
                ):
                    min_and_max["maximum"][attr_name] = attr_value

    return min_and_max


def get_norm(metadata: dict, attr_name: str, min: dict, max: dict) -> float:
    """Wrapper to normalize a value using the Min-Max Normalization method.

    Args:
        metadata (dict): Dictionary that contains the metadata of the object whose values are being normalized.
        attr_name (str): Name of the attribute that must be normalized.
        min (dict): Dictionary that contains the minimum values of the attributes.
        max (dict): Dictionary that contains the maximum values of the attributes.

    Returns:
        normalized_value (float): Normalized value.
    """
    normalized_value = min_max_norm(
        x=metadata[attr_name], minimum=min[attr_name], maximum=max[attr_name]
    )
    return normalized_value


def get_normalized_capacity(object: object) -> float:
    """Returns the normalized capacity of a given entity.

    Args:
        object (object): Entity object to be analyzed.

    Returns:
        (float): Normalized capacity of the given entity.
    """
    return (object.cpu * object.memory * object.disk) ** (1 / 3)


def get_normalized_demand(object: object) -> float:
    """Returns the normalized demand of a given entity.

    Args:
        object (object): Entity object to be analyzed.

    Returns:
        (float): Normalized demand of the given entity.
    """
    if hasattr(object, "disk_demand"):
        return (object.cpu_demand * object.memory_demand * object.disk_demand) ** (
            1 / 3
        )
    else:
        return (object.cpu_demand * object.memory_demand) ** (1 / 2)


def get_shortest_path(origin_switch: object, target_switch: object) -> list:
    """Gets the shortest path between two network nodes (i.e., network switches).

    Args:
        origin_switch (object): Origin network switch.
        target_switch (object): Target network switch.

    Returns:
        shortest_path (list): Shortest network path found.
    """
    topology = origin_switch.model.topology

    if not hasattr(topology, "shortest_paths"):
        topology.shortest_paths = {}

    if frozenset([origin_switch.id, target_switch.id]) in topology.shortest_paths:
        return topology.shortest_paths[frozenset([origin_switch.id, target_switch.id])]
    else:
        shortest_path = nx.shortest_path(
            G=topology,
            source=origin_switch,
            target=target_switch,
            weight="delay",
        )
        topology.shortest_paths[
            frozenset([origin_switch.id, target_switch.id])
        ] = shortest_path
        return topology.shortest_paths[frozenset([origin_switch.id, target_switch.id])]


def get_delay(wireless_delay: int, origin_switch: object, target_switch: object) -> int:
    """Gets the distance (in terms of delay) between two elements (origin and target).

    Args:
        wireless_delay (int): Wireless delay that must be included in the delay calculation.
        origin_switch (object): Origin switch.
        target_switch (object): Target switch.

    Returns:
        delay (int): Delay between the origin and target switches.
    """
    topology = origin_switch.model.topology

    path = get_shortest_path(origin_switch=origin_switch, target_switch=target_switch)
    delay = wireless_delay + topology.calculate_path_delay(path=path)

    return delay


def get_users_with_provisioning_time_issues(
    provisioning_time_threshold: float = 1.0, threshold_logic: str = "reach"
) -> list:
    """Gets the list of users whose last provisioning time exceeds or reaches the given threshold. It is important to
    notice that this function's logic only works for users accessing applications with a single service. Further logic
    is necessary to ensure that the code works with composite applications (applications comprised by multiple services).

    Args:
        provisioning_time_threshold (float, optional): Provisioning time threshold. Defaults to 0.99.
        threshold_logic (str, optional): Provisioning time logic. Valid options: "reach" OR "exceed".

    Returns:
        users_with_provisioning_time_issues = (list): Users with provisioning time issues.
    """
    users_with_provisioning_time_issues = []

    threshold_operation = operator.ge if threshold_logic == "reach" else operator.gt

    for user in User.all():
        for app in user.applications:
            provisioning_time_sla = user.provisioning_time_slas[str(app.id)]
            threshold = provisioning_time_threshold * provisioning_time_sla

            migration = (
                app.services[0]._Service__migrations[-1]
                if len(app.services[0]._Service__migrations) > 0
                and app.services[0]._Service__migrations[-1]["end"] != None
                else None
            )
            if migration is not None:
                migration_duration = migration["end"] - migration["start"]
                if "was_already_considered" not in migration:
                    migration["was_already_considered"] = False

                if migration["was_already_considered"] is False and threshold_operation(
                    migration_duration, threshold
                ):
                    users_with_provisioning_time_issues.append(user)

                migration["was_already_considered"] = True

    return users_with_provisioning_time_issues


def get_uncached_layers_from_list_of_layers(
    list_of_layers: list, server: object
) -> list:
    """Gets the list of container layers from a given list of layers that are not present in an edge server's layers cache list.

    Args:
        list_of_layers: The list of layers to check against the server's layers cache list.
        server (object): Server whose cached layers will be checked.

    Returns:
        uncached_layers (float): List of layers not present in the edge server's layers cache list.
    """
    # Gathering layers present in the target server (layers, download_queue, waiting_queue)
    layers_downloaded = [layer for layer in server.container_layers]
    layers_on_download_queue = [
        flow.metadata["object"]
        for flow in server.download_queue
        if flow.metadata["object"] == "layer"
    ]
    layers_on_waiting_queue = [layer for layer in server.waiting_queue]
    server_layers = (
        layers_downloaded + layers_on_download_queue + layers_on_waiting_queue
    )

    # Gathering the list of uncached layers
    uncached_layers = []
    for layer in list_of_layers:
        if layer not in server_layers:
            uncached_layers.append(layer)

    return uncached_layers


def get_disk_demand_delta_from_list_of_layers(
    list_of_layers: list, server: object
) -> float:
    """Calculates the additional disk demand necessary to accommodate a list of layers inside
    a given edge server considering the layers already cached inside the analyzed edge server.

    Args:
        list_of_layers: The list of layers to check against the server's layers cache list.
        server (object): Server whose cached layers will be checked.

    Returns:
        disk_demand_delta (float): Disk demand delta.
    """
    # Gathering the list of layers that compose the service's image that are not present in the edge server
    uncached_layers = get_uncached_layers_from_list_of_layers(
        list_of_layers=list_of_layers, server=server
    )

    # Calculating the amount of disk resources required by all service layers not present in the host's disk
    disk_demand_delta = sum([layer.size for layer in uncached_layers])

    return disk_demand_delta


def get_edge_servers_suitable_for_hosting_a_registry(
    registry_cpu_demand: int, registry_memory_demand: int, registry_layers: list
) -> list:
    """Gathers a list of edge server with enough computational capacity available to host a container registry with passed requirements.

    Args:
        registry_cpu_demand (int): Registry's CPU demand.
        registry_memory_demand (int): Registry's memory demand.
        registry_layers (list): Layers that must be served by the registry (and therefore must be accommodated by the host).

    Returns:
        edge_servers_suitable_for_hosting_a_registry (list): Edge servers suitable for hosting a container registry.
    """
    edge_servers_suitable_for_hosting_a_registry = []

    for edge_server in EdgeServer.all():
        if len(edge_server.container_registries) == 0:
            additional_disk_demand = get_disk_demand_delta_from_list_of_layers(
                list_of_layers=registry_layers, server=edge_server
            )

            has_enough_cpu = (
                edge_server.cpu >= edge_server.cpu_demand + registry_cpu_demand
            )
            has_enough_memory = (
                edge_server.memory >= edge_server.memory_demand + registry_memory_demand
            )
            has_enough_disk = (
                edge_server.disk >= edge_server.disk_demand + additional_disk_demand
            )

            if has_enough_cpu and has_enough_memory and has_enough_disk:
                edge_servers_suitable_for_hosting_a_registry.append(edge_server)

    return edge_servers_suitable_for_hosting_a_registry


def provision_new_container_registry(
    target_server: object, container_layers: list = []
) -> object:
    """Provisions a container registry with a list of container layers inside a given target server.

    Args:
        target_server (object): Server that will accommodate the container registry.
        container_layers (list, optional): Container layers that will be available in the container registry. Defaults to [].

    Returns:
        created_registry (object): Created container registry.
    """
    additional_disk_demand = get_disk_demand_delta_from_list_of_layers(
        list_of_layers=container_layers, server=target_server
    )
    registry_cpu_demand = ContainerRegistry.first().cpu_demand
    registry_memory_demand = ContainerRegistry.first().memory_demand

    has_enough_cpu = target_server.cpu >= target_server.cpu_demand + registry_cpu_demand
    has_enough_memory = (
        target_server.memory >= target_server.memory_demand + registry_memory_demand
    )
    has_enough_disk = (
        target_server.disk >= target_server.disk_demand + additional_disk_demand
    )

    if has_enough_cpu and has_enough_memory and has_enough_disk:
        created_registry = ContainerRegistry.provision(
            target_server=target_server,
            registry_cpu_demand=registry_cpu_demand,
            registry_memory_demand=registry_memory_demand,
        )

        created_registry.provisioning_initial_time_step = (
            created_registry.model.schedule.steps + 1
        )
        created_registry.provisioning_final_time_step = None
        created_registry.provisioning_time = None
        created_registry.first_usage_time_step = None

        layers_downloaded = [layer for layer in target_server.container_layers]
        layers_on_download_queue = [
            flow.metadata["object"]
            for flow in target_server.download_queue
            if flow.metadata["object"] == "layer"
        ]
        layers_on_waiting_queue = [layer for layer in target_server.waiting_queue]
        target_server_layers = (
            layers_downloaded + layers_on_download_queue + layers_on_waiting_queue
        )

        for existing_layer in container_layers:
            if existing_layer not in target_server_layers:
                # Creating a new layer object that will be pulled to the target server
                layer = ContainerLayer(
                    digest=existing_layer.digest,
                    size=existing_layer.size,
                    instruction=existing_layer.instruction,
                )
                target_server.model.initialize_agent(agent=layer)
                layer.used_to_provision_a_registry = True

                # Reserving the layer disk demand inside the target server
                target_server.disk_demand += layer.size

                # Adding the layer to the target server's waiting queue (layers it must download at some point)
                target_server.waiting_queue.append(layer)
    else:
        raise Exception(
            f"Unable to create a container registry inside {target_server} due to insufficient resources."
        )

    return created_registry


def estimate_provisioning_time(user: object, analyzed_edge_server: object) -> float:
    """Estimates the time needed to pull the container layers of a given application from a given edge server.

    Args:
        user (object): User that owns the application whose layers are to be pulled.
        analyzed_edge_server (object): Analyzed edge server.

    Returns:
        estimated_provisioning_time (float): Estimated provisioning time.
    """
    if analyzed_edge_server.base_station == user.base_station:
        estimated_provisioning_time = 0

    else:
        path = get_shortest_path(
            origin_switch=analyzed_edge_server.base_station.network_switch,
            target_switch=user.base_station.network_switch,
        )

        # Finding the available bandwidth for the service migration
        bandwidth = min(
            [
                Topology.first()[link][path[index + 1]]["bandwidth"]
                for index, link in enumerate(path[:-1])
            ]
        )

        # Gathering the list of container layers used by the user's application
        layers = []
        layers_digests = ContainerImage.find_by(
            attribute_name="digest",
            attribute_value=user.applications[0].services[0].image_digest,
        ).layers_digests
        for layer_digest in layers_digests:
            layers.append(
                ContainerLayer.find_by(
                    attribute_name="digest", attribute_value=layer_digest
                )
            )

        # Estimating the service provisioning time based on the aggregated size of its layers and the available bandwidth
        topology = user.model.topology
        bandwidth = calculate_bandwidth(
            tick_duration=user.model.tick_duration,
            topology=topology,
            path=path,
            minimum_link_bandwidth=min(
                [
                    topology[path[i]][path[i + 1]]["bandwidth"]
                    for i in range(0, len(path) - 1)
                ]
            ),
        )
        estimated_provisioning_time = sum(layer.size for layer in layers) / bandwidth

    return estimated_provisioning_time
