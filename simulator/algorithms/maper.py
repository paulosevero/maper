"""
TBD.
"""

# Importing EdgeSimPy components
from edge_sim_py.components import *

# Importing helper functions
from simulator.helper_functions import *

VERBOSE = True


def maper(parameters: dict = {}):
    """TBD.

    Args:
        parameters (dict, optional): User-defined parameters. Defaults to {}.
    """
    if VERBOSE:
        display_system_state_at_current_step(current_step=parameters["current_step"])

    # Relocating services to keep them as close as possible to their users
    relocate_services(parameters=parameters)

    # Removing container registries that are not close to any of the users in the environment
    removing_farthest_container_registries()

    # Dynamically provisioning new registries within the infrastructure to avoid provisioning time SLA violations
    provision_container_registries(parameters=parameters)


def relocate_services(parameters: dict = {}):
    # Sorting applications according to the distance between their actual and their expected delays
    applications = sorted(
        Application.all(),
        key=lambda a: a.users[0].delay_slas[str(a.id)] - a.users[0].delays[str(a.id)],
    )

    print("\t=== Application Migrations ===")
    for application in applications:
        user = application.users[0]
        delay = user.delays[str(application.id)]
        delay_sla = user.delay_slas[str(application.id)]
        delay_threshold = delay_sla * parameters["delay_threshold"]

        if delay > delay_threshold:
            for service in application.services:
                if service.being_provisioned is False:
                    # Finding the closest edge server that has resources to host the service
                    edge_servers = get_candidate_hosts(user=user, service=service, delay_threshold=delay_threshold)
                    for edge_server_metadata in edge_servers:
                        edge_server = edge_server_metadata["object"]

                        if edge_server == service.server or edge_server_metadata["violates_delay_sla"]:
                            break
                        else:
                            if edge_server.has_capacity_to_host(service):
                                if VERBOSE:
                                    print(f"\t\tMigrating {service} from {service.server} to {edge_server}")
                                service.provision(target_server=edge_server)
                                break


def provision_container_registries(parameters: dict = {}):
    # Gathering the list of users with provisioning time issues
    users_with_long_prov_time = get_users_with_provisioning_time_issues(provisioning_time_threshold=parameters["prov_time_threshold"], threshold_logic="exceed")

    # Calculating the amount of free resources needed to host a registry
    base_registry = ContainerRegistry.first()
    new_registry_layers = []
    registry_layers = [
        ContainerLayer.find_by(attribute_name="digest", attribute_value=layer_digest)
        for layer_digest in ContainerImage.find_by(attribute_name="name", attribute_value="registry").layers_digests
    ]
    new_registry_layers.extend(registry_layers)

    # Gathering the list of container layers usd by all the services in the environment (thereby, we follow a full registry mirroring approach)
    for user in User.all():
        container_image = ContainerImage.find_by(attribute_name="digest", attribute_value=user.applications[0].services[0].image_digest)
        if container_image.name != "registry":
            for layer_digest in container_image.layers_digests:
                layer = ContainerLayer.find_by(attribute_name="digest", attribute_value=layer_digest)
                if layer not in new_registry_layers:
                    new_registry_layers.append(layer)

    # Gathering the list of edge servers that could host a registry
    edge_servers_suitable_for_hosting_a_registry = get_edge_servers_suitable_for_hosting_a_registry(
        registry_cpu_demand=base_registry.cpu_demand,
        registry_memory_demand=base_registry.memory_demand,
        registry_layers=new_registry_layers,
    )

    while len(users_with_long_prov_time) > 0 and len(edge_servers_suitable_for_hosting_a_registry) > 0:
        edge_servers_metadata = []

        for edge_server in edge_servers_suitable_for_hosting_a_registry:
            edge_server_metadata = {
                "object": edge_server,
                "power_consumption": edge_server.get_power_consumption(),
                "supported_users": [],
                "number_of_not_supported_users": len(users_with_long_prov_time),
            }

            for user in users_with_long_prov_time:
                provisioning_time_sla = user.provisioning_time_slas[str(user.applications[0].id)]
                estimated_provisioning_time = estimate_provisioning_time(user=user, analyzed_edge_server=edge_server)

                if estimated_provisioning_time < provisioning_time_sla * parameters["prov_time_threshold"]:
                    edge_server_metadata["supported_users"].append(user)
                    edge_server_metadata["number_of_not_supported_users"] -= 1

            edge_servers_metadata.append(edge_server_metadata)

        # Finding minimum and maximum metadata values for the edge server list
        minimum_and_maximum = find_minimum_and_maximum(metadata=edge_servers_metadata)

        # Calculating the normalized attributes for each edge server
        for edge_server_metadata in edge_servers_metadata:
            edge_server_metadata["norm_number_of_not_supported_users"] = get_norm(
                metadata=edge_server_metadata,
                attr_name="number_of_not_supported_users",
                min=minimum_and_maximum["minimum"],
                max=minimum_and_maximum["maximum"],
            )
            edge_server_metadata["norm_power_consumption"] = get_norm(
                metadata=edge_server_metadata,
                attr_name="power_consumption",
                min=minimum_and_maximum["minimum"],
                max=minimum_and_maximum["maximum"],
            )

        edge_servers_metadata = sorted(edge_servers_metadata, key=lambda s: (s["number_of_not_supported_users"] + s["norm_power_consumption"]))

        for edge_server_metadata in edge_servers_metadata:
            print(f"\t\t\t{edge_server_metadata}")

        best_edge_server_metadata = edge_servers_metadata[0]

        # Provisioning a new registry in the best edge server found IF that server serves at least one user
        if len(best_edge_server_metadata["supported_users"]) > 0:
            best_edge_server = best_edge_server_metadata["object"]
            provision_new_container_registry(container_layers=new_registry_layers, target_server=best_edge_server)
            if VERBOSE:
                print("")
                print(f"\tProvisioning a container registry on {best_edge_server}. NOW ContainerRegistry.count() = {ContainerRegistry.count()}")

            # Updating the list of users with provisioning time issues
            for user in best_edge_server_metadata["supported_users"]:
                if user in users_with_long_prov_time:
                    users_with_long_prov_time.remove(user)

            # Updating the list of edge servers that could host a registry
            edge_servers_suitable_for_hosting_a_registry.remove(best_edge_server)

        else:
            edge_servers_suitable_for_hosting_a_registry = []


def get_candidate_hosts(user: object, service: object, delay_threshold: float) -> list:
    """Gathers a sorted list of edge servers that are candidates for hosting the service owned by a given user.

    Args:
        user (object): User that accesses the service to be migrated.
        service (object): Service to be migrated.
        delay_threshold (float): Delay threshold.

    Returns:
        edge_servers (list): List of candidate hosts.
    """
    # Gathering the service's delay SLA
    delay_sla = user.delay_slas[str(user.applications[0].id)]

    edge_servers = []

    for edge_server in EdgeServer.all():
        # Estimating the delay between the user and the analyzed edge server
        delay = get_delay(
            wireless_delay=user.base_station.wireless_delay,
            origin_switch=user.base_station.network_switch,
            target_switch=edge_server.base_station.network_switch,
        )

        # Gathering the list of container layers used by the service's image
        service_image = ContainerImage.find_by(attribute_name="digest", attribute_value=service.image_digest)
        service_layers = [ContainerLayer.find_by(attribute_name="digest", attribute_value=digest) for digest in service_image.layers_digests]

        # Calculating the aggregated size of container layers used by the service's image that are NOT cached in the candidate server
        layers_downloaded = [layer for layer in edge_server.container_layers]
        amount_of_uncached_layers = 0
        for layer in layers_downloaded:
            if not any(layer.digest == service_layer.digest for service_layer in service_layers):
                amount_of_uncached_layers += layer.size

        edge_server_metadata = {
            "object": edge_server,
            "delay": delay,
            "delay_above_threshold": 1 if delay > delay_threshold else 0,
            "violates_delay_sla": 1 if delay > delay_sla else 0,
            "amount_of_uncached_layers": amount_of_uncached_layers,
            "power_consumption": edge_server.get_power_consumption(),
        }
        edge_servers.append(edge_server_metadata)

    # Finding minimum and maximum metadata values for the edge server list
    min_and_max = find_minimum_and_maximum(metadata=edge_servers)

    # Calculating the normalized attributes for each edge server
    for edge_server_metadata in edge_servers:
        edge_server_metadata["norm_delay"] = get_norm(
            metadata=edge_server_metadata,
            attr_name="delay",
            min=min_and_max["minimum"],
            max=min_and_max["maximum"],
        )
        edge_server_metadata["norm_amount_of_uncached_layers"] = get_norm(
            metadata=edge_server_metadata,
            attr_name="amount_of_uncached_layers",
            min=min_and_max["minimum"],
            max=min_and_max["maximum"],
        )
        edge_server_metadata["norm_power_consumption"] = get_norm(
            metadata=edge_server_metadata,
            attr_name="power_consumption",
            min=min_and_max["minimum"],
            max=min_and_max["maximum"],
        )

    # Sorting edge servers
    edge_servers = sorted(
        edge_servers,
        key=lambda e: (
            e["delay_above_threshold"],
            e["norm_delay"] + e["norm_amount_of_uncached_layers"] + e["norm_power_consumption"],
        ),
    )

    return edge_servers


def removing_farthest_container_registries():
    """Deprovisions the farthest registry in the infrastructure. We consider a registry
    as the farthest one if it is not the "closest registry" to any of the users.
    """
    registries = []
    for registry in ContainerRegistry.all():
        if registry.available:
            registry.users_supported = 0
            registries.append(registry)

    if len(registries) >= 2:
        for user in User.all():
            closest_registry = None
            closest_registry_provisioning_time = float("inf")
            for registry in registries:
                estimated_provisioning_time = estimate_provisioning_time(user=user, analyzed_edge_server=registry.server)

                if closest_registry_provisioning_time > estimated_provisioning_time:
                    closest_registry = registry
                    closest_registry_provisioning_time = estimated_provisioning_time

            closest_registry.users_supported += 1

        print("")
        print("\t=== Registry Analysis ===")
        for index, registry in enumerate(registries, 1):
            if VERBOSE:
                print(f"\t\t[{index}]{registry}. Supported Users: {registry.users_supported}")
            if registry.users_supported == 0:
                deprovision_status = registry.deprovision(purge_images=True)
                if VERBOSE and deprovision_status is True:
                    print(f"\t\t\tDeprovisioning {registry}")
