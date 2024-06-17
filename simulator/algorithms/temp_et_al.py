"""
Mobility-aware strategy that migrates services and container registries according to the performance implications of user mobility.
This strategy provisions registries dynamically based on user mobility, spinning up new registries when application provisioning
times start growing excessively and deprovisioning registries far away from users.

==== REFERENCE ====
Temp, D. C., de Souza, P. S. S., Lorenzon, A. F., Luizelli, M. C., & Rossi, F. D. (2023). Mobility-Aware Registry Migration
for Containerized Applications on Edge Computing Infrastructures. Journal of Network and Computer Applications, 103676.

Link: https://doi.org/10.1016/j.jnca.2023.103676
"""
# Importing EdgeSimPy components
from edge_sim_py.components import *

# Importing helper functions
from simulator.helper_functions import *

VERBOSE = True


def temp_et_al(parameters: dict = {}):
    """Mobility-aware service and registry migration strategy for containerized edge computing infrastructures.

    Args:
        parameters (dict, optional): User-defined parameters. Defaults to {}.
    """
    if VERBOSE:
        registries_on_r620_hosts = len([reg for reg in ContainerRegistry.all() if reg.server and reg.server.model_name == "PowerEdge R620"])
        registries_on_sgi_hosts = len([reg for reg in ContainerRegistry.all() if reg.server and reg.server.model_name == "SGI"])

        services_on_r620_hosts = len([service for service in Service.all() if service.server and service.server.model_name == "PowerEdge R620"])
        services_on_sgi_hosts = len([service for service in Service.all() if service.server and service.server.model_name == "SGI"])

        pw_r620_hosts = sum(s.get_power_consumption() for s in EdgeServer.all() if s.model_name == "PowerEdge R620")
        pw_sgi_hosts = sum(s.get_power_consumption() for s in EdgeServer.all() if s.model_name == "SGI")
        overall_pw = pw_r620_hosts + pw_sgi_hosts

        current_step_metadata = {
            "Time Step": parameters["current_step"],
            "Registries": f"{ContainerRegistry.count()} (R620: {registries_on_r620_hosts}. SGI: {registries_on_sgi_hosts})",
            "Services per Server Model": {"R620": services_on_r620_hosts, "SGI": services_on_sgi_hosts},
            "Power Consumption": f"{overall_pw} (R620: {pw_r620_hosts}. SGI: {pw_sgi_hosts})",
        }
        print("\n")
        print(f"====== {current_step_metadata} ======")

    # Migrating services to keep them as close as possible to their users
    applications = sorted(Application.all(), key=lambda a: a.users[0].delay_slas[str(a.id)] - a.users[0].delays[str(a.id)])
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
                    edge_servers = get_candidate_hosts(user=user)
                    for edge_server in edge_servers:
                        # Stops the search in case the edge server that hosts the service is already the closest to the user
                        if edge_server == service.server:
                            break
                        # Checks if the edge server has resources to host the service
                        elif edge_server.has_capacity_to_host(service) is True:
                            if VERBOSE:
                                print(f"\t\tMigrating {service} from {service.server} to {edge_server}")
                            service.provision(target_server=edge_server)
                            break

    # Removing container registries that are not close to any of the users in the environment
    removing_farthest_container_registries()

    # Calculating the amount of free resources needed to host a registry
    base_registry = ContainerRegistry.first()

    new_registry_layers = []

    registry_layers = [
        ContainerLayer.find_by(attribute_name="digest", attribute_value=layer_digest)
        for layer_digest in ContainerImage.find_by(attribute_name="name", attribute_value="registry").layers_digests
    ]
    new_registry_layers.extend(registry_layers)

    for container_image in ContainerImage.all():
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

    # Trying to provision registries closer to users to avoid SLA violations due to prolonged provisioning times
    users_with_long_prov_time = get_users_with_provisioning_time_issues(
        provisioning_time_threshold=parameters["prov_time_threshold"],
        threshold_logic="exceed",
    )

    while len(users_with_long_prov_time) > 0 and len(edge_servers_suitable_for_hosting_a_registry) > 0:
        for edge_server in edge_servers_suitable_for_hosting_a_registry:
            # List of users whose SLA violations are avoided by putting a registry on that edge server
            edge_server.supported_users = []

            for user in users_with_long_prov_time:
                estimated_provisioning_time = estimate_provisioning_time(user=user, analyzed_edge_server=edge_server)
                provisioning_time_sla = user.provisioning_time_slas[str(user.applications[0].id)]

                if estimated_provisioning_time <= provisioning_time_sla * parameters["prov_time_threshold"]:
                    edge_server.supported_users.append(user)

        best_edge_server = sorted(edge_servers_suitable_for_hosting_a_registry, key=lambda s: -len(s.supported_users))[0]

        # Provisioning a new registry in the best edge server found IF that server serves at least one user
        if len(best_edge_server.supported_users) > 0:
            provision_new_container_registry(container_layers=new_registry_layers, target_server=best_edge_server)
            if VERBOSE:
                print("")
                print(f"\tProvisioning a container registry on {best_edge_server}. NOW ContainerRegistry.count() = {ContainerRegistry.count()}")

            # Updating the list of users with provisioning time issues
            for user in best_edge_server.supported_users:
                if user in users_with_long_prov_time:
                    users_with_long_prov_time.remove(user)

            # Updating the list of edge servers that could host a registry
            edge_servers_suitable_for_hosting_a_registry.remove(best_edge_server)

        else:
            edge_servers_suitable_for_hosting_a_registry = []


def get_candidate_hosts(user: object) -> list:
    """Gathers a sorted list of edge servers that are candidates for hosting the service owned by a given user.

    Args:
        user (object): Analyzed user.

    Returns:
        edge_servers (list): List of candidate hosts.
    """
    edge_servers = []
    for edge_server in EdgeServer.all():
        delay = get_delay(
            wireless_delay=user.base_station.wireless_delay,
            origin_switch=user.base_station.network_switch,
            target_switch=edge_server.base_station.network_switch,
        )
        edge_servers.append({"server": edge_server, "delay": delay})

    # Sorting edge servers by their delay to the user
    edge_servers = [dict_item["server"] for dict_item in sorted(edge_servers, key=lambda e: (e["delay"]))]

    return edge_servers


def removing_farthest_container_registries():
    """Deprovisions the farthest container registries in the infrastructure. We consider a container registry as one of
    the farthest registries if it is not the "closest registry" (in terms of number of hops) to any of the users.

    We use number of hops as distance measure as simulating provisioning times of each user application starting from
    each container registry would incur in a high computational complexity.
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
