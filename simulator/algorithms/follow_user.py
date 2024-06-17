""" Contains the Follow User heuristic.

This strategy strives to keep services as close as possible to their users. As such, whenever a given user moves, the strategy
immediately migrates the corresponding service to the closest edge server available, regardless of the perceived latency.

This strategy is based on the VMs-follow-vehicle presented in the reference below.

==== REFERENCE ====
Yao, H., Bai, C., Zeng, D., Liang, Q., & Fan, Y. (2015). Migrate or not? Exploring virtual machine migration in
roadside cloudlet‐based vehicular cloud. Concurrency and Computation: Practice and Experience, 27(18), 5780-5792.

Link: https://doi.org/10.1002/cpe.3642
"""
# Importing EdgeSimPy components
from edge_sim_py.components import *

# Importing helper functions
from simulator.helper_functions import *

VERBOSE = True


def follow_user(parameters: dict = {}):
    """Simple strategy that migrates services to as close as possible to their users whenever their users move across the map.

    Args:
        parameters (dict, optional): User-defined parameters. Defaults to {}.
    """
    for user in User.all():
        # Getting the list of edge servers sorted by the distance between their base stations and the user base station
        edge_servers = get_candidate_hosts(user=user)

        # Migrating services to keep them as close as possible to their users
        for application in user.applications:
            for service in application.services:
                if service.being_provisioned is False:
                    # Finding the closest edge server that has resources to host the service
                    for edge_server_metadata in edge_servers:
                        edge_server = edge_server_metadata["object"]
                        # Stops the search in case the edge server that hosts the service is already the closest to the user
                        if edge_server == service.server:
                            break
                        # Checks if the edge server has resources to host the service
                        elif edge_server.has_capacity_to_host(service):
                            service.provision(target_server=edge_server)
                            break


def get_candidate_hosts(user: object) -> list:
    """Gathers a sorted list of edge servers that are candidates for hosting the service owned by a given user.

    Args:
        user (object): User that accesses the service to be migrated.

    Returns:
        edge_servers (list): List of candidate hosts.
    """
    edge_servers = []

    for edge_server in EdgeServer.all():
        # Estimating the delay between the user and the analyzed edge server
        delay = get_delay(
            wireless_delay=user.base_station.wireless_delay,
            origin_switch=user.base_station.network_switch,
            target_switch=edge_server.base_station.network_switch,
        )
        edge_server_metadata = {
            "object": edge_server,
            "delay": delay,
        }
        edge_servers.append(edge_server_metadata)

    # Sorting edge servers
    edge_servers = sorted(edge_servers, key=lambda e: (e["delay"]))

    return edge_servers
