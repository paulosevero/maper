# Importing Python Libraries
from statistics import mean, stdev
import networkx as nx
import json
import time

# Importing EdgeSimPy components
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.container_registry import ContainerRegistry
from edge_sim_py.components.network_flow import NetworkFlow
from edge_sim_py.components.container_image import ContainerImage
from edge_sim_py.components.service import Service

# Importing helper functions
from simulator.helper_functions import *


def simulator_monitor(self):
    """Monitors a set of metrics from the model and its agents."""
    # Defining which entities (i.e., agents) are monitored
    agents_to_monitor = EdgeServer.all() + Service.all() + ContainerRegistry.all()

    # Collecting agent-level metrics
    for agent in agents_to_monitor:
        # Creating metrics dictionary for the entity type if it doesn't exist
        if agent.__class__.__name__ not in self.agent_metrics:
            self.agent_metrics[agent.__class__.__name__] = {}

        # Creating metrics list for the entity instance if it doesn't exist
        if str(agent.id) not in self.agent_metrics[agent.__class__.__name__]:
            self.agent_metrics[agent.__class__.__name__][str(agent.id)] = []

        # Adding fresh entity instance metrics to its metrics list
        self.agent_metrics[agent.__class__.__name__][str(agent.id)].append(agent.collect())

    if self.schedule.steps == self.last_dump + self.dump_interval:
        self.dump_data_to_disk()
        self.last_dump = self.schedule.steps


def simulator_consolidate_metrics(self, export_file: bool = True, hide_detailed_metrics: bool = True) -> dict:
    """Consolidates simulation metrics into a researcher-friendly structure.

    Returns:
        metrics (dict): Consolidated list of metrics.
    """
    # Grouping simulation parameters
    simulation_parameters = {"simulation_parameters": self.resource_management_algorithm_parameters}

    # Grouping instance-specific metrics
    edge_server_metrics = self.consolidate_edge_server_metrics()
    container_registry_metrics = self.consolidate_container_registry_metrics()
    service_metrics = self.consolidate_service_metrics()

    # Creating unified data structure for simulation parameters and monitored metrics
    detailed_metrics = prettify_dictionary_for_dump(
        dictionary={
            **simulation_parameters,
            **edge_server_metrics,
            **container_registry_metrics,
            **service_metrics,
        },
        list_size_threshold=5,
    )
    summarized_metrics = prettify_dictionary_for_dump(
        dictionary={
            "timestamp": int(time.time()),
            **self.resource_management_algorithm_parameters,
            "overall_edge_server_overloaded": edge_server_metrics["overall_edge_server_overloaded"],
            "sum_overall_power_consumption": edge_server_metrics["overall_power_consumption"]["sum"],
            "avg_overall_power_consumption": edge_server_metrics["overall_power_consumption"]["avg"],
            "latency_sla_violations": service_metrics["service_latency_sla_viol"]["sum"],
            "number_of_migrations": service_metrics["service_number_of_migrations"]["all"],
            "provisioning_time_sla_violations": service_metrics["service_prov_time_sla_viol"]["all"],
        },
        list_size_threshold=5,
    )

    # Printing metrics
    print("\n\n")
    print("====================================")
    print("======== SIMULATION METRICS ========")
    print("====================================")
    if hide_detailed_metrics is False:
        print("==== DETAILED ====")
        print(json.dumps(detailed_metrics, indent=4))

        print("")

    print("==== SUMMARIZED ====")
    print(json.dumps(summarized_metrics, indent=4))

    print("")

    # Exporting metrics to an external spreadsheet file if the "export_file" is set to True
    if export_file is True:
        save_dict_to_excel(dictionary=detailed_metrics, output_filename=f"detailed_{self.output_file_name}")
        append_dict_to_csv(dictionary=summarized_metrics, output_filename=f"summarized_results_{self.resource_management_algorithm.__name__}")

    return detailed_metrics


def simulator_consolidate_edge_server_metrics(self) -> dict:
    """Consolidates collected edge server metrics into a researcher-friendly structure.

    Returns:
        edge_server_metrics (dict): Consolidated list of edge server metrics.
    """
    # Initializing variables that will help the metrics collection
    all_edge_servers_metrics = list(self.agent_metrics["EdgeServer"].values())
    time_steps = [record["Time Step"] for record in all_edge_servers_metrics[0]]
    edge_server_models = list(set([edge_server.model_name for edge_server in EdgeServer.all()]))

    # Collecting per-step metric observations
    per_step_edge_server_occupation_per_model = {}
    per_step_edge_server_power_consumption_per_model = {}
    overall_edge_server_power_consumption = {"all": []}
    overall_edge_server_overloaded = 0
    for time_step in time_steps:
        occupation = {}
        power_consumption = {}
        overall_power_consumption = 0

        for entity_id in self.agent_metrics["EdgeServer"].keys():
            overall_power_consumption += self.agent_metrics["EdgeServer"][entity_id][int(time_step)]["Power"]
            overall_edge_server_overloaded += self.agent_metrics["EdgeServer"][entity_id][int(time_step)]["Is Overloaded"]

            entity_model_name = self.agent_metrics["EdgeServer"][entity_id][int(time_step)]["Model Name"]
            if entity_model_name not in occupation:
                occupation[entity_model_name] = []
            occupation[entity_model_name].append(self.agent_metrics["EdgeServer"][entity_id][int(time_step)]["CPU Occupation (%)"])

            if entity_model_name not in power_consumption:
                power_consumption[entity_model_name] = []
            power_consumption[entity_model_name].append(self.agent_metrics["EdgeServer"][entity_id][int(time_step)]["Power"])

        for edge_server_model in edge_server_models:
            if edge_server_model not in per_step_edge_server_occupation_per_model:
                per_step_edge_server_occupation_per_model[edge_server_model] = []
            per_step_edge_server_occupation_per_model[edge_server_model].append(
                {
                    "min": min(occupation[edge_server_model]),
                    "max": max(occupation[edge_server_model]),
                    "avg": mean(occupation[edge_server_model]),
                    "stdev": stdev(occupation[edge_server_model]),
                }
            )

            if edge_server_model not in per_step_edge_server_power_consumption_per_model:
                per_step_edge_server_power_consumption_per_model[edge_server_model] = []
            per_step_edge_server_power_consumption_per_model[edge_server_model].append(
                {
                    "min": min(power_consumption[edge_server_model]),
                    "max": max(power_consumption[edge_server_model]),
                    "avg": mean(power_consumption[edge_server_model]),
                    "stdev": stdev(power_consumption[edge_server_model]),
                }
            )

        overall_edge_server_power_consumption["all"].append(overall_power_consumption)

    # Consolidating collected edge server metrics
    edge_server_occupation_per_model = []
    edge_server_power_consumption_per_model = []

    overall_edge_server_power_consumption["min"] = (
        min(overall_edge_server_power_consumption["all"]) if len(overall_edge_server_power_consumption["all"]) > 0 else 0
    )
    overall_edge_server_power_consumption["max"] = (
        max(overall_edge_server_power_consumption["all"]) if len(overall_edge_server_power_consumption["all"]) > 0 else 0
    )
    overall_edge_server_power_consumption["sum"] = (
        sum(overall_edge_server_power_consumption["all"]) if len(overall_edge_server_power_consumption["all"]) > 0 else 0
    )
    overall_edge_server_power_consumption["avg"] = (
        mean(overall_edge_server_power_consumption["all"]) if len(overall_edge_server_power_consumption["all"]) > 0 else 0
    )
    overall_edge_server_power_consumption["stdev"] = (
        stdev(overall_edge_server_power_consumption["all"]) if len(overall_edge_server_power_consumption["all"]) > 1 else 0
    )

    for edge_server_model in edge_server_models:
        average_occupation_values = [per_step_metrics["avg"] for per_step_metrics in per_step_edge_server_occupation_per_model[edge_server_model]]
        summarized_edge_server_model_occupation_metrics = {
            "edge_server_model": edge_server_model,
            "all": average_occupation_values,
            "min": min(average_occupation_values),
            "max": max(average_occupation_values),
            "avg": mean(average_occupation_values),
            "stdev": stdev(average_occupation_values),
        }
        edge_server_occupation_per_model.append(summarized_edge_server_model_occupation_metrics)

        average_power_consumption_values = [per_step_metrics["avg"] for per_step_metrics in per_step_edge_server_power_consumption_per_model[edge_server_model]]
        summarized_edge_server_model_power_consumption_metrics = {
            "edge_server_model": edge_server_model,
            "min": min(average_power_consumption_values),
            "max": max(average_power_consumption_values),
            "avg": mean(average_power_consumption_values),
            "stdev": stdev(average_power_consumption_values),
        }
        edge_server_power_consumption_per_model.append(summarized_edge_server_model_power_consumption_metrics)

    # Creating unified data structure for monitored edge server metrics
    edge_server_metrics = {
        "overall_edge_server_overloaded": overall_edge_server_overloaded,
        "overall_power_consumption": overall_edge_server_power_consumption,
        "occupation_per_server_model": edge_server_occupation_per_model,
        "power_cons_per_server_model": edge_server_power_consumption_per_model,
    }
    return edge_server_metrics


def simulator_consolidate_container_registry_metrics(self) -> dict:
    """Consolidates collected container registry metrics into a researcher-friendly structure.

    Returns:
        container_registry_metrics (dict): Consolidated list of container registry metrics.
    """
    # Initializing variables that will help the metrics collection
    simulated_time_steps = self.schedule.steps

    # Collecting per-step metric observations
    overall_container_registry_provisioning_time = {"all": {}}
    overall_container_registry_time_to_usefulness = {"all": {}}
    per_step_container_registry_provisioned_instances = []
    per_step_container_registry_cpu_demand = []
    per_step_container_registry_memory_demand = []
    per_step_container_registry_disk_demand = []

    for time_step in range(1, simulated_time_steps + 1):
        container_registry_provisioned_instances = 0
        container_registry_cpu_demand = []
        container_registry_memory_demand = []
        container_registry_disk_demand = []

        for entity_id in self.agent_metrics["ContainerRegistry"].keys():
            entity_state_in_the_current_time_step = next(
                (
                    entity_metrics_record
                    for entity_metrics_record in self.agent_metrics["ContainerRegistry"][entity_id]
                    if entity_metrics_record["Time Step"] == int(time_step)
                ),
                None,
            )
            if entity_state_in_the_current_time_step is not None:
                container_registry_provisioned_instances += 1
                container_registry_cpu_demand.append(entity_state_in_the_current_time_step["CPU/RAM Demand"][0])
                container_registry_memory_demand.append(entity_state_in_the_current_time_step["CPU/RAM Demand"][1])
                container_registry_disk_demand.append(entity_state_in_the_current_time_step["Disk Demand"])

                instance_id = entity_state_in_the_current_time_step["Instance ID"]

                if entity_state_in_the_current_time_step["Available"] is True and instance_id not in overall_container_registry_provisioning_time["all"]:
                    overall_container_registry_provisioning_time["all"][instance_id] = entity_state_in_the_current_time_step["Provisioning Time"]

                if (
                    entity_state_in_the_current_time_step["Time to Usefulness"] is not None
                    and instance_id not in overall_container_registry_time_to_usefulness["all"]
                ):
                    overall_container_registry_time_to_usefulness["all"][instance_id] = entity_state_in_the_current_time_step["Time to Usefulness"]

        per_step_container_registry_provisioned_instances.append(container_registry_provisioned_instances)
        per_step_container_registry_cpu_demand.append(
            {
                "min": min(container_registry_cpu_demand) if len(container_registry_cpu_demand) > 0 else 0,
                "max": max(container_registry_cpu_demand) if len(container_registry_cpu_demand) > 0 else 0,
                "avg": mean(container_registry_cpu_demand) if len(container_registry_cpu_demand) > 0 else 0,
                "stdev": stdev(container_registry_cpu_demand) if len(container_registry_cpu_demand) > 1 else 0,
            }
        )
        per_step_container_registry_memory_demand.append(
            {
                "min": min(container_registry_memory_demand) if len(container_registry_memory_demand) > 0 else 0,
                "max": max(container_registry_memory_demand) if len(container_registry_memory_demand) > 0 else 0,
                "avg": mean(container_registry_memory_demand) if len(container_registry_memory_demand) > 0 else 0,
                "stdev": stdev(container_registry_memory_demand) if len(container_registry_memory_demand) > 1 else 0,
            }
        )
        per_step_container_registry_disk_demand.append(
            {
                "min": min(container_registry_disk_demand) if len(container_registry_disk_demand) > 0 else 0,
                "max": max(container_registry_disk_demand) if len(container_registry_disk_demand) > 0 else 0,
                "avg": mean(container_registry_disk_demand) if len(container_registry_disk_demand) > 0 else 0,
                "stdev": stdev(container_registry_disk_demand) if len(container_registry_disk_demand) > 1 else 0,
            }
        )

    # Consolidating collected container registry metrics
    overall_container_registry_provisioning_time["min"] = (
        min(overall_container_registry_provisioning_time["all"].values()) if len(overall_container_registry_provisioning_time["all"].values()) > 0 else 0
    )
    overall_container_registry_provisioning_time["max"] = (
        max(overall_container_registry_provisioning_time["all"].values()) if len(overall_container_registry_provisioning_time["all"].values()) > 0 else 0
    )
    overall_container_registry_provisioning_time["avg"] = (
        mean(overall_container_registry_provisioning_time["all"].values()) if len(overall_container_registry_provisioning_time["all"].values()) > 0 else 0
    )
    overall_container_registry_provisioning_time["stdev"] = (
        stdev(overall_container_registry_provisioning_time["all"].values()) if len(overall_container_registry_provisioning_time["all"].values()) > 1 else 0
    )

    overall_container_registry_time_to_usefulness["min"] = (
        min(overall_container_registry_time_to_usefulness["all"].values()) if len(overall_container_registry_time_to_usefulness["all"].values()) > 0 else 0
    )
    overall_container_registry_time_to_usefulness["max"] = (
        max(overall_container_registry_time_to_usefulness["all"].values()) if len(overall_container_registry_time_to_usefulness["all"].values()) > 0 else 0
    )
    overall_container_registry_time_to_usefulness["avg"] = (
        mean(overall_container_registry_time_to_usefulness["all"].values()) if len(overall_container_registry_time_to_usefulness["all"].values()) > 0 else 0
    )
    overall_container_registry_time_to_usefulness["stdev"] = (
        stdev(overall_container_registry_time_to_usefulness["all"].values()) if len(overall_container_registry_time_to_usefulness["all"].values()) > 1 else 0
    )

    overall_container_registry_provisioned_instances = {
        "all": per_step_container_registry_provisioned_instances,
        "min": min(per_step_container_registry_provisioned_instances) if len(per_step_container_registry_provisioned_instances) > 0 else 0,
        "max": max(per_step_container_registry_provisioned_instances) if len(per_step_container_registry_provisioned_instances) > 0 else 0,
        "avg": mean(per_step_container_registry_provisioned_instances) if len(per_step_container_registry_provisioned_instances) > 0 else 0,
        "stdev": stdev(per_step_container_registry_provisioned_instances) if len(per_step_container_registry_provisioned_instances) > 1 else 0,
    }
    avg_per_step_container_registry_cpu_demand_values = [per_step_metrics["avg"] for per_step_metrics in per_step_container_registry_cpu_demand]
    overall_container_registry_cpu_demand = {
        "min": min(avg_per_step_container_registry_cpu_demand_values) if len(avg_per_step_container_registry_cpu_demand_values) > 0 else 0,
        "max": max(avg_per_step_container_registry_cpu_demand_values) if len(avg_per_step_container_registry_cpu_demand_values) > 0 else 0,
        "avg": mean(avg_per_step_container_registry_cpu_demand_values) if len(avg_per_step_container_registry_cpu_demand_values) > 0 else 0,
        "stdev": stdev(avg_per_step_container_registry_cpu_demand_values) if len(avg_per_step_container_registry_cpu_demand_values) > 1 else 0,
    }
    avg_per_step_container_registry_memory_demand_values = [per_step_metrics["avg"] for per_step_metrics in per_step_container_registry_memory_demand]
    overall_container_registry_memory_demand = {
        "min": min(avg_per_step_container_registry_memory_demand_values) if len(avg_per_step_container_registry_memory_demand_values) > 0 else 0,
        "max": max(avg_per_step_container_registry_memory_demand_values) if len(avg_per_step_container_registry_memory_demand_values) > 0 else 0,
        "avg": mean(avg_per_step_container_registry_memory_demand_values) if len(avg_per_step_container_registry_memory_demand_values) > 0 else 0,
        "stdev": stdev(avg_per_step_container_registry_memory_demand_values) if len(avg_per_step_container_registry_memory_demand_values) > 1 else 0,
    }
    avg_per_step_container_registry_disk_demand_values = [per_step_metrics["avg"] for per_step_metrics in per_step_container_registry_disk_demand]
    overall_container_registry_disk_demand = {
        "step_by_step_avg": avg_per_step_container_registry_disk_demand_values,
        "min": min(avg_per_step_container_registry_disk_demand_values) if len(avg_per_step_container_registry_disk_demand_values) > 0 else 0,
        "max": max(avg_per_step_container_registry_disk_demand_values) if len(avg_per_step_container_registry_disk_demand_values) > 0 else 0,
        "avg": mean(avg_per_step_container_registry_disk_demand_values) if len(avg_per_step_container_registry_disk_demand_values) > 0 else 0,
        "stdev": stdev(avg_per_step_container_registry_disk_demand_values) if len(avg_per_step_container_registry_disk_demand_values) > 1 else 0,
    }

    # Creating unified data structure for monitored container registry metrics
    container_registry_metrics = {
        "registry_prov_time": overall_container_registry_provisioning_time,
        "registry_time_to_usefulness": overall_container_registry_time_to_usefulness,
        "registry_provisioned_instances": overall_container_registry_provisioned_instances,
        "registry_cpu_demand": overall_container_registry_cpu_demand,
        "registry_memory_demand": overall_container_registry_memory_demand,
        "registry_disk_demand": overall_container_registry_disk_demand,
    }
    return container_registry_metrics


def simulator_consolidate_service_metrics(self) -> dict:
    """Consolidates collected service metrics into a researcher-friendly structure.

    Returns:
        service_metrics (dict): Consolidated list of service metrics.
    """
    # Initializing variables that will help the metrics collection
    simulated_time_steps = self.schedule.steps

    # Collecting migration-related metrics
    overall_service_number_of_migrations = 0
    overall_service_provisioning_time_sla_violations = 0
    overall_service_migration_times = {"all": []}
    overall_service_waiting_times = {"all": []}
    overall_service_layer_pulling_times = {"all": []}
    overall_service_state_migration_times = {"all": []}
    for service in Service.all():
        application = service.application
        user = application.users[0]
        provisioning_time_sla = user.provisioning_time_slas[str(application.id)]

        for migration in service._Service__migrations:
            if migration["end"] != None:
                overall_service_number_of_migrations += 1
                overall_service_migration_times["all"].append(migration["end"] - migration["start"])
                overall_service_waiting_times["all"].append(migration["waiting_time"])
                overall_service_layer_pulling_times["all"].append(migration["pulling_layers_time"])
                overall_service_state_migration_times["all"].append(migration["migrating_service_state_time"])
                if migration["end"] - migration["start"] > provisioning_time_sla:
                    overall_service_provisioning_time_sla_violations += 1

    # Collecting latency-specific metrics
    per_step_service_latency_sla_violations = []
    per_step_service_latency_observations = []
    for time_step in range(1, simulated_time_steps + 1):
        latency_sla_violations = 0
        latency_observations = []
        for entity_id in self.agent_metrics["Service"].keys():
            service_metrics = self.agent_metrics["Service"][entity_id][int(time_step)]
            if service_metrics["Latency"] != float("inf"):
                latency_observations.append(service_metrics["Latency"])
            if service_metrics["Is Violating Latency SLA"] == 1:
                latency_sla_violations += 1

        per_step_service_latency_sla_violations.append(latency_sla_violations)
        per_step_service_latency_observations.append(
            {
                "min": min(latency_observations),
                "max": max(latency_observations),
                "avg": mean(latency_observations),
                "stdev": stdev(latency_observations),
            }
        )

    # Consolidating collected service metrics
    overall_service_migration_times["min"] = min(overall_service_migration_times["all"]) if len(overall_service_migration_times["all"]) > 0 else 0
    overall_service_migration_times["max"] = max(overall_service_migration_times["all"]) if len(overall_service_migration_times["all"]) > 0 else 0
    overall_service_migration_times["avg"] = mean(overall_service_migration_times["all"]) if len(overall_service_migration_times["all"]) > 0 else 0
    overall_service_migration_times["stdev"] = stdev(overall_service_migration_times["all"]) if len(overall_service_migration_times["all"]) > 1 else 0

    overall_service_waiting_times["min"] = min(overall_service_waiting_times["all"]) if len(overall_service_waiting_times["all"]) > 0 else 0
    overall_service_waiting_times["max"] = max(overall_service_waiting_times["all"]) if len(overall_service_waiting_times["all"]) > 0 else 0
    overall_service_waiting_times["avg"] = mean(overall_service_waiting_times["all"]) if len(overall_service_waiting_times["all"]) > 0 else 0
    overall_service_waiting_times["stdev"] = stdev(overall_service_waiting_times["all"]) if len(overall_service_waiting_times["all"]) > 1 else 0

    overall_service_layer_pulling_times["min"] = min(overall_service_layer_pulling_times["all"]) if len(overall_service_layer_pulling_times["all"]) > 0 else 0
    overall_service_layer_pulling_times["max"] = max(overall_service_layer_pulling_times["all"]) if len(overall_service_layer_pulling_times["all"]) > 0 else 0
    overall_service_layer_pulling_times["avg"] = mean(overall_service_layer_pulling_times["all"]) if len(overall_service_layer_pulling_times["all"]) > 0 else 0
    overall_service_layer_pulling_times["stdev"] = (
        stdev(overall_service_layer_pulling_times["all"]) if len(overall_service_layer_pulling_times["all"]) > 1 else 0
    )

    overall_service_state_migration_times["min"] = (
        min(overall_service_state_migration_times["all"]) if len(overall_service_state_migration_times["all"]) > 0 else 0
    )
    overall_service_state_migration_times["max"] = (
        max(overall_service_state_migration_times["all"]) if len(overall_service_state_migration_times["all"]) > 0 else 0
    )
    overall_service_state_migration_times["avg"] = (
        mean(overall_service_state_migration_times["all"]) if len(overall_service_state_migration_times["all"]) > 0 else 0
    )
    overall_service_state_migration_times["stdev"] = (
        stdev(overall_service_state_migration_times["all"]) if len(overall_service_state_migration_times["all"]) > 1 else 0
    )

    overall_service_latency_sla_violations = {
        "all": per_step_service_latency_sla_violations,
        "sum": sum(per_step_service_latency_sla_violations),
        "min": min(per_step_service_latency_sla_violations),
        "max": max(per_step_service_latency_sla_violations),
        "avg": mean(per_step_service_latency_sla_violations),
        "stdev": stdev(per_step_service_latency_sla_violations),
    }
    overall_service_latency_observations = {
        "min": min([per_step_metrics["avg"] for per_step_metrics in per_step_service_latency_observations]),
        "max": max([per_step_metrics["avg"] for per_step_metrics in per_step_service_latency_observations]),
        "avg": mean([per_step_metrics["avg"] for per_step_metrics in per_step_service_latency_observations]),
        "stdev": stdev([per_step_metrics["avg"] for per_step_metrics in per_step_service_latency_observations]),
    }

    # Creating unified data structure for monitored service metrics
    service_metrics = {
        "service_number_of_migrations": {"all": overall_service_number_of_migrations},
        "service_prov_time_sla_viol": {"all": overall_service_provisioning_time_sla_violations},
        "service_migration_times": overall_service_migration_times,
        "service_waiting_times": overall_service_waiting_times,
        "service_layer_pulling_times": overall_service_layer_pulling_times,
        "service_state_migration_times": overall_service_state_migration_times,
        "service_latency_sla_viol": overall_service_latency_sla_violations,
        "service_latency_observations": overall_service_latency_observations,
    }
    return service_metrics


def edge_server_collect(self) -> dict:
    """Method that collects a set of metrics for the object.

    Returns:
        metrics (dict): Object metrics.
    """
    is_overloaded = 0
    if self.cpu_demand > self.cpu or self.memory_demand > self.memory or self.disk_demand > self.disk:
        is_overloaded = 1
    metrics = {
        "Time Step": self.model.schedule.steps,
        "Instance ID": self.id,
        "Model Name": self.model_name,
        "Capacity": [self.cpu, self.memory, self.disk],
        "Demand": [self.cpu_demand, self.memory_demand, self.disk_demand],
        "CPU Occupation (%)": self.cpu_demand * 100 / self.cpu,
        "Is Overloaded": is_overloaded,
        "Services": [service.id for service in self.services],
        "Registries": [registry.id for registry in self.container_registries],
        "Power": self.get_power_consumption(),
    }
    return metrics


def service_collect(self) -> dict:
    """Method that collects a set of metrics for the object.

    Returns:
        metrics (dict): Object metrics.
    """
    application = self.application
    user = application.users[0]
    latency = user.delays[str(application.id)]
    latency_sla = user.delay_slas[str(application.id)]
    provisioning_time_sla = user.provisioning_time_slas[str(application.id)]
    metrics = {
        "Time Step": self.model.schedule.steps,
        "Instance ID": self.id,
        "Available": self._available,
        "Server": self.server.id if self.server else None,
        "Being Provisioned": self.being_provisioned,
        "Latency": latency,
        "Latency SLA": latency_sla,
        "Is Violating Latency SLA": 1 if latency > latency_sla else 0,
        "Provisioning Time SLA": provisioning_time_sla,
    }
    return metrics


def container_registry_collect(self) -> dict:
    """Method that collects a set of metrics for the object.

    Returns:
        metrics (dict): Object metrics.
    """
    time_to_usefulness = None if self.first_usage_time_step is None else self.first_usage_time_step - self.provisioning_final_time_step
    metrics = {
        "Time Step": self.model.schedule.steps,
        "Instance ID": self.id,
        "Available": self.available,
        "CPU/RAM Demand": [self.cpu_demand, self.memory_demand],
        "Disk Demand": self.server.disk_demand if self.server else None,
        "Provisioning Time": self.provisioning_time,
        "When Got Provisioned": self.provisioning_final_time_step,
        "First Usage": self.first_usage_time_step,
        "Time to Usefulness": time_to_usefulness,
        "Server": self.server.id if self.server else None,
    }
    return metrics


def network_flow_step(self):
    """Method that executes the events involving the object at each time step."""
    if self.status == "active":
        # Updating the flow progress according to the available bandwidth
        if len(self.bandwidth.values()) > 0 and not any([bw == None for bw in self.bandwidth.values()]):
            actual_bandwidth = calculate_bandwidth(
                tick_duration=self.model.tick_duration,
                topology=self.topology,
                path=self.path,
                minimum_link_bandwidth=min(self.bandwidth.values()),
            )
            self.data_to_transfer -= actual_bandwidth

        if self.data_to_transfer <= 0:
            # Updating the completed flow's properties
            self.data_to_transfer = 0

            # Storing the current step as when the flow ended
            self.end = self.model.schedule.steps + 1

            # Updating the flow status to "finished"
            self.status = "finished"

            # Releasing links used by the completed flow
            for i in range(0, len(self.path) - 1):
                link = self.model.topology[self.path[i]][self.path[i + 1]]
                link["active_flows"].remove(self)

            # When container layer flows finish: Adds the container layer to its target host
            if self.metadata["type"] == "layer":
                # Removing the flow from its target host's download queue
                self.target.download_queue.remove(self)

                # Adding the layer to its target host
                layer = self.metadata["object"]
                layer.server = self.target
                self.target.container_layers.append(layer)

            # When service state flows finish: change the service migration status
            elif self.metadata["type"] == "service_state":
                service = self.metadata["object"]
                service._Service__migrations[-1]["status"] = "finished"


def edge_server_step(self):
    """Method that executes the events involving the object at each time step."""
    while len(self.waiting_queue) > 0 and len(self.download_queue) < self.max_concurrent_layer_downloads:
        layer = self.waiting_queue.pop(0)

        # Gathering the list of registries that have the layer
        registries_with_layer = []
        for registry in [reg for reg in ContainerRegistry.all() if reg.available]:
            # Checking if the registry is hosted on a valid host in the infrastructure and if it has the layer we need to pull
            if registry.server and any(layer.digest == l.digest for l in registry.server.container_layers):
                # Selecting a network path to be used to pull the layer from the registry
                path = get_shortest_path(
                    origin_switch=registry.server.base_station.network_switch,
                    target_switch=self.base_station.network_switch,
                )

                registries_with_layer.append({"object": registry, "path": path})

        # Selecting the registry from which the layer will be pulled to the (target) edge server
        registries_with_layer = sorted(registries_with_layer, key=lambda r: len(r["path"]))
        registry = registries_with_layer[0]["object"]
        path = registries_with_layer[0]["path"]

        # Storing the first time the selected registry was used to pull a layer (excluding layers pulled to provision a registry)
        if registry.first_usage_time_step is None and "registry" not in layer.instruction and not hasattr(layer, "used_to_provision_a_registry"):
            registry.first_usage_time_step = registry.model.schedule.steps + 1

        # Creating the flow object
        flow = NetworkFlow(
            topology=self.model.topology,
            source=registry.server,
            target=self,
            start=self.model.schedule.steps + 1,
            path=path,
            data_to_transfer=layer.size,
            metadata={"type": "layer", "object": layer, "container_registry": registry},
        )
        self.model.initialize_agent(agent=flow)

        # Adding the created flow to the edge server's download queue
        self.download_queue.append(flow)


def service_step(self):
    """Method that executes the events involving the object at each time step."""
    if len(self._Service__migrations) > 0 and self._Service__migrations[-1]["end"] == None:
        migration = self._Service__migrations[-1]

        # Gathering information about the service's image
        image = ContainerImage.find_by(attribute_name="digest", attribute_value=self.image_digest)

        # Gathering layers present in the target server (layers, download_queue, waiting_queue)
        layers_downloaded = [l for l in migration["target"].container_layers if l.digest in image.layers_digests]
        layers_on_download_queue = [
            flow.metadata["object"] for flow in migration["target"].download_queue if flow.metadata["object"].digest in image.layers_digests
        ]

        # Setting the migration status to "pulling_layers" once any of the service layers start being downloaded
        if migration["status"] == "waiting":
            layers_on_target_server = layers_downloaded + layers_on_download_queue

            if len(layers_on_target_server) > 0:
                migration["status"] = "pulling_layers"

        if migration["status"] == "pulling_layers" and len(image.layers_digests) == len(layers_downloaded):
            # Once all the layers that compose the service's image are pulled, the service container is deprovisioned on its
            # origin host even though it still is in there (that's why it is still on the origin's services list). This action
            # is only taken in case the current provisioning process regards a migration.
            if self.server:
                self.server.cpu_demand -= self.cpu_demand
                self.server.memory_demand -= self.memory_demand

            # Once all service layers have been pulled, creates a ContainerImage object representing
            # the service image on the target host if that host didn't already have such image
            if not any([image.digest == self.image_digest for image in migration["target"].container_images]):
                # Finding similar image provisioned on the infrastructure to get metadata from it when creating the new image
                template_image = ContainerImage.find_by(attribute_name="digest", attribute_value=self.image_digest)
                if template_image is None:
                    raise Exception(f"Could not find any container image with digest: {self.image_digest}")

                # Creating the new image on the target host
                image = ContainerImage()
                image.name = template_image.name
                image.digest = template_image.digest
                image.tag = template_image.tag
                image.architecture = template_image.architecture
                image.layers_digests = template_image.layers_digests

                self.model.initialize_agent(agent=image)

                # Connecting the new image to the target host
                image.server = migration["target"]
                migration["target"].container_images.append(image)

            if self.state == 0 or self.server == None:
                # Stateless Services: migration is set to finished immediately after layers are pulled
                migration["status"] = "finished"
            else:
                # Stateful Services: state must be migrated to the target host after layers are pulled
                migration["status"] = "migrating_service_state"

                # Services are unavailable during the period where their states are being migrated
                self._available = False

                # Selecting the path that will be used to transfer the service state
                path = get_shortest_path(
                    origin_switch=self.server.base_station.network_switch,
                    target_switch=migration["target"].base_station.network_switch,
                )

                # Creating network flow representing the service state that will be migrated to its target host
                flow = NetworkFlow(
                    topology=self.model.topology,
                    source=self.server,
                    target=migration["target"],
                    start=self.model.schedule.steps + 1,
                    path=path,
                    data_to_transfer=self.state,
                    metadata={"type": "service_state", "object": self},
                )
                self.model.initialize_agent(agent=flow)

        # Incrementing the migration time metadata
        if migration["status"] == "waiting":
            migration["waiting_time"] += 1
        elif migration["status"] == "pulling_layers":
            migration["pulling_layers_time"] += 1
        elif migration["status"] == "migrating_service_state":
            migration["migrating_service_state_time"] += 1

        if migration["status"] == "finished":
            # Storing when the migration has finished
            migration["end"] = self.model.schedule.steps + 1

            # Updating the service's origin server metadata
            if self.server:
                self.server.services.remove(self)
                self.server.ongoing_migrations -= 1

            # Updating the service's target server metadata
            self.server = migration["target"]
            self.server.services.append(self)
            self.server.ongoing_migrations -= 1

            # Tagging the service as available once their migrations finish
            self._available = True
            self.being_provisioned = False

            # Changing the routes used to communicate the application that owns the service to its users
            app = self.application
            user = app.users[0]
            user.set_communication_path(app)

            # Accounting for the provisioning time SLA violations
            if not hasattr(self, "provisioning_time_sla_violations"):
                self.provisioning_time_sla_violations = 0

            migration_time = migration["end"] - migration["start"]
            if migration_time > user.provisioning_time_slas[str(app.id)]:
                self.provisioning_time_sla_violations += 1


def container_registry_step(self):
    """Method that executes the events involving the object at each time step."""
    if self.available is False:
        # Gathering a template registry container image
        registry_image = ContainerImage.find_by(attribute_name="name", attribute_value="registry")

        # Checking if the host has the container layers that compose the container registry image
        layers_hosted_by_server = 0
        for layer_digest in registry_image.layers_digests:
            if any([layer_digest == layer.digest for layer in self.server.container_layers]):
                layers_hosted_by_server += 1

        # Checking if the host has the container registry image
        server_images_digests = [image.digest for image in self.server.container_images]
        if layers_hosted_by_server == len(registry_image.layers_digests) and registry_image.digest not in server_images_digests:
            self.server._add_container_image(template_container_image=registry_image)

        # Updating registry's availability status if its provisioning process has ended
        if self.available is False and registry_image.digest in [image.digest for image in self.server.container_images]:
            self.provisioning_final_time_step = self.model.schedule.steps + 1
            self.provisioning_time = self.provisioning_final_time_step - self.provisioning_initial_time_step
            self.available = True


def is_hosted_by_other_registry(registry: object, object_being_analyzed: object, object_type: str) -> bool:
    """Checks if a given object (a container image or a container layer) is hosted by a registry other than the given one.

    Args:
        registry (object): Registry being analyzed.
        object_being_analyzed (object): Object being analyzed.
        object_type (str): Object type. Valid options: "container_image" and "container_layer".

    Returns:
        (bool): Whether the object is hosted by a registry other than the given one or not.
    """
    for other_registry in ContainerRegistry.all():
        if other_registry != registry and other_registry.available and other_registry.server != registry.server:
            if object_type == "container_image":
                if any(object_being_analyzed.digest == other_image.digest for other_image in other_registry.server.container_images):
                    return True
            elif object_type == "container_layer":
                if any(object_being_analyzed.digest == other_layer.digest for other_layer in other_registry.server.container_layers):
                    return True

    return False


def container_registry_deprovision(self, purge_images: bool = False):
    """Deprovisions a container registry, releasing the allocated resources on its host server.

    Args:
        purge_images (bool, optional): Removes all container images and associated layers from the server. Defaults to False.
    """
    # Checking if the registry has a host server and if is not currently being used to pull any container layer
    flows_using_the_registry = [
        flow for flow in NetworkFlow.all() if flow.status == "active" and flow.metadata["type"] == "layer" and flow.source == self.server
    ]
    all_images_are_hosted_by_other_registries = all(
        is_hosted_by_other_registry(
            registry=self,
            object_being_analyzed=image,
            object_type="container_image",
        )
        for image in self.server.container_images
    )
    all_layers_are_hosted_by_other_registries = all(
        is_hosted_by_other_registry(
            registry=self,
            object_being_analyzed=layer,
            object_type="container_layer",
        )
        for layer in self.server.container_layers
    )
    if self.server and len(flows_using_the_registry) == 0 and all_images_are_hosted_by_other_registries and all_layers_are_hosted_by_other_registries:
        # Removing unused container images and associated layers from the server if the "purge_images" flag is True
        if purge_images:
            # Gathering the list of unused container images and layers
            unused_images = list(self.server.container_images)
            unused_layers = list(self.server.container_layers)
            for service in Service.all():
                service_target = service._Service__migrations[-1]["target"] if len(service._Service__migrations) > 0 else None
                if service.server == self.server or service_target == self.server:
                    image = next(
                        (img for img in unused_images if img.digest == service.image_digest),
                        None,
                    )
                    if image is not None:
                        # Removing the used image from the "unused_images" list
                        unused_images.remove(image)

                        # Removing used layers from the "unused_layers" list
                        for layer in unused_layers:
                            if layer.digest in image.layers_digests:
                                unused_layers.remove(layer)

            # Removing unused images
            for image in unused_images:
                # Removing the unused image from its host
                image.server.container_images.remove(image)
                image.server = None

                # Removing the unused image from the simulator's agent list and from its class instance list
                image.model.schedule.remove(image)
                image.__class__._instances.remove(image)

            # Removing unused layers
            for layer in unused_layers:
                # Removing the unused layer from its host
                layer.server.disk_demand -= layer.size
                layer.server.container_layers.remove(layer)
                layer.server = None

                # Removing the unused layer from the simulator's agent list and from its class instance list
                layer.model.schedule.remove(layer)
                layer.__class__._instances.remove(layer)

                break

        # Removing relationship between the registry and its server
        self.server.container_registries.remove(self)
        self.server.memory_demand -= self.memory_demand
        self.server.cpu_demand -= self.cpu_demand
        self.server = None

        # Removing the registry
        self.model.schedule.remove(self)
        self.__class__._instances.remove(self)

        return True

    return False
