""" Contains the Never Follow heuristic.

This strategy makes no service and registry migration decisions regardless of
ongoing events (e.g., user mobility, SLA violations, etc.) within the infrastructure.

This strategy is based on the VMs-Never-migrate presented in the reference below.

==== REFERENCE ====
Yao, H., Bai, C., Zeng, D., Liang, Q., & Fan, Y. (2015). Migrate or not? Exploring virtual machine migration in
roadside cloudlet‐based vehicular cloud. Concurrency and Computation: Practice and Experience, 27(18), 5780-5792.

Link: https://doi.org/10.1002/cpe.3642
"""


def never_follow(parameters: dict = {}):
    """Simple strategy that performs no migration decision regardless ongoing events (e.g., user mobility, SLA violations, etc.).

    Args:
        parameters (dict, optional): User-defined parameters. Defaults to {}.
    """
    ...
