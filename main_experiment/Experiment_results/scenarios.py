from __future__ import annotations


def node_type(node: int, num_critical: int = 3) -> str:
    if node == 0:
        return "implant10"
    if 1 <= node < num_critical:
        return "implant30"
    return "surface"


def is_critical_node(node: int, num_critical: int = 3) -> bool:
    return node < num_critical


def rest_label(node: int, num_critical: int = 3) -> str:
    cls = node_type(node, num_critical)
    if cls == "implant10":
        return "implant10_rest"
    if cls == "implant30":
        return "implant30_rest"
    return "surface_rest"


def stress_label(node: int, num_critical: int = 3) -> str:
    cls = node_type(node, num_critical)
    if cls == "implant10":
        return "implant10_stress"
    if cls == "implant30":
        return "implant30_stress"
    return "surface_contact_failure"


def condition_name(t: int) -> str:
    if t < 300:
        return "rest"
    if t < 600:
        return "surface_sweat"
    if t < 900:
        return "surface_moderate_loose"
    return "recovery_rest"


def condition_switch_labels(t: int, num_nodes: int, num_critical: int = 3) -> list[str]:
    labels = [rest_label(node, num_critical) for node in range(num_nodes)]
    if 300 <= t < 600:
        for node in range(num_nodes):
            if node_type(node, num_critical) == "surface":
                labels[node] = "surface_sweat"
    elif 600 <= t < 900:
        for node in range(num_nodes):
            if node_type(node, num_critical) == "surface":
                labels[node] = "surface_moderate_loose"
    return labels


def stress_window_labels(
    t: int,
    num_nodes: int,
    *,
    affected_nodes: list[int] | tuple[int, ...],
    start: int,
    end: int,
    num_critical: int = 3,
) -> list[str]:
    labels = [rest_label(node, num_critical) for node in range(num_nodes)]
    if start <= t < end:
        for node in affected_nodes:
            labels[node] = stress_label(node, num_critical)
    return labels


def homogeneous_surface_labels(t: int, num_nodes: int) -> list[str]:
    if 300 <= t < 600:
        label = "surface_sweat"
    elif 600 <= t < 900:
        label = "surface_moderate_loose"
    else:
        label = "surface_rest"
    return [label for _ in range(num_nodes)]
