from __future__ import annotations


def linear_parameters(input_dim: int, output_dim: int) -> int:
    return int(input_dim) * int(output_dim) + int(output_dim)


def layer_norm_parameters(hidden_dim: int) -> int:
    return int(hidden_dim) * 2


def transolver_microenvironment_active_parameters(
    hidden_dim: int,
    depth: int,
    head_count: int,
    slice_count: int,
    coordinate_dim: int,
    species_count: int,
    control_dim: int,
    context_dim: int,
    expert_count: int,
    top_k: int,
    expert_hidden_multiplier: int,
    physics_mlp_multiplier: int,
) -> int:
    embeddings = (
        linear_parameters(species_count, hidden_dim) * 3
        + linear_parameters(control_dim, hidden_dim)
        + linear_parameters(context_dim, hidden_dim)
        + linear_parameters(coordinate_dim, hidden_dim)
        + linear_parameters(1, hidden_dim) * 3
        + hidden_dim
    )
    head_dim = int(hidden_dim) // int(head_count)
    attention_projection_parameters = (
        linear_parameters(hidden_dim, hidden_dim)
        + linear_parameters(head_dim, slice_count)
        + linear_parameters(head_dim, slice_count)
        + linear_parameters(slice_count, 1)
        + head_count
        + linear_parameters(head_dim, head_dim) * 3
        + linear_parameters(hidden_dim, hidden_dim)
    )
    per_block = (
        attention_projection_parameters
        + layer_norm_parameters(hidden_dim) * 2
        + linear_parameters(hidden_dim, hidden_dim * 2)
        + linear_parameters(hidden_dim, hidden_dim * physics_mlp_multiplier)
        + linear_parameters(hidden_dim * physics_mlp_multiplier, hidden_dim)
    )
    active_expert_parameters = int(top_k) * (
        layer_norm_parameters(hidden_dim)
        + linear_parameters(hidden_dim, hidden_dim * expert_hidden_multiplier)
        + linear_parameters(hidden_dim * expert_hidden_multiplier, hidden_dim)
    )
    router_parameters = linear_parameters(hidden_dim, expert_count)
    heads = linear_parameters(hidden_dim, hidden_dim) + linear_parameters(hidden_dim, species_count)
    return int(embeddings + depth * per_block + router_parameters + active_expert_parameters + heads)


def neuronsifter_world_action_active_parameters(
    hidden_dim: int,
    depth: int,
    head_count: int,
    mechanism_token_count: int,
    coordinate_dim: int,
    species_count: int,
    control_dim: int,
    context_dim: int,
    expert_count: int,
    top_k: int,
    expert_hidden_multiplier: int,
    dit_mlp_multiplier: int,
    operator_splitting_bottleneck_divisor: int,
) -> int:
    embeddings = (
        linear_parameters(species_count, hidden_dim) * 3
        + linear_parameters(control_dim, hidden_dim)
        + linear_parameters(context_dim, hidden_dim)
        + linear_parameters(coordinate_dim, hidden_dim)
        + linear_parameters(1, hidden_dim) * 2
        + linear_parameters(hidden_dim, hidden_dim) * 3
        + mechanism_token_count * hidden_dim
    )
    attention_parameters = linear_parameters(hidden_dim, hidden_dim) * 8
    adaptive_norm_parameters = (layer_norm_parameters(hidden_dim) + linear_parameters(hidden_dim, hidden_dim * 3)) * 3
    feed_forward_parameters = (
        linear_parameters(hidden_dim, hidden_dim * dit_mlp_multiplier)
        + linear_parameters(hidden_dim * dit_mlp_multiplier, hidden_dim)
    )
    active_expert_parameters = int(top_k) * (
        layer_norm_parameters(hidden_dim)
        + linear_parameters(hidden_dim, hidden_dim * expert_hidden_multiplier)
        + linear_parameters(hidden_dim * expert_hidden_multiplier, hidden_dim)
    )
    router_parameters = linear_parameters(hidden_dim, expert_count)
    splitting_bottleneck_dim = max(int(hidden_dim) // int(operator_splitting_bottleneck_divisor), 1)
    splitting_expert_parameters = int(expert_count) * (
        layer_norm_parameters(hidden_dim) * 3
        + linear_parameters(hidden_dim, splitting_bottleneck_dim) * 2
        + linear_parameters(splitting_bottleneck_dim, hidden_dim) * 2
        + linear_parameters(hidden_dim, hidden_dim)
    )
    splitting_router_parameters = linear_parameters(hidden_dim * 2, expert_count)
    shared_coupling_parameters = (
        layer_norm_parameters(hidden_dim)
        + linear_parameters(hidden_dim, splitting_bottleneck_dim)
        + linear_parameters(splitting_bottleneck_dim, hidden_dim)
    )
    operator_splitting_parameters = splitting_router_parameters + splitting_expert_parameters + shared_coupling_parameters
    per_block = attention_parameters + adaptive_norm_parameters + feed_forward_parameters + router_parameters + active_expert_parameters + operator_splitting_parameters
    heads = layer_norm_parameters(hidden_dim) + linear_parameters(hidden_dim, hidden_dim) + linear_parameters(hidden_dim, species_count)
    return int(embeddings + int(depth) * per_block + heads)


def validate_parameter_budget(parameter_count: int, lower_bound: int, upper_bound: int) -> None:
    if parameter_count < lower_bound or parameter_count > upper_bound:
        raise ValueError(
            f"Transolver active parameter estimate {parameter_count} outside required range [{lower_bound}, {upper_bound}]."
        )
