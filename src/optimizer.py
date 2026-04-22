import torch
import torch_optimizer as torch_opt

def build_optimizer(name, model, lr, wd):
    """
    Integration Test: Function to initialise the requested optimizer.
    Filters for trainable_params to ensure only unfrozen layers receive gradient updates
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Standard Stochastic Gradient with Nesterov Momentum
    if name == "SGD":
        return torch.optim.SGD(
            trainable_params,
            lr=lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=wd
        )

    # Adaptive Moment Estimation
    if name == "Adam":
        return torch.optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=wd
        )

    # Switching from Adam to SGD 
    if name == "SWATS":
        return torch_opt.SWATS(
            trainable_params,
            lr=lr,
            weight_decay=wd
        )

    # System Error Handling
    raise ValueError(f"Unknown optimizer: {name}")


def get_swats_phase(optimizer):
    """
    Extracts the current phase of SWATS (Adam or SGD)
    """
    try:
        return optimizer.param_groups[0]["phase"]
    except Exception:
        return "UNKNOWN"
    