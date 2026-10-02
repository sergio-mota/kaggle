import torch
from matplotlib import pyplot as plt



def plot_batchnorm_activations(
        model : torch.nn.Module,
        activations : dict[str, torch.Tensor]
    ):
    batchnorm_activations = [
        (i, layer, activation)
        for i, (layer, activation) in enumerate(
            zip(model.all_layers, activations.values())
        )
        if isinstance(layer, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d))
    ]

    _, axs = plt.subplots(
        1,
        len(batchnorm_activations),
        figsize=(15, 4),
        constrained_layout=True
    )

    for ax, (i, _, activation) in zip(axs, batchnorm_activations):
        ax.hist(
            activation.flatten(),
            bins=100,
            density=True
        )
        ax.set_title(f"Layer {i}")
        ax.set_xlabel("Activation")
        ax.set_ylabel("Density")

        print(
            f"Layer {i}: "
            f"mean={activation.mean():.3f}, "
            f"std={activation.std():.3f}"
        )

    plt.show()

def plot_linear_activations(
        model : torch.nn.Module,
        activations : dict[str, torch.Tensor]
    ):
    linear_activations = [
        (i, layer, activation)
        for i, (layer, activation) in enumerate(
            zip(model.all_layers, activations.values())
        )
        if isinstance(layer, torch.nn.Linear)
    ]

    _, axs = plt.subplots(
        1,
        len(linear_activations),
        figsize=(15, 4),
        constrained_layout=True
    )

    for ax, (i, _, activation) in zip(axs, linear_activations):
        ax.hist(
            activation.flatten(),
            bins=100,
            density=True
        )
        ax.set_title(f"Layer {i}")
        ax.set_xlabel("Activation")
        ax.set_ylabel("Density")

        print(
            f"Layer {i}: "
            f"mean={activation.mean():.3f}, "
            f"std={activation.std():.3f}"
        )

    plt.show()

def plot_relu_activations(
        model : torch.nn.Module,
        activations : dict[str, torch.Tensor]
    ):
    relu_activations = [
        (i, layer, activation)
        for i, (layer, activation) in enumerate(
            zip(model.all_layers, activations.values())
        )
        if isinstance(layer, torch.nn.ReLU)
    ]

    _, axs = plt.subplots(
        1,
        len(relu_activations),
        figsize=(15, 4),
        constrained_layout=True
    )

    for ax, (i, _, activation) in zip(axs, relu_activations):
        ax.hist(
            activation.flatten(),
            bins=100,
            density=True
        )
        ax.set_title(f"Layer {i}")
        ax.set_xlabel("Activation")
        ax.set_ylabel("Density")
        ax.set_yscale("log")

        zero_fraction = (activation == 0).float().mean()
        print(
            f"Layer {i}: "
            f"mean={activation.mean():.3f}, "
            f"std={activation.std():.3f}, "
            f"zero={100*zero_fraction:.1f}%"
        )

    plt.show()