import torch


def initialize_weights(m):
    classname = m.__class__.__name__
    
    # Linear layers
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.01) # Small constant for bias
        print(f"Initialized {classname} with Xavier Uniform for weights and 0.01 for bias.")
    # Convolutional layers
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
        print(f"Initialized {classname} with Kaiming Normal for weights and 0 for bias.")
