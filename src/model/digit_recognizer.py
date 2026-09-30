import torch


class DigitRecognizer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st convolution
            torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # 2nd convolution
            torch.nn.Conv2d(6, 20, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # 1st fully connected layer
            torch.nn.Flatten(),
            torch.nn.Linear(20 * 4 * 4, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),

            # 2nd fully connected layer
            torch.nn.Linear(100, 10)
        )

    def forward(self, x):
        logits = self.all_layers(x)
        return logits