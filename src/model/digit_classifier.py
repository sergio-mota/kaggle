import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics

from src.model import utilities


class Model(torch.nn.Module):
    def __init__(
            self,
            feature_size: list[int],
            feature_map_sizes: list[int],
            conv_kernel_sizes: list[int],
            conv_dropout_probs: list[float],
            pool_kernel_sizes: list[int],
            pool_strides: list[int],
            fc_sizes: list[int],
            fc_dropout_probs: list[float],
        ):
        super().__init__()

        if not (
            len(feature_map_sizes) == 
            len(conv_kernel_sizes) == 
            len(conv_dropout_probs) == 
            len(pool_kernel_sizes) == 
            len(pool_strides)
        ):
            raise ValueError("convolutional layers must have the same length")
        if not (len(fc_sizes) == len(fc_dropout_probs)):
            raise ValueError("fully connected layers must have the same length")

        conv_layers = []
        for i in range(len(feature_map_sizes)):
            conv_layers.append(torch.nn.Conv2d(
                in_channels=feature_map_sizes[i-1] if i > 0 else 1,
                out_channels=feature_map_sizes[i],
                kernel_size=conv_kernel_sizes[i],
                bias=False
            ))
            conv_layers.append(torch.nn.BatchNorm2d(feature_map_sizes[i]))
            conv_layers.append(torch.nn.ReLU())
            conv_layers.append(torch.nn.Dropout(p=conv_dropout_probs[i]))
            conv_layers.append(torch.nn.MaxPool2d(
                kernel_size=pool_kernel_sizes[i],
                stride=pool_strides[i]
            ))

            feature_size = [(x - conv_kernel_sizes[i] + 1) // pool_strides[i] for x in feature_size]

        fc_layers = []
        for i in range(len(fc_sizes)):
            fc_layers.append(torch.nn.Linear(
                in_features=fc_sizes[i-1] if i > 0 else feature_map_sizes[-1] * feature_size[0] * feature_size[1],
                out_features=fc_sizes[i],
                bias=i==(len(fc_sizes) - 1)
            ))
            if i < len(fc_sizes) - 1:
                fc_layers.append(torch.nn.BatchNorm1d(fc_sizes[i]))
                fc_layers.append(torch.nn.ReLU())
                fc_layers.append(torch.nn.Dropout(p=fc_dropout_probs[i]))
        self.all_layers = torch.nn.Sequential(
            *(conv_layers + [torch.nn.Flatten()] + fc_layers)
        )

    def forward(self, x):
        logits = self.all_layers(x)
        return logits

class LitModel(L.LightningModule):
    def __init__(
            self,
            model_cfg : dict,
            num_classes : int,
            learning_rate : float,
            *,
            weight_decay : float = 0.0,
            eta_min : float = 1e-6
        ):
        super().__init__()
        self.save_hyperparameters()

        self.model = Model(**model_cfg)
        self.model.apply(utilities.initialize_weights)

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self._shared_step(batch)
        self.train_acc(predicted_labels, labels)
        self.log(
            "loss/train",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False
        )
        self.log(
            "accuracy/train",
            self.train_acc,
            prog_bar=True,
            on_epoch=True,
            on_step=False
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self._shared_step(batch)
        self.val_acc(predicted_labels, labels)
        self.log(
            "loss/val",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False
        )
        self.log(
            "accuracy/val",
            self.val_acc,
            prog_bar=True,
            on_epoch=True,
            on_step=False
        )

    def test_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, labels)
        self.log(
            "loss/test",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False
        )
        self.log(
            "accuracy/test",
            self.test_acc,
            prog_bar=True,
            on_epoch=True,
            on_step=False
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=self.hparams.eta_min
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
