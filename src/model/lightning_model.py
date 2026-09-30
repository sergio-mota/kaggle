import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics


class Classifier(L.LightningModule):
    def __init__(
            self,
            model : torch.nn.Module,
            num_classes : int,
            learning_rate : float,
            *,
            weight_decay : float = 0.0
        ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["model"])

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

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
        self.log("loss/train", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("accuracy/train", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self._shared_step(batch)
        self.val_acc(predicted_labels, labels)
        self.log("loss/val", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("accuracy/val", self.val_acc, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, labels)
        self.log("loss/test", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("accuracy/test", self.test_acc, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer