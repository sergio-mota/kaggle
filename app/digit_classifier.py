import os

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from src.dataset.digit import DigitDataModule
from src.model.digit_classifier import LitModel as DigitClassifier

if __name__ == "__main__":
    # Data parameters
    data_dir = "data/digit/"
    batch_size = 32
    num_workers = 0

    # Learning parameters
    num_epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-2
    eta_min = 1e-6

    # Model parameters
    num_classes = 10
    model_config = {
        "feature_size": [28, 28],
        "feature_map_sizes": [6, 20],
        "conv_kernel_sizes": [5, 5],
        "conv_dropout_probs": [0.1, 0.1],
        "pool_kernel_sizes": [2, 2],
        "pool_strides": [2, 2],
        "fc_sizes": [100, 10],
        "fc_dropout_probs": [0.2, 0.2]
    }

    torch.manual_seed(42)

    # Data
    dm = DigitDataModule(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Model
    lightning_model = DigitClassifier(
        model_config,
        num_classes,
        learning_rate,
        weight_decay=weight_decay,
        eta_min=eta_min
    )

    # Trainer
    callbacks = [
        ModelCheckpoint(
            save_top_k=1,
            mode="max",
            monitor="accuracy/val",
            save_last=True
        )
    ]
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices="auto",
        deterministic=True,
        callbacks=callbacks
    )
    trainer.fit(
        model=lightning_model,
        datamodule=dm
    )
    print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")

    # Load best model and make predictions
    best_model = DigitClassifier.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    prediction_batches = trainer.predict(
        model=best_model,
        datamodule=dm
    )
    predictions = torch.cat(
        [torch.argmax(logits, dim=1) for logits in prediction_batches]
    )

    # Prepare submission
    submission = pd.DataFrame({
        "ImageId": range(1, len(predictions) + 1),
        "Label": predictions.cpu().numpy()
    })
    submission_path = os.path.join(data_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Saved {len(predictions)} predictions to {submission_path}")
    