import os

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from src.dataset.digit import DigitDataModule
from src.model.digit_recognizer import DigitRecognizer
from src.model.lightning_model import Classifier

if __name__ == "__main__":
    # Data parameters
    data_dir = "data/digit/"
    batch_size = 32
    num_workers = 0

    # Learning parameters
    learning_rate = 0.001
    weight_decay = 0.01
    num_epochs = 30

    torch.manual_seed(42)

    pytorch_model = DigitRecognizer()
    lightning_model = Classifier(pytorch_model, num_classes=10, learning_rate=learning_rate, weight_decay=weight_decay)

    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="max", monitor="accuracy/val", save_last=True)
    ]
    trainer = L.Trainer(max_epochs=num_epochs, accelerator="auto", devices="auto", deterministic=True, callbacks=callbacks)

    dm = DigitDataModule(data_dir, batch_size=batch_size, num_workers=num_workers)
    trainer.fit(model=lightning_model, datamodule=dm)

    best_model = Classifier.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        model=pytorch_model,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    prediction_batches = trainer.predict(model=best_model, datamodule=dm)
    predictions = torch.cat(
        [torch.argmax(logits, dim=1) for logits in prediction_batches]
    )

    submission = pd.DataFrame({
        "ImageId": range(1, len(predictions) + 1),
        "Label": predictions.cpu().numpy()
    })
    submission_path = os.path.join(data_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"Saved {len(predictions)} predictions to {submission_path}")
    
