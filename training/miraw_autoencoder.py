from pytorch_lightning import Trainer

from data_modules.interaction_data_module import InteractionDataModule
from lightning_modules.miraw_autoencoder import MirawAutoencoderLM

data_module = InteractionDataModule("../data/miRAW.csv", batch_size=32, train_val_ratio=(0.8, 0.2))
x_key, y_key = data_module.get_batch_keys()
model = MirawAutoencoderLM(x_key, y_key)
trainer = Trainer()
trainer.fit(model, datamodule=data_module)
