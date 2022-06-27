import torch
from pytorch_lightning import Trainer

from data_modules.interaction_data_module import InteractionDataModule
from lightning_modules.miraw.miraw_autoencoder import MirawAutoencoderLM

data_module = InteractionDataModule("../data_modules/configs/miraw_config.json")
x_key, y_key = data_module.get_batch_keys()
model = MirawAutoencoderLM(x_key, y_key)
trainer = Trainer(accelerator='gpu', max_epochs=10, limit_train_batches=0.1, limit_test_batches=0.1)
trainer.fit(model, datamodule=data_module)
torch.save(model, "autoencoder.pt")
