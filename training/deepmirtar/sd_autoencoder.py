import torch
import torch.nn as nn
from pytorch_lightning import Trainer

from data_modules.interaction_data_module import InteractionDataModule
from lightning_modules.deepmirtar.sd_autoencoder import SdAutoencoderLM
from lightning_modules.models.deepmirtar.DenoisingAutoencoder import DenoisingAutoencoder

data_module = InteractionDataModule("../../data_modules/configs/miraw_config.json")
x_key, y_key = data_module.get_batch_keys()


autoencoders = [DenoisingAutoencoder(350, 400), DenoisingAutoencoder(400, 450), DenoisingAutoencoder(450, 500)]

module = SdAutoencoderLM(autoencoders[0], x_key, y_key)

trainer = Trainer(accelerator='gpu', max_epochs=2, limit_train_batches=0.1, limit_test_batches=0.1)
trainer.fit(module, datamodule=data_module)
torch.save(module, "autoencoder.pt")
