import torch
from pytorch_lightning import Trainer

from data_modules.interaction_data_module import InteractionDataModule
from lightning_modules.deepmirtar.sd_autoencoder import SdAutoencoderLM
from lightning_modules.models.deepmirtar.denoising_autoencoder import DenoisingAutoencoder

data_module = InteractionDataModule("../../data_modules/configs/deepmirtar_config.json", 128)
x_key, y_key = data_module.get_batch_keys()

autoencoders = [
    DenoisingAutoencoder(474, 1500),
    DenoisingAutoencoder(1500, 2000),
    DenoisingAutoencoder(2000, 2000),
    DenoisingAutoencoder(2000, 2000),
    DenoisingAutoencoder(2000, 1500)
]

for i, autoencoder in enumerate(autoencoders):
    if i == 0:
        module = SdAutoencoderLM(autoencoders[i], x_key, y_key)
    else:
        module.append_module(autoencoder)
    trainer = Trainer(accelerator='gpu', max_epochs=2, limit_train_batches=0.1, limit_test_batches=0.1)
    trainer.fit(module, datamodule=data_module)
torch.save(module.sda, "autoencoder.pt")
