import sys

import torch
from pytorch_lightning import Trainer

from data_modules.interaction_data_module import InteractionDataModule
from lightning_modules.deepmirtar.sd_autoencoder import SdAutoencoderLM
from lightning_modules.models.deepmirtar.denoising_autoencoder import DenoisingAutoencoder

config_path, data_split_seed, lr, batch_size, epochs_num = sys.argv
config_name = config_path.split('/')[-1].split('.')[0]
# config_path = "../../data_modules/configs/deepmirtar_config.json"


data_module = InteractionDataModule(config_path, int(batch_size), int(data_split_seed))
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
        module = SdAutoencoderLM(autoencoders[i], x_key, y_key, float(lr))
    else:
        module.append_module(autoencoder)
    trainer = Trainer(accelerator='gpu', max_epochs=epochs_num)
    trainer.fit(module, datamodule=data_module)
torch.save(module.sda, "models/autoencoder_{}_{}_{}_{}.pt".format(config_name, batch_size, data_split_seed, lr))
