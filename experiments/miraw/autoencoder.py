import os
import sys

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from data_modules.datasets.transforms.pad import Pad
from data_modules.interaction_data_module import InteractionDataModule
from lightning_modules.miraw.autoencoder import AutoencoderLM

config_path, data_split_seed, lr, batch_size, epochs_num, model_dir = sys.argv[1:]
config_name = config_path.split('/')[-1].split('.')[0]

model_filename = "autoencoder_{}_{}_{}_{}.pt".format(config_name, batch_size, data_split_seed, lr)

data_module = InteractionDataModule(config_path, int(batch_size), int(data_split_seed))
x_key, y_key = data_module.get_batch_keys()

callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=50)]

input_size = 5 * sum([transform.target_length for transform in data_module.dataset_config.transform.transforms if
                      type(transform) is Pad])
module = AutoencoderLM(input_size, x_key, y_key, float(lr))
trainer = Trainer(gpus=1, max_epochs=int(epochs_num),
                  callbacks=callbacks,
                  logger=False
                  )
trainer.fit(module, datamodule=data_module)
torch.save(module.model, os.path.join(model_dir, model_filename))
