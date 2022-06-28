import torch
from pytorch_lightning import Trainer

from data_modules.interaction_data_module import InteractionDataModule
from lightning_modules.miraw.ann import AnnLM

data_module = InteractionDataModule("../../data_modules/configs/miraw_config.json")
x_key, y_key = data_module.get_batch_keys()

autoencoder = torch.load("autoencoder.pt")
module = AnnLM(x_key, y_key, encoder=autoencoder.model.encoder)
module.model.encoder.requires_grad_(False)
trainer = Trainer(accelerator='gpu', max_epochs=2, limit_train_batches=0.1, limit_test_batches=0.1)
trainer.fit(module, datamodule=data_module)
