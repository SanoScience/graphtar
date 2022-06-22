import torch
from pytorch_lightning import Trainer

from data_modules.interaction_data_module import InteractionDataModule
from lightning_modules.miraw_ann import MirawANN

data_module = InteractionDataModule("../data/miRAW.csv", batch_size=128, train_val_ratio=(0.8, 0.2))
x_key, y_key = data_module.get_batch_keys()

autoencoder = torch.load("autoencoder.pt")
module = MirawANN(x_key, y_key, encoder=autoencoder.model.encoder)
module.model.encoder.requires_grad_(False)
trainer = Trainer(accelerator='gpu', max_epochs=100, limit_train_batches=0.1, limit_test_batches=0.1)
trainer.fit(module, datamodule=data_module)
