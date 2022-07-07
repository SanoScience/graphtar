from pytorch_lightning import Trainer

from data_modules.interaction_data_module import InteractionDataModule
from lightning_modules.deepmirtar.ann import AnnLM

data_module = InteractionDataModule("../../data_modules/configs/deepmirtar_config.json")
x_key, y_key = data_module.get_batch_keys()

module = AnnLM("autoencoder.pt", x_key, y_key)

trainer = Trainer(accelerator='gpu', max_epochs=2, limit_train_batches=0.1, limit_test_batches=0.1)
trainer.fit(module, datamodule=data_module)
