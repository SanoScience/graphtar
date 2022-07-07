from pytorch_lightning import Trainer

from data_modules.datasets.transforms.pad import Pad
from data_modules.interaction_data_module import InteractionDataModule
from lightning_modules.mitar.mitar_net import MitarNetLM

data_module = InteractionDataModule("../../data_modules/configs/mitar_config.json", 128)
x_key, y_key = data_module.get_batch_keys()

module = MitarNetLM(6,
                    sum([transform.target_length for transform in data_module.dataset_config['transform'].transforms if
                         type(transform) is Pad]), x_key, y_key)
trainer = Trainer(accelerator='gpu', max_epochs=2, limit_train_batches=0.1, limit_test_batches=0.1)
trainer.fit(module, datamodule=data_module)
