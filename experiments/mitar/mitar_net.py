import sys

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from data_modules.datasets.transforms.pad import Pad
from data_modules.interaction_data_module import InteractionDataModule
from lightning_modules.mitar.mitar_net import MitarNetLM

# config = dotenv_values("neptune_config.env")
#
# neptune_logger = NeptuneLogger(
#     project=config["NEPTUNE_PROJECT"],
#     api_token=config["NEPTUNE_API_TOKEN"],
#     log_model_checkpoints=False,
# )

config_path, data_split_seed, lr, batch_size, epochs_num, model_dir = sys.argv[1:]
config_name = config_path.split('/')[-1].split('.')[0]

data_module = InteractionDataModule(config_path, int(batch_size), int(data_split_seed))
x_key, y_key = data_module.get_batch_keys()

n_embeddings = sum([transform.target_length for transform in data_module.dataset_config.transform.transforms if
                    type(transform) is Pad])

module = MitarNetLM(6, n_embeddings, float(lr), x_key, y_key)

checkpoint_callback = ModelCheckpoint(dirpath=model_dir,
                                      filename="mitar_net_{}_{}_{}_{}".format(config_name, batch_size, data_split_seed,
                                                                              lr), save_top_k=1, monitor="val_loss")

trainer = Trainer(gpus=1, max_epochs=int(epochs_num),
                  callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=100), checkpoint_callback],
                  # logger=neptune_logger
                  )
trainer.fit(module, datamodule=data_module)
