import sys

from dotenv import dotenv_values
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from data_modules.interaction_data_module import InteractionDataModule
from lightning_modules.deepmirtar.ann import AnnLM

config = dotenv_values("neptune_config.env")

neptune_logger = NeptuneLogger(
    project=config["NEPTUNE_PROJECT"],
    api_token=config["NEPTUNE_API_TOKEN"],
    log_model_checkpoints=False,
)

config_path, data_split_seed, lr, batch_size, epochs_num, model_dir = sys.argv[1:]
config_name = config_path.split('/')[-1].split('.')[0]
autoencoder_path = "{}/autoencoder_{}_{}_{}_{}.pt".format(model_dir, config_name, batch_size, data_split_seed, lr)

hyperparams = {
    "config_path": config_path,
    "data_split_seed": data_split_seed,
    "lr": lr,
    "batch_size": batch_size,
    "epochs_num": epochs_num,
    "model_dir": model_dir,
    "autoencoder_path": autoencoder_path
}
neptune_logger.log_hyperparams(hyperparams)

data_module = InteractionDataModule(config_path, int(batch_size), int(data_split_seed))
x_key, y_key = data_module.get_batch_keys()

module = AnnLM(autoencoder_path, x_key, y_key, float(lr))
checkpoint_callback = ModelCheckpoint(dirpath=model_dir,
                                      filename="deepmirtar_ann_{}_{}_{}_{}".format(config_name, batch_size,
                                                                                   data_split_seed,
                                                                                   lr), save_top_k=1,
                                      monitor="val_loss")
trainer = Trainer(gpus=1, max_epochs=int(epochs_num),
                  callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=100), checkpoint_callback],
                  logger=neptune_logger
                  )
trainer.fit(module, datamodule=data_module)
