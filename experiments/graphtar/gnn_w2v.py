import sys

from dotenv import dotenv_values
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from torchvision.transforms import Compose

from data_modules.datasets.transforms.to_tensor import ToTensor
from data_modules.datasets.transforms.word2vec.to_word2vec_embedding import ToWord2vecEmbedding
from data_modules.graph_interaction_data_module import GraphInteractionDataModule
from lightning_modules.graphtar.gnn import GnnLM
from lightning_modules.models.graphtar.gnn import LayerType, GlobalPoolingType

config = dotenv_values("neptune_config.env")

neptune_logger = NeptuneLogger(
    project=config["NEPTUNE_PROJECT"],
    api_token=config["NEPTUNE_API_TOKEN"],
    log_model_checkpoints=False,
)

config_path, gnn_layer_type, global_pooling, gnn_hidden_size, n_gnn_layers, fc_hidden_size, n_fc_layers, dropout_rate, data_split_seed, lr, batch_size, epochs_num, model_dir = sys.argv[
                                                                                                                                                                                1:]
config_name = config_path.split('/')[-1].split('.')[0]

hyperparams = {
    "gnn_hidden_size": int(gnn_hidden_size),
    "n_gnn_layers": int(n_gnn_layers),
    "fc_hidden_size": int(fc_hidden_size),
    "n_fc_layers": int(n_fc_layers),
    "dropout_rate": float(dropout_rate),
    "gnn_layer_type": gnn_layer_type,
    "global_pooling": global_pooling,
    "config_path": config_path,
    "data_split_seed": data_split_seed,
    "lr": float(lr),
    "batch_size": batch_size,
    "epochs_num": int(epochs_num),
    "model_dir": model_dir
}
neptune_logger.log_hyperparams(hyperparams)
neptune_logger.run['sys/tags'].add(["graphtar_architecture"])

w2v_model_path_mirna = "./data_modules/datasets/transforms/word2vec/models/word2vec_{}_{}_mirna.model".format(
    config_path.split("_")[-2], data_split_seed)
w2v_model_path_target = "./data_modules/datasets/transforms/word2vec/models/word2vec_{}_{}_target.model".format(
    config_path.split("_")[-2], data_split_seed)

transform = Compose([
    ToWord2vecEmbedding(w2v_model_path_mirna, "mirna"),
    ToWord2vecEmbedding(w2v_model_path_target, "target"),
    ToTensor(("mirna", "target"))
])

data_module = GraphInteractionDataModule(config_path, int(batch_size), int(data_split_seed), transform=transform)

graph_layer_sizes = [(16, hyperparams['gnn_hidden_size']),
                     *[(hyperparams['gnn_hidden_size'], hyperparams['gnn_hidden_size']) for i in
                       range(hyperparams['n_gnn_layers'] - 1)]]
fc_layer_sizes = [(hyperparams['gnn_hidden_size'], hyperparams['fc_hidden_size']),
                  *[(hyperparams['fc_hidden_size'], hyperparams['fc_hidden_size']) for i in
                    range(hyperparams['n_fc_layers'] - 1)]]

module = GnnLM(LayerType[gnn_layer_type], graph_layer_sizes, GlobalPoolingType[global_pooling], fc_layer_sizes,
               hyperparams['dropout_rate'], hyperparams['lr'])

checkpoint_callback = ModelCheckpoint(dirpath=model_dir,
                                      filename="graphtar_net_{}_{}_{}_{}_{}".format(config_name, gnn_layer_type, batch_size,
                                                                                 data_split_seed,
                                                                                 lr), save_top_k=1, monitor="val_loss")

trainer = Trainer(gpus=1, max_epochs=hyperparams['epochs_num'],
                  callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=100), checkpoint_callback],
                  logger=neptune_logger
                  )
trainer.fit(module, datamodule=data_module)
