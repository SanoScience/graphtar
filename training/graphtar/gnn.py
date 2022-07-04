from pytorch_lightning import Trainer

from data_modules.graph_interaction_data_module import GraphInteractionDataModule
from lightning_modules.graphtar.gnn import GnnLM
from lightning_modules.models.graphtar.gnn import LayerType, GlobalPoolingType

data_module = GraphInteractionDataModule("../../data_modules/configs/graphtar_config.json")

module = GnnLM(LayerType.GCN, [(5, 16), (16, 32)], GlobalPoolingType.MEAN, [(32, 16), (16, 8)], 0.4)

trainer = Trainer(accelerator='gpu', max_epochs=2, limit_train_batches=0.1, limit_test_batches=0.1)
trainer.fit(module, datamodule=data_module)
