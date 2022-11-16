import csv

import torch
from torchvision.transforms import Compose

from data_modules.datasets.transforms.to_tensor import ToTensor
from data_modules.datasets.transforms.word2vec.to_word2vec_embedding import ToWord2vecEmbedding
from data_modules.graph_interaction_data_module import GraphInteractionDataModule
from lightning_modules.graphtar.gnn import GnnLM
from lightning_modules.models.graphtar.gnn import LayerType, GlobalPoolingType

if __name__ == "__main__":
    configs = [
        ("GCN", 128, "deepmirtar", 128, 3, "MAX", 512, 2, 0.4, 0.001),
        ("GCN", 64, "miraw", 128, 3, "MAX", 512, 2, 0.4, 0.001),
        ("GCN", 64, "mirtarraw", 128, 3, "MAX", 512, 2, 0.4, 0.001),
        ("GAT", 128, "deepmirtar", 256, 5, "ADD", 128, 2, 0.4, 0.001),
        ("GAT", 512, "miraw", 256, 5, "ADD", 128, 2, 0.4, 0.001),
        ("GAT", 512, "mirtarraw", 256, 5, "ADD", 128, 2, 0.4, 0.001),
        ("GRAPHSAGE", 128, "deepmirtar", 256, 5, "ADD", 256, 3, 0.4, 0.001),
        ("GRAPHSAGE", 32, "miraw", 256, 5, "ADD", 256, 3, 0.4, 0.001),
        ("GRAPHSAGE", 256, "mirtarraw", 256, 5, "ADD", 256, 3, 0.4, 0.001),
    ]
    seeds = [418, 627, 960, 426, 16, 523, 708, 541, 747, 897, 714, 515, 127, 657, 662, 284, 595, 852, 734, 136, 394,
             321, 200, 502, 786, 817, 411, 264, 929, 407]

    for config in configs:
        for seed in seeds:
            gnn_layer_type, batch_size, dataset_type, graph_layer_size, n_gnn_layers, global_pooling, fc_layer_size, n_fc_layers, dropout_rate, lr = config
            model_path = "./experiments/graphtar/models/graphtar_net_graphtar_config_{}_w2v_{}_{}_{}_0.001.ckpt".format(
                dataset_type, gnn_layer_type, batch_size, seed)
            config_path = "./data_modules/configs/graphtar_config_{}_w2v.json".format(config[2])
            w2v_model_path_mirna = "./data_modules/datasets/transforms/word2vec/models/word2vec_{}_{}_mirna.model".format(
                dataset_type, seed)
            w2v_model_path_target = "./data_modules/datasets/transforms/word2vec/models/word2vec_{}_{}_target.model".format(
                dataset_type, seed)
            graph_layer_sizes = [(16, graph_layer_size),
                                 *[(graph_layer_size, graph_layer_size) for i in
                                   range(n_gnn_layers - 1)]]
            fc_layer_sizes = [(graph_layer_size, fc_layer_size),
                              *[(fc_layer_size, fc_layer_size) for i in
                                range(n_fc_layers - 1)]]
            path = "../"

            transform = Compose([
                ToWord2vecEmbedding(w2v_model_path_mirna, "mirna"),
                ToWord2vecEmbedding(w2v_model_path_target, "target"),
                ToTensor(("mirna", "target"))
            ])
            model = GnnLM.load_from_checkpoint(
                model_path,
                layer_type=LayerType[gnn_layer_type],
                graph_layer_sizes=graph_layer_sizes,
                global_pooling=GlobalPoolingType[global_pooling],
                hidden_layer_sizes=fc_layer_sizes,
                dropout_rate=dropout_rate, lr=lr)
            data_module = GraphInteractionDataModule(config_path, 1024, seed, transform=transform)
            data_module.setup(stage="test")
            test_dataloader = data_module.test_dataloader()
            ground_truths = []
            inference_results = []
            for i, data in enumerate(test_dataloader):
                with torch.no_grad():
                    model.model.eval()
                    result = model.forward(data[0])
                    ground_truths += data[0].y.numpy().tolist()
                    inference_results += result.numpy()[:, 0].tolist()
            with open("experiments/graphtar/results/{}_{}_{}.csv".format(gnn_layer_type, dataset_type, seed), 'w') as f:
                writer = csv.writer(f)
                writer.writerows(zip(inference_results, ground_truths))
