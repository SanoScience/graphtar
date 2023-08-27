import csv

import torch
from torchvision.transforms import Compose

from data_modules.datasets.transforms.to_tensor import ToTensor
from data_modules.datasets.transforms.word2vec.to_word2vec_embedding import (
    ToWord2vecEmbedding,
)
from data_modules.graph_interaction_data_module import GraphInteractionDataModule
from lightning_modules.graphtar.gnn import GnnLM
from lightning_modules.models.graphtar.gnn import LayerType, GlobalPoolingType

if __name__ == "__main__":
    configs = [
        # variable layers num
        ("mirtarraw", "GAT", 256, 1, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GAT", 256, 2, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GAT", 256, 3, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GAT", 256, 4, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GAT", 256, 5, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GAT", 256, 6, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GAT", 256, 7, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GAT", 256, 8, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GAT", 256, 9, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GAT", 256, 10, "ADD", 128, 2, 0.4, 512, 0.001),
        # variable embedding size
        ("mirtarraw", "GAT", 16, 5, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GAT", 32, 5, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GAT", 64, 5, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GAT", 128, 5, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GAT", 256, 5, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GAT", 512, 5, "ADD", 128, 2, 0.4, 512, 0.001),
        # GCN vs GAT vs SAGE
        ("mirtarraw", "GAT", 256, 5, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GCN", 256, 5, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GRAPHSAGE", 256, 5, "ADD", 128, 2, 0.4, 512, 0.001),
        # MAX vs MEAN vs ADD
        ("mirtarraw", "GAT", 256, 5, "ADD", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GAT", 256, 5, "MEAN", 128, 2, 0.4, 512, 0.001),
        ("mirtarraw", "GAT", 256, 5, "MAX", 128, 2, 0.4, 512, 0.001),
    ]
    seeds = [
        418,
        627,
        960,
        426,
        16,
        523,
        708,
        541,
        747,
        897,
        714,
        515,
        127,
        657,
        662,
        284,
        595,
        852,
        734,
        136,
        394,
        321,
        200,
        502,
        786,
        817,
        411,
        264,
        929,
        407,
    ]
    model_dir = "experiments/graphtar/models"

    for config in configs:
        for seed in seeds:
            (
                dataset_type,
                gnn_layer_type,
                graph_layer_size,
                n_gnn_layers,
                global_pooling,
                fc_layer_size,
                n_fc_layers,
                dropout_rate,
                batch_size,
                lr,
            ) = config
            model_path = "{}/graphtar_net_graphtar_config_{}_w2v_{}_{}_{}_{}_{}_{}_{}.ckpt".format(
                model_dir,
                dataset_type,
                gnn_layer_type,
                n_gnn_layers,
                graph_layer_size,
                global_pooling,
                batch_size,
                seed,
                lr,
            )
            config_path = "./data_modules/configs/graphtar_config_{}_w2v.json".format(
                dataset_type
            )
            w2v_model_path_mirna = "./data_modules/datasets/transforms/word2vec/models/word2vec_{}_{}_mirna.model".format(
                dataset_type, seed
            )
            w2v_model_path_target = "./data_modules/datasets/transforms/word2vec/models/word2vec_{}_{}_target.model".format(
                dataset_type, seed
            )
            graph_layer_sizes = [
                (16, graph_layer_size),
                *[
                    (graph_layer_size, graph_layer_size)
                    for i in range(n_gnn_layers - 1)
                ],
            ]
            fc_layer_sizes = [
                (graph_layer_size, fc_layer_size),
                *[(fc_layer_size, fc_layer_size) for i in range(n_fc_layers - 1)],
            ]
            path = "../"

            transform = Compose(
                [
                    ToWord2vecEmbedding(w2v_model_path_mirna, "mirna"),
                    ToWord2vecEmbedding(w2v_model_path_target, "target"),
                    ToTensor(("mirna", "target")),
                ]
            )
            model = GnnLM.load_from_checkpoint(
                model_path,
                layer_type=LayerType[gnn_layer_type],
                graph_layer_sizes=graph_layer_sizes,
                global_pooling=GlobalPoolingType[global_pooling],
                hidden_layer_sizes=fc_layer_sizes,
                dropout_rate=dropout_rate,
                lr=lr,
            )
            data_module = GraphInteractionDataModule(
                config_path, 1024, seed, transform=transform
            )
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
            with open(
                "experiments/graphtar/results/ablation/graphtar_net_graphtar_config_{}_w2v_{}_{}_{}_{}_{}_{}_{}.csv".format(
                    dataset_type,
                    gnn_layer_type,
                    n_gnn_layers,
                    graph_layer_size,
                    global_pooling,
                    batch_size,
                    seed,
                    lr,
                ),
                "w",
            ) as f:
                writer = csv.writer(f)
                writer.writerows(zip(inference_results, ground_truths))
