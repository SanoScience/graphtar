import csv

import torch

from data_modules.interaction_data_module import InteractionDataModule
from lightning_modules.miraw.ann import AnnLM

if __name__ == "__main__":
    configs = [
        (128, "deepmirtar", 0.001),
        (64, "miraw", 0.001),
        (128, "mirtarraw", 0.001),
    ]
    seeds = [418, 627, 960, 426, 16, 523, 708, 541, 747, 897, 714, 515, 127, 657, 662, 284, 595, 852, 734, 136, 394,
             321, 200, 502, 786, 817, 411, 264, 929, 407]

    # deepmirtar_config_128_929_0.001

    model_dir = ""

    for config in configs:
        for seed in seeds:
            batch_size, dataset_type, lr = config
            model_path = "{}/miraw_ann_{}_config_{}_{}_{}.ckpt".format(
                model_dir, dataset_type, batch_size, seed, lr)
            config_path = "./data_modules/configs/{}_config.json".format(dataset_type)
            data_module = InteractionDataModule(config_path, 1024, seed)
            x_key, y_key = data_module.get_batch_keys()
            data_module.setup(stage="test")
            autoencoder_path = "{}/autoencoder_{}_config_{}_{}_{}.pt".format(
                model_dir, dataset_type, batch_size, seed, lr)
            autoencoder = torch.load(autoencoder_path)
            model = AnnLM.load_from_checkpoint(
                model_path,
                x_key=x_key,
                y_key=y_key,
                lr=float(lr),
                encoder=autoencoder.encoder,
            )
            test_dataloader = data_module.test_dataloader()
            ground_truths = []
            inference_results = []
            for i, data in enumerate(test_dataloader):
                with torch.no_grad():
                    model.model.eval()
                    result = model.forward(data[x_key])
                    ground_truths += data[y_key].numpy()[:, 0].tolist()
                    inference_results += result.numpy()[:, 0].tolist()
            with open("experiments/miraw/results/miraw_ann_{}_config_{}_{}_{}.json".format(
                    dataset_type, batch_size, seed, lr), 'w') as f:
                writer = csv.writer(f)
                writer.writerows(zip(inference_results, ground_truths))
