import logging
import argparse
import json
import numpy as np
from pathlib import Path
import os
import pandas as pd
import time
from tqdm import tqdm
import sys

# torch imports
import torch
import torchvision
import torchvision.transforms as transforms

# MODEL IMPORTS
# pixelreg
from binn1_pixelreg.models import ResNet18
print("pixelreg imports done")

# khmodel
from binn2_khmodel.tests.context import *
from binn2_khmodel.src.LocalLearning.LocalLearning import FKHL3, KHModel
from binn2_khmodel.src.LocalLearning import Data
print("khmodel imports done")

# eat
from binn3_eat.model import model_dispatcher
from binn3_eat.helper_class import AddEdgeMap

print("eat imports done")

# cnnf
from binn4_cnnf.cnnf.model_cifar import WideResNet
from binn4_cnnf.cnnf.iterative_wrapper import IterativeWrapper

print("cnnf imports done")

# vonenet
from binn5_vonenet import vonenet
from binn5_vonenet.vonenet import CIFARVOneNetWrapper
from binn5_vonenet.train import load_model as load_vonenet

print("vonenet imports done")

# VERONA imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "VERONA"))
from ada_verona import PGDAttack
from ada_verona.epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)

from ada_verona.database.dataset.pytorch_experiment_dataset import (
    PytorchExperimentDataset,
)
from ada_verona.database.experiment_repository import ExperimentRepository
from ada_verona.database.machine_learning_model.pytorch_network import (
    PyTorchNetwork,
)

from ada_verona.dataset_sampler.predictions_based_sampler import (
    PredictionsBasedSampler,
)
from ada_verona.dataset_sampler.dataset_sampler import DatasetSampler
from ada_verona.database.dataset.experiment_dataset import ExperimentDataset
from ada_verona.epsilon_value_estimator.epsilon_value_estimator import (
    EpsilonValueEstimator,
)

from ada_verona.verification_module.attack_estimation_module import (
    AttackEstimationModule,
)
from ada_verona.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from ada_verona.verification_module.property_generator.property_generator import (
    PropertyGenerator,
)

print("verona imports done")

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
torch.manual_seed(0)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def create_transforms(network_name):
    if network_name == "pixelreg":
        print("Transforming for pixelreg")
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: (x * 5) - 2.5)]
        )
    elif network_name == "vonenet":
        print("Transforming for vonenet")
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    elif network_name == "cnnf":
        print("Transforming for cnnf")
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
        )
    elif network_name == "eat":
        print("Transforming for eat")
        return transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)
                ),
                AddEdgeMap(),
            ]
        )
    else:
        print("Not performing any special datatransforms")
        return transforms.ToTensor()


def get_epsilon_list(search_space: str):
    """A function to easily swtich between epsilon lists.
    orig_min and orig_max as the minimum and maximum pixel values assumed for creating epsilon_list. These may be used later for rescaling the list for datasets with different pixel ranges.
    """

    if search_space == "berger":
        epsilon_list = np.arange(0, 0.4, 1 / 22)
        orig_min = 0
        orig_max = 1
        return epsilon_list, orig_min, orig_max
    elif search_space == "bosman":
        epsilon_list = np.arange(0.001, 0.4, 0.002)
        orig_min = 0
        orig_max = 1
        return epsilon_list, orig_min, orig_max
    else:
        raise Exception(
            f"Only implemented search spaces are berger and bosman. got {search_space} instead."
        )


def normalize_epsilon_list(dataset, epsilon_list: list, eps_max, eps_min):
    """
    Rescale the epsilons so they have the have the appropriate relative perturbation on images.
    Range is defined as the difference between the max anx min pixel value in the dataset.
    The scaling factor is the qoutient of the data range and the eps range.
    Each eps in the original list is multiplied by the scaling facotr.
    """

    data_max = max([torch.max(dataset[i][0]) for i in range(len(dataset))])
    data_min = min([torch.min(dataset[i][0]) for i in range(len(dataset))])
    print(f"Normalizing. Dataset max is {data_max}, dataset min is {data_min}")

    normalized_list = []
    for eps in epsilon_list:
        eps_normalized = eps * ((data_max - data_min)) / (eps_max - eps_min)
        normalized_list.append(np.float64(eps_normalized.numpy()))
    print(
        f"Max and min epsilons were {max(epsilon_list), min(epsilon_list)}, now they are {max(normalized_list), min(normalized_list)}"
    )
    return normalized_list


def create_distribution(
    experiment_repository: ExperimentRepository,
    dataset: ExperimentDataset,
    dataset_sampler: DatasetSampler,
    epsilon_value_estimator: EpsilonValueEstimator,
    property_generator: PropertyGenerator,
):
    if NETWORK_TYPE == "pytorch":
        network = load_pt_network(NETWORK_NAME, NETWORK_PATH)
    else:
        raise Exception(
            f"Only supported NETWORK_TYPE currently is 'pytorch'. Got {NETWORK_TYPE}."
        )
    print(f"network: {network}")

    try:
        if args.bool_debug:
            num_samples = 10
            sampled_data = dataset.get_subset(range(num_samples))
            print(f"{len(sampled_data)} images sampled from {len(dataset)}.")
            start_time = time.time()
            correct_data = dataset_sampler.sample(network, sampled_data)
            end_time = time.time()
            print(f"time to sample was {end_time - start_time}")
            save_correct_instances(correct_data, experiment_repository)
            print(f"Clean accuracy on subset is {len(correct_data)/num_samples}")
        else:
            sampled_data = dataset
            print(f"{len(sampled_data)} images sampled from {len(dataset)}.")
            if hasattr(network.model, "run_average"):
                print("model has 'run_average' attr")
            else:
                print("model does not have 'run_average' attr")
                print("Sampling started")
                correct_data = dataset_sampler.sample(network, sampled_data)
                print("Sampling finished")
                save_correct_instances(correct_data, experiment_repository)
                print(
                    f"Clean accuracy on subset is {len(correct_data)/len(sampled_data)}"
                )
        print("Data sampled")

    except Exception as e:
        logging.info(f"failed for network {network} with error {e}")
    for data_point in tqdm(sampled_data):
        verification_context = experiment_repository.create_verification_context(
            network=network,
            data_point=data_point,
            property_generator=property_generator,
        )
        epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(
            verification_context
        )
        experiment_repository.save_result(epsilon_value_result)

    experiment_repository.save_plots()


def load_pt_network(network_name: str, network_path):
    print(f"Loading network {network_name}")
    if network_name == "pixelreg":
        model_info = torch.load(network_path)
        model_hash = list(model_info.keys())[0]
        state_dict = model_info[model_hash]["best_state"]
        model = ResNet18(in_shape=(3, 32, 32), num_classes=10)
        model.load_state_dict(state_dict)
        network = PyTorchNetwork(model, (1, 3, 32, 32), network_name)
        return network
    elif network_name == "khmodel":
        print("load khmodel")
    elif network_name == "eat":
        model, _, _, _ = model_dispatcher("cifar10", "rgbedge", "cifar10", 64, 10)
        state_dict = torch.load(network_path)
        model.load_state_dict(state_dict)
        network = PyTorchNetwork(model, (1, 4, 64, 64), network_name)
        return network
    elif network_name == "cnnf":
        model = WideResNet(40, 10, 2, 0.0, ind=5, cycles=2, res_param=0.1)
        state_dict = torch.load(network_path)
        model.load_state_dict(state_dict)
        cycles_model = IterativeWrapper(model)
        network = PyTorchNetwork(cycles_model, (1, 3, 32, 32), network_name)
        return network
    elif network_name == "vonenet":
        ckpt_data = torch.load(network_path)
        state_dict = ckpt_data["state_dict"]
        vonenet = load_vonenet()
        vonenet.load_state_dict(state_dict)
        model = CIFARVOneNetWrapper(vonenet)
        network = PyTorchNetwork(model, (1, 3, 32, 32), network_name)
        return network
    else:
        raise Exception(
            f"Supported architectures are pixelreg, khmodel, eat, cnnf, and vonenet. Update this script with relevant imports to add a new architecture."
        )


def save_correct_instances(
    dataset: ExperimentDataset, experiment_repository: ExperimentRepository
):
    id_label_dict = dict.fromkeys(range(len(dataset)))
    for i in range(len(dataset)):
        data_point = dataset[i]
        id_label_dict[i] = {"id": None, "label": None}
        id_label_dict[i]["id"] = data_point.id
        id_label_dict[i]["label"] = data_point.label
    id_label_df = pd.DataFrame(id_label_dict)
    id_label_df.to_csv(
        experiment_repository.get_act_experiment_path() / "correct_instances.csv"
    )


def main():
    transform = create_transforms(NETWORK_NAME)

    torch_dataset = getattr(torchvision.datasets, DATASET_NAME)(
        root=DATASET_DIR, train=False, download=True, transform=transform
    )

    epsilon_list = normalize_epsilon_list(
        torch_dataset, EPSILON_LIST, ORIG_MAX, ORIG_MIN
    )
    dataset = PytorchExperimentDataset(dataset=torch_dataset)

    experiment_repository = ExperimentRepository(
        base_path=EXPERIMENT_REPOSITORY_PATH, network_folder=NETWORK_FOLDER
    )

    experiment_name = f"{NETWORK_NAME}_{model_seed}_rd"
    epsilon_value_estimator = BinarySearchEpsilonValueEstimator(
        epsilon_value_list=epsilon_list.copy(), verifier=VERIFIER
    )
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True)
    experiment_repository.initialize_new_experiment(experiment_name)
    experiment_repository.save_configuration(
        dict(
            experiment_name=experiment_name,
            experiment_repository_path=str(EXPERIMENT_REPOSITORY_PATH),
            network_folder=str(NETWORK_FOLDER),
            dataset=str(dataset),
            timeout=600,
            epsilon_list=[str(x) for x in epsilon_list],
        )
    )
    print("Creating robustness distribution...")
    create_distribution(
        experiment_repository=experiment_repository,
        dataset=dataset,
        dataset_sampler=dataset_sampler,
        epsilon_value_estimator=epsilon_value_estimator,
        property_generator=PROPERTY_GENERATOR,
    )
    print(f"Distribution created for {NETWORK_NAME}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Obtain robustness distribution")

    parser.add_argument(
        "model",
        choices=["pixelreg", "eat", "cnnf", "vonenet"],
        help="Model on which to obtain robustness distributions.",
    )

    # Attack paramters
    parser.add_argument(
        "--epsilon_space",
        choices=["berger", "bosman"],
        help="Preset for epsilon space. berger: (0, 0.4] step size 1/22; bosman: (0.001, 0.4] step size 0.002.",
    )
    parser.add_argument(
        "--pgd_num_iter",
        type=int,
        help="Number of iterations for PGD attack",
    )
    parser.add_argument(
        "--pgd_step_size",
        type=float,
        help="Steps size for PGD attack",
    )

    # Other
    parser.add_argument(
        "--bool_debug",
        type=str2bool,
        default=False,
        help="whether to run the script in debug mode",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="path to dataset root"
    )
    parser.add_argument(
        "--exp_repo_path",
        type=str,
        default="experiments",
        help="Path for robustness distribution experiments",
    )
    parser.add_argument(
        "--model_seed", type=int, help="the seed used to train this model"
    )
    parser.add_argument(
        "--train_type", choices=['clean', 'adv'], default = 'clean', help="whether model was trained normally or adversarially"
    )
    
    args = parser.parse_args()

    if args.bool_debug:
        print("Running in debug mode.")

    models = json.load(open("models.json", "r"))

    EXPERIMENT_REPOSITORY_PATH = Path(args.exp_repo_path)
    print(EXPERIMENT_REPOSITORY_PATH)

    DATASET_NAME = "CIFAR10"
    DATASET_DIR = args.data_dir

    NETWORK_NAME = args.model
    model_dict = models[NETWORK_NAME]
    if args.model_seed:
        model_seed = f"seed_{args.model_seed}"
    else:
        model_seed = list(model_dict["paths"].keys())[0]

    NETWORK_TYPE = model_dict["type"]
    if type(model_dict["paths"][model_seed]) == dict:
        NETWORK_PATH = model_dict["paths"][model_seed][args.train_type]
        print(f"Model trained with type {args.train_type}")
    else:
        NETWORK_PATH = Path(model_dict["paths"][model_seed])
    network_folder_str, network_full_name_str = os.path.split(NETWORK_PATH)
    NETWORK_FOLDER, NETWORK_FULL_NAME = Path(network_folder_str), Path(
        network_full_name_str
    )
    print(f"Network folder: {NETWORK_FOLDER} \nNetwork name: {NETWORK_FULL_NAME}")

    PGD_NUM_ITER = args.pgd_num_iter
    PGD_STEP_SIZE = args.pgd_step_size
    EPSILON_LIST, ORIG_MIN, ORIG_MAX = get_epsilon_list(search_space=args.epsilon_space)
    print(f"Length of epsilon list: {len(EPSILON_LIST)}")

    VERIFIER = AttackEstimationModule(
        attack=PGDAttack(number_iterations=PGD_NUM_ITER, step_size=PGD_STEP_SIZE)
    )
    PROPERTY_GENERATOR = One2AnyPropertyGenerator()
    main()
