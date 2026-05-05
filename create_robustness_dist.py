import argparse
import json
import numpy as np

# torch imports
import torchvision.transforms as transforms

# MODEL IMPORTS
# pixelreg
from binn1_pixelreg.models import ResNet18
# khmodel
from binn2_khmodel.tests.context import *
from binn2_khmodel.src.LocalLearning.LocalLearning import FKHL3, KHModel
from binn2_khmodel.src.LocalLearning import Data
# eat
from binn3_eat.model import model_dispatcher
from binn3_eat.helper_class import AddEdgeMap
#cnnf
from binn4_cnnf.cnnf.model_cifar import WideResNet
from binn4_cnnf.cnnf.iterative_wrapper import IterativeWrapper
#vonenet
from binn5_vonenet import vonenet
from binn5_vonenet.vonenet import CIFARVOneNetWrapper
from binn5_vonenet.train import load_model as load_vonenet

# VERONA imports
from VERONA.ada_verona.verification_module.attack_estimation_module import (
    AttackEstimationModule,
)
from VERONA.ada_verona import PGDAttack
from VERONA.ada_verona.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)


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


def main():
    print("Creating robustness distribution...")
    transform = create_transforms(NETWORK_NAME)


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
    args = parser.parse_args()

    if args.bool_debug:
        print("Running in debug mode.")

    models = json.load(open("models.json", "r"))

    NETWORK_NAME = args.model
    PGD_NUM_ITER = args.pgd_num_iter
    PGD_STEP_SIZE = args.pgd_step_size
    NETWORK_TYPE = models[NETWORK_NAME]["type"]
    EPSILON_LIST, ORIG_MIN, ORIG_MAX = get_epsilon_list(search_space=args.epsilon_space)
    print(f"Length of epsilon list: {len(EPSILON_LIST)}")

    VERIFIER = AttackEstimationModule(
        attack=PGDAttack(number_iterations=PGD_NUM_ITER, step_size=PGD_STEP_SIZE)
    )
    PROPERTY_GENERATOR = One2AnyPropertyGenerator()
    main()
