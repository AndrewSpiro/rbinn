# -*- coding: utf-8 -*-
import os, time
import utils

from dotenv import load_dotenv

from args_control import runtime_parser, get_configs
from records_management import TrainingEnvironment

from utils import (
    timer_func,
    name_wandb_run,
    get_epsilon_range,
)
import wandb

from attack import attack_resnet


@timer_func
def main(
    data_dir,
    save_dir,
    run_config,
    task_config,
    r_model_config,
    train_config,
    reg_config,
    attack_config,
):
    device = run_config["device"]

    print(
        f"run_config = {run_config}\n\ntask_config = {task_config}\n\nr_model_config = {r_model_config}\n\ntrain_config = {train_config}\n\nreg_config = {reg_config}\n\nattack_config = {attack_config}"
    )
    
    training_environment = TrainingEnvironment(
        data_dir, save_dir, run_config, train_config
    )
    training_environment.setup()

    r_training_id = training_environment.records.fetch_id(
        {
            "task_config": task_config,
            "r_model_config": r_model_config,
            "train_config": train_config,
            "reg_config": reg_config,
        }
    )
    print(f"Target training id: {r_training_id}")

    # 1. Load the pre-trained model and its original configs
    try:
        resnet, saved_train_config, saved_task_config, saved_reg_config = utils.load_model(
            save_dir, r_training_id
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model from {save_dir} with ID {r_training_id}")
        print(f"Error: {e}")
        return r_training_id, None

    if wandb_log:
        wandb.run.name = name_wandb_run(saved_task_config, saved_reg_config) + "_attack"

    # 2. Set the seed specifically for the attacks (Critical for your Gaussian noise)
    current_seed = attack_config.get("seed", saved_train_config.get("seed"))
    utils.set_seed(current_seed)
    print(f"Random seed set to: {current_seed}")

    # 3. Create ONLY the test dataloader 
    _, _, resnet_loader_test = utils.create_resnet_loaders(
        data_dir, saved_task_config, saved_train_config
    )

    attacks = attack_config.get("attack_list", ["gaussian"])
    results = {}

    # 4. Perform the attacks
    if attack_config.get("attack", True):
        for attack_type in attacks:
            print(f"\nPerforming {attack_type} attack")

            attack_config["attack_type"] = attack_type
            if not attack_config.get("epsilon_range"):
                attack_config["epsilon_range"] = get_epsilon_range(
                    saved_task_config["task"], attack_type
                )

            result = attack_resnet(
                data_dir,
                save_dir,
                resnet,
                resnet_loader_test,
                saved_task_config,
                saved_reg_config,
                attack_config,
                wandb_log=wandb_log,
                device=device,
            )
            results[attack_type] = result
            
            # Save the result uniquely so different seeds/attacks don't overwrite each other
            training_environment.metas.update(
                r_training_id,
                {f"attack_{attack_type}_seed_{current_seed}": result},
            )

    return r_training_id, results


wandb_log = False
wandb_project = "your_project_name"
reinit = True

if __name__ == "__main__":
    # Ensure this pulls the arguments meant for attacking
    parser = runtime_parser("attack_resnet")
    config = get_configs("attack_resnet", parser.parse_args())

    load_dotenv()

    if wandb_log:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(
            project=wandb_project,
            reinit=reinit,
            config={
                **config[2],
                **config[3],
                **config[4],
                **config[5],
                **config[6],
                **config[7],
            },
        )
        wandb.define_metric("Perturbation_Strength")

    r_training_id, accuracies = main(*config)
    print(f"Completed attacks on training id: {r_training_id}")
    print(f"accuracies: {accuracies}")
    if wandb_log:
        wandb.finish()