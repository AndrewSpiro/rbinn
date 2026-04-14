# -*- coding: utf-8 -*-
import os, time
import utils
from numpy import r_
import torch.optim as optim

from dotenv import load_dotenv


from args_control import runtime_parser, get_configs
from records_management import TrainingEnvironment


from utils import (
    model_state,
    timer_func,
    name_wandb_run,
    get_s_matrix,
    get_reg_loader,
    evaluate_all,
    get_epsilon_range,
)
import wandb

from attack import attack_resnet
from classification_and_regularization import train


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

    t_start = time.time()
    print(
        f"run_config = {run_config}\n\ntask_config = {task_config}\n\nr_model_config = {r_model_config}\n\ntrain_config = {train_config}\n\nreg_config = {reg_config}\n\nattack_config = {attack_config}"
    )
    training_environment = TrainingEnvironment(
        data_dir, save_dir, run_config, train_config
    )
    training_environment.setup()

    # r_training_id = training_environment.records.fetch_id(
    #     {
    #         "task_config": task_config,
    #         "r_model_config": r_model_config,
    #         "train_config": train_config,
    #         "reg_config": reg_config,
    #         "attack_config": attack_config,
    #     }
    # )
    r_training_id = "B7A86935"
    print(f"training id: {r_training_id}")
    
    print(f"Loading pre-trained model: {r_training_id}")
    breakpoint()
    resnet, train_config, task_config, reg_config = utils.load_model(save_dir, r_training_id)
    resnet = resnet.to(device)

    _,_, resnet_loader_test = utils.create_resnet_loaders(data_dir, task_config, train_config)

    attacks = [
        "Gaussian",
        "Uniform",
        "SaltPepper",
        # "TransferredFGSM", make sure an unregularized model is trained first
        "BoundaryAttack",
    ]
    resnet, train_config, task_config, reg_config = utils.load_model(
        save_dir, r_training_id
    )
    if attack_config["attack"]:

        results = {}
        for attack_type in attacks:
            print(f"\nPerforming {attack_type} attack")

            attack_config["attack_type"] = attack_type
            # attack_config["epsilon_range"] = get_epsilon_range(
            #     task_config["task"], attack_type
            # )
            if attack_type == "Gaussian":
               attack_config["epsilon_range"] = [0,0.3]
            elif attack_type == "BoundaryAttack":
                pass
            else:
                print(f"Skipping {attack_type}")
                continue
                
            print(attack_config["epsilon_range"])
            
            result = attack_resnet(
                data_dir,
                save_dir,
                resnet,
                resnet_loader_test,
                task_config,
                reg_config,
                attack_config,
                wandb_log=wandb_log,
                device=device,
            )
            results[attack_type] = result
            training_environment.metas.update(
                r_training_id,
                {attack_type: result},
            )

    return r_training_id, results


wandb_log = False
wandb_project = "your_project_name"
reinit = True

if __name__ == "__main__":
    print('Running evaluations...')
    parser = runtime_parser("train_resnet")

    config = get_configs("train_resnet", parser.parse_args())

    load_dotenv()

    if wandb_log:
        # add wandb key to environment variables
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(
            project=wandb_project,
            reinit=reinit,
            # track hyperparameters and run metadata
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
        wandb.define_metric("Epoch")

    r_training_id, accuracies = main(*config)
    print(f"training id: {r_training_id}")
    print(f"accuracies: {accuracies}")
    if wandb_log:
        wandb.finish()
# #