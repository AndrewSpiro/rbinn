# Brain-Inspired Mechanisms for Robustness in Artificial Neural Networks: An Analysis via Robustness Distributions

## Quick Start

### Overview

1. The full pipeline including validations of the original BINNs, adversarial training and obtaining robustness distributions can be run from `run.sh`.
2. Obtaining robustness distributions alone can be done from `run.sh` by setting `RUN_VALIDS` to `false`.
3. Validating the original BINNs and performing adversarial training can either be done from `run.sh` or by `cd`ing into the BINN of interest and running `repro.sh`. The latter option allows for more fine-grained control of the training and attacks.

### BINN Env Setup

Some BINNs require specific versions of packages, so separate environments should be created for each. To create the env for each BINN, the following steps should be taken

```bash
~/rbinn$ cd binn<binn-id>_<binn_name>
~/rbinn/binn<binn-id>_<binn_name>$ conda env create -f environment.yml
~/rbinn/binn<binn-id>_<binn_name>$ conda activate <binn_name>
```

for example,

```bash
~/rbinn$ cd binn1_pixelreg
~/rbinn/binn1_pixelreg$ conda env create -f environment.yml
~/rbinn/binn1_pixelreg$ conda activate pixelreg
```

Then, install PyTorch according to the [official instructions](https://pytorch.org/get-started/locally/).

### RD Experiments Env Setup

Using the various BINNs with the VERONA package requires additional packages. To create the `rbinn_env` environment which is compatible with both VERONA and the BINNs, run the following:

```bash
~/rbinn$ conda env create --file environment.yml
```

Then, activate the environment and install PyTorch according to the [official instructions](https://pytorch.org/get-started/locally/).

### Running the Full Pipeline

Basic variables:

* `DEBUG` (`true`/`false`): set to true to test robustness distributions on a small subset of data.
* `RUN_VALIDS` (`true`/`false`): set to true to run the validation pipelines: (adversarial) training and model evaluations.
* `GET_RDS` (`true`/`false`): set to true to obtain robustness distributions.
* `AGG_RESULTS` (`true`/`false`): set to true to generate tables and figures from robustness distributions (if `GET_RDS` is `true`, this is best set to `false` since experiment paths must be specified in a JSON file).

Pipeline control variables:

* `EXPS`: use this to label groups of experiments. Results will be saved into a subdir of the same name and the script will scan experiment paths specified in a file named `experiments_<EXPS>.json`.
* `MODELS` (`[pixelreg, khmodel, eat, cnnf, vonenet, cifar_7_1024, convbig]`): use this to specify which models the pipeline should be run on.
* `TRAIN_TYPE` (`[clean, fgsm]`): specify which variant of model to use in obtaining the robustness distribution.
* `EPSILON_SPACE` (`[berger, bosman]`): perturbation magnitude presets. Both use range [0,1]; `berger`, adapted from [Empirical Analysis of Upper Bounds of Robustness Distributions using Adversarial Attacks](https://ada.liacs.nl/papers/BerEtAl25.pdf), uses step sizes 1/22 while `bosman`, adapted from [Robustness Distributions in Neural Network Verification](https://www.jair.org/index.php/jair/article/view/18403/27200), uses steps sizes 1/500.
* `PGD_NUM_ITER`: number of iterations to use in PGD for obtaining robustness distributions
* `PGD_STEP_SIZE`: step size to use in PGD for obtaining robustness distributions

Once variables are set, run the following

```bash
~/rbinn$ run.sh
```

* Note on up-to-dateness with VERONA. To pull the lates changes from VERONA, run the following (though this will overwrite the plotting script used in this project):

```bash
~/rbinn$ bash pull_from_verona.sh
```
