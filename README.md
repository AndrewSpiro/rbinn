# Brain-Inspired Mechanisms for Robustness in Artificial Neural Networks: An Analysis via Robustness Distributions

## Quick Start

### Overview

1. The full pipeline including validations of the original BINNs, adversariral training and obtaining robustness distributions can be run from `run.sh`.
2. Obtaining robustness distributions can be done from `run.sh`
3. Validating the original BINNs and performing adversarial training can either be done from run.sh or by `cd`ing into the BINN of interest and running `repro.sh`. The latter option allows for more fine-grained control of the training and attacks.

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

### RD Experiments Env Setup

Using the various BINNs with the VERONA package requires additional packages. To create the VERONA environment so that it is compatible with the BINNs, run the following:

```bash
~/rbinn$ conda env create --file verona_env
```

Note this environment setup differs from that outlined in official VERONA documentation.

Run from rbinn/

```bash
run.sh
```

* Note on up-to-dateness with VERONA. To pull the lates changes from VERONA, run the following:

```bash
~/rbinn$ bash pull_from_verona.sh
```

### Running the Full Pipeline



## Experiments
{
    "PixelReg clean": {"path": "experiments/attias_testset_nnenum_one2one_18-02-2026+22_31"},
    "PixelReg FGSM": {"path": "experiments/pixelreg_seed_0_rd_26-05-2026+06_50"},
    "KHModel clean": {"path": "experiments/hulz_testset_nnenum_one2one_21-02-2026+15_30"},
    "KHModel FGSM": {"path": "experiments/khmodel_seed_0_rd_26-05-2026+19_21"},
    "EAT clean": {"path": "experiments/eat_seed_0_rd_26-05-2026+04_20"},
    "EAT FGSM a=0.5": {"path": "experiments/eat_seed_0_rd_05-06-2026+13_15"},
    "CNNF clean": {"path": "experiments/cnnf_seed_0_rd_25-05-2026+17_49"},
    "CNNF FGSM": {"path": "experiments/convbig_seed_0_rd_30-05-2026+22_15"},
    "CNNF SupClean": {"path": "experiments/huang_avg_nnenum_one2one_29-03-2026+21_49"},
    "VOneNet clean": {"path": "experiments/dapellot_testset_nnenum_one2one_25-02-2026+16_43"},
    "VOneNet FGSM": {"path": "experiments/vonenet_seed_0_rd_30-05-2026+18_22"},
    "CIFAR_7_1024 clean": {"path": "experiments/cifar_7_1024_seed_0_rd_25-05-2026+17_04"},
    "CIFAR_7_1024 FGSM": {"path": "experiments/cifar_7_1024_fgsm_testset_nnenum_one2one_18-02-2026+21_17"},
    "ConvBig clean": {"path": "experiments/convbig_testset_nnenum_one2one_19-02-2026+15_48"},
    "ConvBig FGSM": {"path": "experiments/convbig_seed_0_rd_26-05-2026+06_47"}
}