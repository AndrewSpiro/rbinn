# Brain-Inspired Mechanisms for Robustness in Artificial Neural Networks: An Analysis via Robustness Distributions

## Quick Start

### Setup

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
## Experiments
{
    "PixelReg clean": {"path": "experiments/attias_testset_nnenum_one2one_18-02-2026+22_31"},
    "PixelReg FGSM": {"path": "experiments/pixelreg_seed_0_rd_26-05-2026+06_50"},
    "KHModel clean": {"path": "experiments/hulz_testset_nnenum_one2one_21-02-2026+15_30"},
    "KHModel FGSM": {"path": "experiments/khmodel_seed_0_rd_26-05-2026+19_21"},
    "EAT clean": {"path": "experiments/eat_seed_0_rd_26-05-2026+04_20"},
    "CNNF": {"path": "experiments/huang_avg_nnenum_one2one_29-03-2026+21_49"},
    "VOneNet clean": {"path": "experiments/dapellot_testset_nnenum_one2one_25-02-2026+16_43"},
    "VOneNet FGSM": {"path": "experiments/vonenet_seed_0_rd_30-05-2026+18_22"},
    "CIFAR_7_1024 clean": {"path": "experiments/cifar_7_1024_seed_0_rd_25-05-2026+17_04"},
    "CIFAR_7_1024 FGSM": {"path": "experiments/cifar_7_1024_fgsm_testset_nnenum_one2one_18-02-2026+21_17"},
    "ConvBig clean": {"path": "experiments/convbig_testset_nnenum_one2one_19-02-2026+15_48"},
    "ConvBig FGSM": {"path": "experiments/convbig_seed_0_rd_26-05-2026+06_47"}
}