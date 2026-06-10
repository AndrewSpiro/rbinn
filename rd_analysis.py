import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
import json
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "VERONA"))
from ada_verona.analysis.report_creator import ReportCreator


def get_relative_df(experiment):

    absolute_df = experiment["absolute_df"]

    correct_inst_path = f"{experiment['path']}/correct_instances.csv"
    if not os.path.exists(correct_inst_path):
        raise Warning(f"{correct_inst_path} does not exist for this experiment")

    correct_inst_ids_df = pd.read_csv(correct_inst_path, index_col=0)
    correct_inst_ids = list(correct_inst_ids_df.transpose()["id"])

    all_inst_ids = absolute_df["image_id"]

    relative_df = absolute_df[[data_id in correct_inst_ids for data_id in all_inst_ids]]

    return relative_df


def create_dfs(experiments):
    for name, experiment in experiments.items():
        print(f"Loading {name}")
        experiment["absolute_df"] = pd.read_csv(
            f"{experiment['path']}/results/result_df.csv", index_col=0
        )
        experiment["absolute_df"] = experiment["absolute_df"].rename(
            columns={"smallest_sat_value": "scaled_smallest_sat_value"}
        )
        # print(experiment["absolute_df"].columns)
        experiment["absolute_df"]["network"] = name
        experiment["relative_df"] = get_relative_df(experiment)

    absolute_dfs = pd.concat(
        [experiment["absolute_df"] for experiment in experiments.values()]
    )
    relative_dfs = pd.concat(
        [experiment["relative_df"] for experiment in experiments.values()]
    )

    return absolute_dfs, relative_dfs


def unscale_eps(row):
    '''eps_max and eps_min refer to the maximum and minimum pixel values assumed in the epsilon space'''

    unnorm_dict = {
        "PixelReg clean": {
            "data_max": 2.5,
            "data_min": -2.5,
            "eps_max": 1,
            "eps_min": 0,
        },
        "PixelReg FGSM": {
            "data_max": 2.5,
            "data_min": -2.5,
            "eps_max": 1,
            "eps_min": 0,
        },
        "KHModel clean": {"data_max": 0.31337952613830566, "data_min": 0, "eps_max": 1, "eps_min": 0},
        "KHModel FGSM": {"data_max": 0.31337952613830566, "data_min": 0, "eps_max": 1, "eps_min": 0},
        "EAT clean": {"data_max": 2.64, "data_min": -1.25, "eps_max": 1, "eps_min": 0},
        "EAT FGSM": {"data_max": 2.64, "data_min": -1.25, "eps_max": 1, "eps_min": 0},
        "EAT Orig": {"data_max": 2.64, "data_min": -1.25, "eps_max": 1, "eps_min": 0},
        "CNNF": {"data_max": 1.0, "data_min": -1.0, "eps_max": 1, "eps_min": 0},
        "VOneNet clean": {
            "data_max": 1.0,
            "data_min": -1.0,
            "eps_max": 1,
            "eps_min": 0,
        },
        "VOneNet FGSM": {
            "data_max": 1.0,
            "data_min": -1.0,
            "eps_max": 1,
            "eps_min": 0,
        },
    }

    model = row["network"]
    scaled_sat_val = row["scaled_smallest_sat_value"]

    if model not in unnorm_dict:
        return scaled_sat_val

    config = unnorm_dict[model]

    factor = (config["data_max"] - config["data_min"]) / (
        config["eps_max"] - config["eps_min"]
    )

    return scaled_sat_val / factor


def create_figures(bool_relative: bool):
    """bool_relative: absolute if False, relative if True
    absolute includes originally misclassified instances, relative excludes them."""
    if bool_relative:
        zero_shifted = relative_dfs.copy()
    else:
        zero_shifted = absolute_dfs.copy()
        zero_shifted["smallest_sat_value"] = (
            zero_shifted["smallest_sat_value"] + 1e-6
        )  # shifting so that plots are compatible with log-scaling
    report_creator = ReportCreator(zero_shifted)
    hist_figure = report_creator.create_hist_figure(log_scale=False)
    box_figure = report_creator.create_box_figure(log_scale=True)
    kde_figure = report_creator.create_kde_figure(log_scale=False)
    ecdf_figure = report_creator.create_ecdf_figure(log_scale=False)
    anneplot = report_creator.create_anneplot()
    # figures = [hist_figure, box_figure, kde_figure, ecdf_figure, anneplot]
    figures = [hist_figure, box_figure, kde_figure, ecdf_figure]
    # figures = [hist_figure, box_figure, ecdf_figure, anneplot]

    for i, figure in enumerate(figures):
        figure.savefig(f"{args.results_dir}/fig_{i}.png")

    return figures

def create_tables(experiments, absolute_dfs, relative_dfs):
    summary_dict = {}

    for experiment in experiments:
        print(f"processing {experiment}")
        
        abs_net = absolute_dfs[absolute_dfs["network"] == experiment]
        rel_net = relative_dfs[relative_dfs["network"] == experiment]

        # breakpoint()
        
        # Calculate base metrics
        num_clean_corr = len(rel_net)
        total_instances = len(abs_net)
        clean_acc = num_clean_corr / total_instances if total_instances > 0 else 0

        # Calculate Percentiles for relative
        # 50th = Median; 90th = Value where 10% are greater or equal
        if not rel_net.empty:
            sat_rel_values = rel_net["smallest_sat_value"]
            p50_sat_rel = np.percentile(sat_rel_values, 50)
            p90_sat_rel = np.percentile(sat_rel_values, 90)
            
            min_sat_rel = sat_rel_values.min()
            max_sat_rel = sat_rel_values.max()
            mean_sat_rel = sat_rel_values.mean()
            median_sat_rel = sat_rel_values.median()
            std_sat_rel = sat_rel_values.std()
            print(min_sat_rel)
            assert(min_sat_rel > 0)
            assert(max_sat_rel < 0.4)
        else:
            p50_sat_rel = p90_sat_rel = min_sat_rel = max_sat_rel = mean_sat_rel = std_sat_rel = np.nan

        summary_dict[experiment] = {
            "min_sat_test_abs": abs_net["smallest_sat_value"].min() if not abs_net.empty else np.nan,
            "max_sat_test_abs": abs_net["smallest_sat_value"].max() if not abs_net.empty else np.nan,
            "mean_sat_test_abs": abs_net["smallest_sat_value"].mean() if not abs_net.empty else np.nan,
            "med_sat_test_abs": abs_net["smallest_sat_value"].median() if not abs_net.empty else np.nan,
            "std_sat_test_abs": abs_net["smallest_sat_value"].std() if not abs_net.empty else np.nan,
            "min_sat_test_rel": min_sat_rel,
            "max_sat_test_rel": max_sat_rel,
            "mean_sat_test_rel": mean_sat_rel,
            "med_sat_test_rel": median_sat_rel,
            "std_sat_test_rel": std_sat_rel,
            "p50_sat_test_rel": p50_sat_rel,
            "p90_sat_test_rel": p90_sat_rel,
            "num_clean_corr": num_clean_corr,
            "clean_acc": clean_acc
        }

    return pd.DataFrame(summary_dict).transpose()

if __name__ == "__main__":

    print("main called")
    parser = argparse.ArgumentParser(description="Stats and plots for RDs")

    parser.add_argument(
        "--experiments_path",
        type=str,
        default="experiments.json",
        help="path to experiment selection json",
    )

    parser.add_argument(
        '--results_dir',
        type = str,
        help="dir for saving tables and plots"
    )
    args = parser.parse_args()

    print("args parsed")
    experiments_path = args.experiments_path
    experiments = json.load(open(experiments_path, "r"))
    # print(experiments)
    absolute_dfs, relative_dfs = create_dfs(experiments)
    # print(absolute_dfs.columns)
    # print(relative_dfs.columns)
    relative_dfs["smallest_sat_value"] = relative_dfs.apply(unscale_eps, axis=1)
    absolute_dfs["smallest_sat_value"] = absolute_dfs.apply(unscale_eps, axis=1)
    figures = create_figures(bool_relative=True)
    tables = create_tables(experiments, absolute_dfs, relative_dfs)
    table_abs = tables[['max_sat_test_abs', 'mean_sat_test_abs','std_sat_test_abs','clean_acc']]
    table_rel = tables[['max_sat_test_rel', 'mean_sat_test_rel','std_sat_test_rel','num_clean_corr']]

    table_abs.to_latex(escape=True, float_format="%.3f", buf = f"{args.results_dir}/absolute.txt")
    table_rel.to_latex(escape=True, float_format="%.3f", buf = f"{args.results_dir}/relative.txt")