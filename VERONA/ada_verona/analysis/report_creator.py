# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_style("darkgrid")
sns.set_theme(rc={"figure.figsize": (11.7, 8.27)})
sns.set_palette(sns.color_palette("Paired"))
# matplotlib.rcParams["figure.figsize"]=(11.7, 8.27)

class ReportCreator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def create_hist_figure(self, log_scale: bool = False) -> plt.Figure:
        # hist_plot = sns.histplot(data=self.df, x="epsilon_value", hue="network", multiple="stack")
        hist_plot = sns.histplot(data=self.df, x="smallest_sat_value", hue="network", multiple="stack")
        hist_plot.set_xlabel("Epsilon value")
        
        if log_scale:
            hist_plot.set_yscale("log")
        
        plt.tight_layout()
        figure = hist_plot.get_figure()

        plt.close()

        return figure

    def create_box_figure(self, log_scale: bool = False) -> plt.Figure:
        # box_plot = sns.boxplot(data=self.df, x="network", y="epsilon_value")
        box_plot = sns.boxplot(data=self.df, x="network", y="smallest_sat_value")
        box_plot.set_xticklabels(box_plot.get_xticklabels(), rotation=23)
        box_plot.set_xlabel("Network")
        box_plot.set_ylabel("Epsilon values")

        if log_scale:
            box_plot.set_yscale("log")
    
        plt.tight_layout()
        figure = box_plot.get_figure()

        plt.close()

        return figure

    def create_kde_figure(self, log_scale: bool = False) -> plt.Figure:
        # kde_plot = sns.kdeplot(data=self.df, x="epsilon_value", hue="network", multiple="stack")
        kde_plot = sns.kdeplot(data=self.df, x="smallest_sat_value", hue="network", multiple="stack")
        kde_plot.set_xlabel("Epsilon value")
        
        if log_scale:
            kde_plot.set_yscale("log")

        plt.tight_layout()
        figure = kde_plot.get_figure()

        plt.close()

        return figure

    def create_ecdf_figure(self, log_scale: bool = False) -> plt.Figure:
        # ecdf_plot = sns.ecdfplot(data=self.df, x="epsilon_value", hue="network")
        ecdf_plot = sns.ecdfplot(data=self.df, x="smallest_sat_value", hue="network")
        ecdf_plot.set_xlabel("Epsilon value")
        
        if log_scale:
            ecdf_plot.set_yscale("log")

        plt.tight_layout()
        figure = ecdf_plot.get_figure()

        plt.close()

        return figure

    def create_anneplot(self):
        df = self.df
        for network in df.network.unique():
            df = df.sort_values(by="epsilon_value")
            cdf_x = np.linspace(0, 1, len(df))
            plt.plot(df.epsilon_value, cdf_x, label=network)
            plt.fill_betweenx(cdf_x, df.epsilon_value, df.smallest_sat_value, alpha=0.3)
            plt.xlim(0, 0.35)
            plt.xlabel("Epsilon values")
            plt.ylabel("Fraction critical epsilon values found")
            plt.legend()

        return plt.gca()
