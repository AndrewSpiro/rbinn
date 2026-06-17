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
sns.set_style("white")
sns.set_theme(
    style="white",
    rc={
        "figure.figsize": (11.7, 8.27),
        "font.size": 22,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
        "legend.title_fontsize": 20,
    })
sns.set_palette(sns.color_palette("Paired"))

class ReportCreator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def create_hist_figure(self, log_scale: bool = False, base = 10) -> plt.Figure:
        fig, ax = plt.subplots()
        sns.histplot(data=self.df, x="smallest_sat_value", hue="network", multiple="stack", ax=ax)
        ax.set_xlabel("Epsilon value")
        
        if log_scale:
            ax.set_yscale("log", base=base)
        
        sns.despine(ax=ax, top=False, right=False)
        fig.tight_layout()
        return fig

    def create_box_figure(self, log_scale: bool = False, base=10) -> plt.Figure:
        sort_order = [
            "pixelreg", 
            "khmodel", 
            "eat",        
            "cnnf", 
            "vonenet", 
            "cifar_7_1024", 
            "convbig"
        ]

        def get_sort_key(network_name):
            net_lower = str(network_name).lower()
            for idx, prefix in enumerate(sort_order):
                if prefix in net_lower:
                    return idx
            return len(sort_order)

        unique_networks = sorted(self.df["network"].unique(), key=get_sort_key)
        
        data = [
            self.df[self.df["network"] == net]["smallest_sat_value"].dropna().values 
            for net in unique_networks
        ]
        positions = list(range(1, len(unique_networks) + 1))

        fig, ax = plt.subplots()
        
        ax.set_facecolor("white")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color('black') 
            spine.set_visible(True)

        ax.boxplot(
            data, 
            positions=positions, 
            showfliers=True, 
            medianprops={'color': 'orange', 'linewidth': 2.0},
            showmeans=True,
            meanprops={
                'marker': 'D',         
                'markerfacecolor': 'red',
                'markeredgecolor': 'black',
                'markersize': 8           
            }
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(unique_networks, rotation=23, ha='right')
        ax.set_xlim(0.5, len(unique_networks) + 0.5)
        
        ax.set_xlabel("Network")
        ax.set_ylabel("Epsilon values")

        if log_scale:
            ax.set_yscale("log", base=base)
    
        fig.tight_layout()
        return fig

    def create_kde_figure(self, log_scale: bool = False, base=10) -> plt.Figure:
        fig, ax = plt.subplots()
        sns.kdeplot(data=self.df, x="smallest_sat_value", hue="network", multiple="stack", ax=ax)
        ax.set_xlabel("Epsilon value")
        
        if log_scale:
            ax.set_xscale("log", base=base)

        sns.despine(ax=ax, top=False, right=False)
        fig.tight_layout()
        return fig

    def create_ecdf_figure(self, log_scale: bool = False, base=10) -> plt.Figure:
        fig, ax = plt.subplots()
        sns.ecdfplot(data=self.df, x="smallest_sat_value", hue="network", ax=ax, linewidth=3.5)
        ax.set_xlabel("Epsilon value")
        
        if log_scale:
            ax.set_xscale("log", base=base)

        sns.despine(ax=ax, top=False, right=False)
        fig.tight_layout()
        return fig

    def create_anneplot(self) -> plt.Figure:
        fig, ax = plt.subplots()
        df = self.df
        for network in df.network.unique():
            df = df.sort_values(by="epsilon_value")
            cdf_x = np.linspace(0, 1, len(df))
            ax.plot(df.epsilon_value, cdf_x, label=network, linewidth=2.5)
            ax.fill_betweenx(cdf_x, df.epsilon_value, df.smallest_sat_value, alpha=0.3)
            ax.set_xlim(0, 0.35)
            ax.set_xlabel("Epsilon values")
            ax.set_ylabel("Fraction critical epsilon values found")
            ax.legend()

        sns.despine(ax=ax, top=False, right=False)
        fig.tight_layout()
        return fig