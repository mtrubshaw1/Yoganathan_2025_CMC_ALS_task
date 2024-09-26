
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
# import statsmodels.api as sm
# from scipy.stats import kruskal
# from scikit_posthocs import posthoc_dunn
import numpy as np
import pandas as pd

# beta_rest = np.load('data/beta_rest.npy')
beta_15 = np.load('data/beta_15.npy')
burst_FOs = np.load('data/burst_FOs.npy')
# burst_rates = np.load('data/burst_rates.npy')
burst_meanLTs = np.load('data/burst_meanLTs.npy',)
burst_amps = np.load('data/burst_amps.npy')


means = np.vstack((burst_FOs,burst_meanLTs,burst_amps,beta_15)).T

# Load regressor data
demographics = pd.read_csv("../demographics/task_demographics.csv")


category_list = demographics["Group"].values
category_list[category_list == "HC"] = 1
category_list[category_list == "ALS"] = 2
category_list[category_list == "rALS"] = 2
category_list[category_list == "PLS"] = 3
category_list[category_list == "rPLS"] = 3
category_list[category_list == "FDR"] = 3
category_list[category_list == "rFDR"] = 3



tstats = np.load('data/tstats.npy')[-1]
pvalues = np.load('data/contrast_0_pvalues.npy')

#"Raw beta power \nat rest", "BL corrected beta \npower (1-5s)",
group_labels = ["Burst fractional \noccupancy",
                "Mean burst \nduration", "Mean burst \namplitude","Mean beta \npower"]

values = tstats
n = len(values)

x = range(-n // 2, n // 2)

barplot = sns.barplot(x=list(x), y=values, palette='pastel')

max_abs_value = max(max(values), abs(min(values)))
plt.ylim(-max_abs_value * 1.2, max_abs_value * 1.2)
plt.axhline(0, color='black', linewidth=0.8)

# plt.xlabel('Contrasts', fontsize=12)
plt.ylabel('T-stats',fontsize=12)
plt.ylim(-4,4)
# plt.title(f'Network contribution to total activity - network {contrast + 1}')
# plt.title(f'Network kurtosis', fontsize=20)
plt.xticks(np.arange(len(group_labels)), group_labels, fontsize=10)

legend_labels = ['* = p < 0.05', '** = p < 0.01', '*** = p < 0.001']
handles = [plt.Line2D([0], [0], color='w', markerfacecolor='red', markersize=10, label=label) for label in legend_labels]

plt.legend(handles=handles, labels=legend_labels, loc='lower right', fontsize=5)

for i in range(n):
    plt.text(i, values[i], f'p = {pvalues[i]:.3f}', ha='center', va='top', color='black', fontsize=8)
    if pvalues[i] < 0.001:
        plt.text(i, values[i], '***', ha='center', va='baseline', color='red', fontsize=24)
    elif pvalues[i] < 0.01:
        plt.text(i, values[i], '**', ha='center', va='baseline', color='red', fontsize=24)
    elif pvalues[i] < 0.05:
        plt.text(i, values[i], '*', ha='center', va='baseline', color='red', fontsize=24)
# plt.tight_layout()
# plt.title('Beta burst metrics \nmotor cortex 1-3s (ALS-HC)',fontsize=18)
plt.savefig(f"plots/beta_burst_stats.png",dpi=300)
# plt.close()  # Close the figure to release memory and avoid overlap in the next figure

