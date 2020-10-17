import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap

font = {'family' : 'times',
        'size'   : 14}

plt.rc('font', **font)

plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)

#samples = [1, 2, 3]
samples = [1, 2, 3]
colors = ['red', 'blue']
bars = ['original', 'counterfactual']
hatches = ['..', '//']
n_bars = 1
bar_width = 0.25

values_svm_original = [0.86, 0.82, 0.82]
values_svm_counterfactual = [0.79, 0.55, 0.54]

all_values = [values_svm_original, values_svm_counterfactual]
fig = plt.figure(figsize = (8, 4))
ax = fig.add_subplot(111)

plt.setp(ax, xticks = samples, xticklabels = ['linear', 'rbf', 'poly'])
for i in range(2):
    x_offset = (i - n_bars/2)*bar_width + bar_width/2
    ax.bar([(x + x_offset) for x in samples], all_values[i], align = 'center', width = bar_width, alpha = 0.5, color = colors[i],  capsize = 5, hatch = hatches[i], label = bars[i])

ax.set_xlabel("Different SVM Kernels", fontsize=16)
ax.set_ylabel("Interpretability", fontsize = 16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower right', borderaxespad=0.)

plt.tight_layout(pad=3.0)

plt.show()

