import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap

font = {'family' : 'times',
        'size'   : 14}

plt.rc('font', **font)

plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)

#samples = [1, 2, 3]
samples = [2, 4, 6]
no_classes = 4
markers = ['o', '^', '*', 's']
colors = ['red', 'blue', 'green', 'yellow']
classes_original = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
classes_counterfactual = ['Class 1', 'Class 2', 'Class 4', 'Class 5']
bar_vals = [-0.5, -0.25, 0.25, 0.5]
hatches = ['..', '//', '--', '**']
n_bars = 4
bar_width = 0.25

no_samples_original = [[79, 326, 1, 82], [79, 326, 1, 82], [79, 325, 2, 82]]
no_samples_counterfactual = [[61, 362, 62, 3], [61, 356, 69, 2], [63, 355, 67, 3]]


values_svm_linear_original_total = [989.4542, 4087.7488, 12.1448, 967.7633]
values_svm_rbf_original_total = [1003.1617, 4023.4961, 12.3011, 1004.7776]
values_svm_poly_original_total = [12.5192*no_samples_original[2][0], 12.3673*no_samples_original[2][1], 12.5737*no_samples_original[2][2], 12.4655*no_samples_original[2][3]]

#values_svm_linear_counterfactual_total = []
#values_svm_rbf_counterfactual_total = []
#values_svm_poly_counterfactual_total = []
	
values_svm_linear_original_per_class = []
values_svm_rbf_original_per_class = []
values_svm_poly_original_per_class = []

values_svm_linear_counterfactual_per_class = [12.4794, 12.3506, 12.6305, 12.3506]
values_svm_rbf_counterfactual_per_class = [12.3995, 12.3843, 12.3532, 12.3767]
values_svm_poly_counterfactual_per_class = [12.5931, 12.3737, 12.4698, 12.417]

for i in range(no_classes):
    values_svm_linear_original_per_class.append(values_svm_linear_original_total[i]/(no_samples_original[0][i]))
    values_svm_rbf_original_per_class.append(values_svm_rbf_original_total[i]/(no_samples_original[1][i]))
    values_svm_poly_original_per_class.append(values_svm_poly_original_total[i]/(no_samples_original[2][i]))

values_svm_original_per_class = []
for i in range(no_classes):
    values_class = []
    values_class.append(values_svm_linear_original_per_class[i])
    values_class.append(values_svm_rbf_original_per_class[i])
    values_class.append(values_svm_poly_original_per_class[i])
    values_svm_original_per_class.append(values_class)

values_svm_counterfactual_per_class = []
for i in range(no_classes):
    values_class = []
    values_class.append(values_svm_linear_counterfactual_per_class[i])
    values_class.append(values_svm_rbf_counterfactual_per_class[i])
    values_class.append(values_svm_poly_counterfactual_per_class[i])
    values_svm_counterfactual_per_class.append(values_class)
    

fig = plt.figure(figsize = (8, 4))
ax = fig.add_subplot(121)

plt.setp(ax, xticks = samples, xticklabels = ['linear', 'rbf', 'poly'])
plt.ylim(11.5, 12.8)
for i in range(no_classes):
    x_offset = (i - n_bars/2)*bar_width + bar_width/2
    ax.bar([(x + x_offset) for x in samples], values_svm_original_per_class[i], align = 'center', width = bar_width, alpha = 0.5, color = colors[i],  capsize = 5, hatch = hatches[i], label = classes_original[i])

ax.set_xlabel("Different SVM Kernels", fontsize=16)
ax.set_title("\n".join(wrap("Original Dataset")))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower right', borderaxespad=0.)

ax = fig.add_subplot(122)

plt.setp(ax, xticks = samples, xticklabels = ['linear', 'rbf', 'poly'])
plt.ylim(11.5, 12.8)
for i in range(no_classes):
    x_offset = (i - n_bars/2)*bar_width + bar_width/2
    #x_offset = bar_vals[i]
    ax.bar([(x + x_offset) for x in samples], values_svm_counterfactual_per_class[i], align = 'center', width = bar_width, alpha = 0.5, color = colors[i],  capsize = 5, hatch = hatches[i], label = classes_counterfactual[i])

ax.set_xlabel("Different SVM Kernels", fontsize=16)
ax.set_title("\n".join(wrap("Counter Factual Dataset")))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower right', borderaxespad=0.)

plt.tight_layout(pad=3.0)

# =============================================================================
# plt.subplot(121)
# for i in range(no_classes):
#     plt.plot(samples, values_svm_original_per_class[i], marker = markers[i], color = colors[i], label = classes_original[i])
# #for i in samples:
# #    plt.axvline(x = i)
# plt.xlabel("Different SVM Kernels", fontsize=16)
# plt.xticks(np.arange(1, 4), ['linear', 'rbf', 'poly'])
# plt.title("\n".join(wrap("On Original Sentiment Analysis Dataset")))
# plt.legend(loc = 'center right', bbox_to_anchor=(1, 0.2))
# plt.subplot(122)
# for i in range(no_classes):
#     #plt.scatter(samples, values_svm_counterfactual_per_class[i], marker = markers[i], color = colors[i], label = classes_counterfactual[i])
#     plt.plot(samples, values_svm_counterfactual_per_class[i], marker = markers[i], color = colors[i], label = classes_counterfactual[i])
# #for i in samples:
# #    plt.axvline(x = i)
# plt.xlabel("Different SVM Kernels", fontsize=16)
# plt.xticks(np.arange(1, 4), ['linear', 'rbf', 'poly'])
# plt.title("\n".join(wrap("On Counter Factual Sentiment Analysis Dataset")))
# plt.legend(loc = 'center right', bbox_to_anchor=(0.7, 0.8))
# =============================================================================
plt.show()

