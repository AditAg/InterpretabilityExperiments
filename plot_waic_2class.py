import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap

font = {'family' : 'times',
        'size'   : 14}

plt.rc('font', **font)

plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)

#samples = [1, 2, 3]
samples = [1, 1.75, 2.5]
no_classes = 2
markers = ['o', '^']
colors = ['red', 'blue']
classes_original = ['Negative sentiment', 'Positive sentiment']
classes_counterfactual = ['Negative sentiment', 'Positive sentiment']
bar_vals = [-0.25, 0.25]
hatches = ['..', '//']
n_bars = 2
bar_width = 0.25

values_svm_linear_original_per_class = [12.602, 12.4296]
values_svm_rbf_original_per_class = [12.3809, 12.3804]
values_svm_poly_original_per_class = [12.4033, 12.3969]

values_svm_linear_counterfactual_per_class = [12.5686, 12.4205]
values_svm_rbf_counterfactual_per_class = [12.3811, 12.3802]
values_svm_poly_counterfactual_per_class = [12.4145, 12.3867]

differences_class1, differences_class2 = [], []

differences_class1.append(values_svm_linear_original_per_class[0] - values_svm_linear_counterfactual_per_class[0])
differences_class1.append(values_svm_rbf_original_per_class[0] - values_svm_rbf_counterfactual_per_class[0])
differences_class1.append(values_svm_poly_original_per_class[0] - values_svm_poly_counterfactual_per_class[0])

differences_class2.append(values_svm_linear_original_per_class[1] - values_svm_linear_counterfactual_per_class[1])
differences_class2.append(values_svm_rbf_original_per_class[1] - values_svm_rbf_counterfactual_per_class[1])
differences_class2.append(values_svm_poly_original_per_class[1] - values_svm_poly_counterfactual_per_class[1])

differences_classes = []
differences_classes.append(differences_class1)
differences_classes.append(differences_class2)

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
ax = fig.add_subplot(111)

plt.setp(ax, xticks = samples, xticklabels = ['linear', 'rbf', 'poly'])
plt.ylim(12.3, 12.7)
for i in range(no_classes):
    x_offset = (i - n_bars/2)*bar_width + bar_width/2
    ax.bar([(x + x_offset) for x in samples], values_svm_original_per_class[i], yerr = [tuple(differences_classes[i]), (0, 0, 0)], align = 'center', width = bar_width, alpha = 0.5, color = colors[i],  capsize = 5, label = classes_original[i]
    , error_kw=dict(ecolor='green', lw=2, capsize=5, capthick=2))
#, hatch = hatches[i]
ax.set_xlabel("Different SVM Kernels", fontsize=16)
#ax.set_title("\n".join(wrap("WAIC values")))
ax.set_ylabel("WAIC values per class", fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower right', borderaxespad=0.)


plt.tight_layout(pad=3.0)


# =============================================================================
# fig = plt.figure(figsize = (8, 4))
# ax = fig.add_subplot(121)
# 
# plt.setp(ax, xticks = samples, xticklabels = ['linear', 'rbf', 'poly'])
# plt.ylim(12.2, 12.8)
# for i in range(no_classes):
#     x_offset = (i - n_bars/2)*bar_width + bar_width/2
#     ax.bar([(x + x_offset) for x in samples], values_svm_original_per_class[i], align = 'center', width = bar_width, alpha = 0.5, color = colors[i],  capsize = 5, hatch = hatches[i], label = classes_original[i])
# 
# ax.set_xlabel("Different SVM Kernels", fontsize=16)
# ax.set_title("\n".join(wrap("Original Dataset")))
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels, loc='lower right', borderaxespad=0.)
# 
# ax = fig.add_subplot(122)
# 
# plt.setp(ax, xticks = samples, xticklabels = ['linear', 'rbf', 'poly'])
# plt.ylim(12.2, 12.8)
# for i in range(no_classes):
#     x_offset = (i - n_bars/2)*bar_width + bar_width/2
#     #x_offset = bar_vals[i]
#     ax.bar([(x + x_offset) for x in samples], values_svm_counterfactual_per_class[i], align = 'center', width = bar_width, alpha = 0.5, color = colors[i],  capsize = 5, hatch = hatches[i], label = classes_counterfactual[i])
# 
# ax.set_xlabel("Different SVM Kernels", fontsize=16)
# ax.set_title("\n".join(wrap("Counter Factual Dataset")))
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels, loc='lower right', borderaxespad=0.)
# 
# plt.tight_layout(pad=3.0)
# =============================================================================

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


