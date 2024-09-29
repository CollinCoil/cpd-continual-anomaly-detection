import numpy as np
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

##################################################


def lifelong_roc(A):
    lower = np.tril(A)
    avg = lower.sum() / (((len(A) * (len(A) - 1)) / 2) + len(A))
    return avg
##################################################


def forward_transfer(A):
    upper = np.triu(A)
    for i in range(len(upper)):
        upper[i][i] = 0
    avg = upper.sum() / ((len(A) * (len(A) - 1)) / 2)
    return avg
##################################################


def backward_transfer(A):
    tasks_no = A.shape[0]
    if tasks_no == 1:
        return 0

    values = []
    for i in range(1, tasks_no):
        for j in range(0, i):
            values.append(A[i][j] - A[j][j])

    bwt = np.mean(values)
    return bwt
##################################################


def heatmap(A, distance_metric, dataset_name, strategy, params, model, transfer=''):
    if transfer == 'f':
        mask = np.zeros_like(A)
        mask[np.tril_indices_from(mask)] = True
        with sns.axes_style("white"):
            sns.set(rc={'figure.figsize': (10, 10)})
            sns.set(font_scale=1.)
            sns.heatmap(A, mask=mask, vmax=np.max(A), annot=True, square=True,  cmap="YlGnBu")
    elif transfer == 'b':
        mask = np.zeros_like(A)
        mask[np.triu_indices_from(mask)] = True
        mask[np.diag_indices_from(mask)] = False
        with sns.axes_style("white"):
            sns.set(rc={'figure.figsize': (10, 10)})
            sns.set(font_scale=1.)
            sns.heatmap(A, mask=mask, vmax=np.max(A), annot=True, square=True,  cmap="YlGnBu")
    else:
        with sns.axes_style("white"):
            sns.set(rc={'figure.figsize': (10, 10)})
            sns.set(font_scale=1.)
            sns.heatmap(A, vmax=np.max(A), annot=True, square=True,  cmap="YlGnBu")

    plt.title(strategy, fontsize=14)
    plt.ylabel('Training task', fontsize=14)
    plt.xlabel('Evaluation task', fontsize=14)

    if "AutoEncoder" in model:
        model = "AutoEncoder"

    plt.savefig(f'logs/{distance_metric}/{dataset_name}_{strategy}_{params}_{model}_{transfer}.pdf', format='pdf')
    # plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
##################################################



# A = np.random.randint(3, 10, size=(5, 5))
# print(A)
#
#
# fwt = forward_transfer(A)
# bwt = backward_transfer(A)
# l_roc = lifelong_roc(A)
#
# print(fwt)
# print(bwt)
# print(l_roc)
#
# heatmap(A, "pv-italy", "Naive", "b")
# heatmap(A, "pv-italy", "Naive", "f")
# heatmap(A, "pv-italy", "Naive", "all")
