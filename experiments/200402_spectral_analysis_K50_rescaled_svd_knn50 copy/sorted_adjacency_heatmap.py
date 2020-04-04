import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

k_neighbors = 50

A_sorted = pickle.load(open("./data/A_sorted_K{}.p".format(k_neighbors),"rb"))
A_mc_sorted = pickle.load(open("./data/A_mc_sorted_K{}.p".format(k_neighbors),"rb"))

def plot_heatmap(sorted_m, matname):
	# blue_red_cmap = sns.diverging_palette(240, 10, as_cmap=True)
	mean = sorted_m.mean()

	sorted_m.data = np.where(sorted_m.data > 0, 1, 0)

	plt.figure(figsize=(80,60))

	idx = np.array([i for i in range(A_sorted.shape[0]) if i % 5 == 0])

	sns.heatmap(sorted_m[idx][:, idx].toarray(), cmap="Greys", cbar=False)
	fontsize = 110
	plt.xlabel("Patients sorted by cluster membership", fontsize=fontsize)
	plt.ylabel("Patients sorted by cluster membership", fontsize=fontsize)
	plt.tight_layout()
	plt.savefig("./tests/heatmap_{}.png".format(matname))

# plot_heatmap(A_sorted, "A_sorted")
plot_heatmap(A_mc_sorted, "A_mc_sorted")