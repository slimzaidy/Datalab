import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 2, 12),
                linewidths=1, colors='k')

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                linewidths=2, colors='r', linestyles='dashed')
    
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

#upload the dat file
path = "5_Netzbasierte_Angriffserkennung\data_train.csv"
data = pd.read_csv(path)

data_new = data.rename(columns={data.columns[6]: 'Packets AB', data.columns[7]: 'Bytes AB', data.columns[8]: 'Bytes BA',data.columns[9]: "Bytes/s BA", data.columns[12]: "Bits/s AB", data.columns[13]: "Bits/s BA"})
#y = np.array(data_new["Address A"]+data_new["Address B"], data_new["Port A"])
data_new = data_new.drop(columns = data_new.columns[:4])
#print(data_new.head())
trafo = RobustScaler()
data_new = trafo.fit_transform(data_new)
pca = PCA(0.98)
pca.fit(data_new)
#print(pca.explained_variance_ratio_)
red_dim_data_frame = pca.fit_transform(data_new)
print(red_dim_data_frame.shape)
#print(pca.components_)


# model = KMeans(n_clusters=3)
# output = model.fit_predict(red_dim_data_frame) #.reshape(-1,1))
# centroids = model.cluster_centers_
# print(centroids)
# print(set(model.labels_))
# #plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'red')
# plt.scatter(np.arange(len(output)), output)
# #plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
# plt.show()
# plt.close()


gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(red_dim_data_frame)

print(gm.weights_)

print(f"{gm.converged_=}")


densities = gm.score_samples(red_dim_data_frame)
density_threshold = np.percentile(densities, 0.9)   #4
anomalies = red_dim_data_frame[densities < density_threshold]
#anomalies_indices = red_dim_data_frame.indexOf(densities < density_threshold)

plt.figure(figsize=(8, 4))

plot_gaussian_mixture(gm, red_dim_data_frame)
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='r', marker='*')
plt.ylim(top=5.1)

#save_fig("mixture_anomaly_detection_plot")
plt.show()