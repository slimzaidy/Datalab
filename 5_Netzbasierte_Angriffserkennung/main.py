#########################################################################################
# Datalab
# Netzbasierte Angriffserkennung
# Gruppe: Hex Haxors
# Zaid Askari & Oussama Ben Taarit
#########################################################################################

import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm
import csv

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
path = "5_Netzbasierte_Angriffserkennung/data_test.csv"
data = pd.read_csv(path)

data_new = data.rename(columns={data.columns[6]: 'Packets AB', data.columns[7]: 'Bytes AB', data.columns[8]: 'Bytes BA',data.columns[9]: "Bytes/s BA", data.columns[12]: "Bits/s AB", data.columns[13]: "Bits/s BA"})
data_new = data_new.drop(columns = data_new.columns[:4])
trafo = RobustScaler()
data_new = trafo.fit_transform(data_new)
pca = PCA(0.98)
pca.fit(data_new)
red_dim_data_frame = pca.fit_transform(data_new)
#print(red_dim_data_frame.shape)

gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(red_dim_data_frame)
#print(gm.weights_)
#print(f"{gm.converged_=}")


densities = gm.score_samples(red_dim_data_frame)
density_threshold = np.percentile(densities, 0.7)   #0.9
anomalies = red_dim_data_frame[densities < density_threshold]
indices_malic = [i for i in range(len(densities)) if densities[i] < density_threshold]

plt.figure(figsize=(8, 4))

plot_gaussian_mixture(gm, red_dim_data_frame)
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='r', marker='*')
plt.ylim(top=5.1)

#save_fig("mixture_anomaly_detection_plot")
#plt.show()
#plt.close()

output_list = []

for i in range (0, len(data)): 
    label = 0
    if i in indices_malic:
        label = 1
    current_row_string = f"{data.loc[i ,'Address A']}:{data.loc[i ,'Port A']}->{data.loc[i ,'Address B']}:{data.loc[i ,'Port B']};{label}"
    #print(current_row_string)
    output_list.append(current_row_string)

output_df = pd.DataFrame()
output_df = output_df.append(output_list)


#output_df.to_csv("5_Netzbasierte_Angriffserkennung/5_output_experiment.csv", header= False, index = False)
f_csv = open("5_Netzbasierte_Angriffserkennung/output_5.csv", "w+", newline ='')
writer = csv.writer(f_csv, quoting=csv.QUOTE_ALL) 

for name in output_list:
    x = name 
    writer.writerow([x])

f_csv.close()