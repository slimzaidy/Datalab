import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

adressA_list =[]
adressB_list =[]
portA_list = []
portB_list = []
packets_list = []
Bytes_list= []
Bytes_A_B_list = []
Bytes_B_A_list = []
Realstart_list = []
duration_list = []
bits_A_B_list = []
bits_B_A_list = []

#upload the dat file
path = "5_Netzbasierte_Angriffserkennung\data_train.csv"
data = pd.read_csv(path)

"""
data["Address A"] = adressA_list
data["Address B"] = adressB_list
data["Port A"] = portA_list
data["Port B"] = portB_list
data["Packets"] = packets_list
data["Duration"] = duration_list
data["Bytes"] = Bytes_list
data["Real Start"] = Realstart_list
data["Bits/s A → B"] = bits_A_B_list
data["Bits/s B → A"] = bits_B_A_list
data["Bytes/s A → B"] = Bytes_A_B_list
data["Bytes/s B → A"] = Bytes_B_A_list
"""
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

# X_train, X_test, y_train, y_test = train_test_split(
#     red_dim_data_frame,
#     test_size=0.1,
#     shuffle=False)
#X_train = red_dim_data_frame[:int(0.9*len(red_dim_data_frame))]
#X_test = red_dim_data_frame[int(0.9*len(red_dim_data_frame)):]

model = KMeans(n_clusters=3)
output = model.fit_predict(red_dim_data_frame) #.reshape(-1,1))
centroids = model.cluster_centers_
print(centroids)
print(set(model.labels_))
#print(output.shape)
#plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'red')
plt.scatter(np.arange(len(output)), output)
#plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.show()
plt.close()
#accuracy = accuracy_score(y_test.values.ravel(), y_hat)
#print(accuracy)
