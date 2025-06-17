from sklearn.semi_supervised import LabelSpreading
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.semi_supervised import LabelSpreading
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import hdbscan
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import math
import numpy as np
from configuration import num_of_labeled_data



class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 9)
        )
        self.decoder = nn.Sequential(
            nn.Linear(9, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim)
        )
        self.classification_layer = nn.Sequential(
            nn.Linear(9, 3),
            nn.Softmax(dim=1)
        )
    # Forward pass remains the same
    def forward(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      classification = self.classification_layer(encoded)
      return decoded, classification
  

class Clustering:
    def __init__(self,data,labeled_data=None): # sample data : [prompted_q1_doc1,...] sample labeled data : ([prompted_q1_doc1,...],[label1,...])
        self.labels = []
        self.data = []
        self.LOW_LABEL = 0
        self.AMBIGOUS_LABEL = 1
        self.HIGH_LABEL = 2
        self.UNK_LABEL = -1
        for d , l in zip(labeled_data[0],labeled_data[1]):
            self.data.append(d)
            self.labels.append(l)
        for d in data:
            self.data.append(d)
            self.labels.append(self.UNK_LABEL) #for unknowns

    def start_hdbscan(self):
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=50, prediction_data=True)  # Save as attribute
        self.hdbscan.fit(self.data)
        return hdbscan.approximate_predict(self.hdbscan, self.data)[0]


    def start_dbscan(self):
        self.dbscan = DBSCAN(eps=0.5, min_samples=10).fit(self.data)
        return self.dbscan.labels_

    def start_kmeans(self):
        self.kmeans = KMeans(n_clusters=3,random_state=42,max_iter=2000).fit(self.data)
        print(self.kmeans.labels_)
        return self.kmeans.labels_

    def start_kmedoids(self):
        self.kmedoids = KMedoids(n_clusters=3, random_state=42, max_iter=1000, metric='cosine').fit(self.data)
        print(self.kmedoids.labels_)
        return self.kmedoids.labels_

    def start_agglomerative(self):
        self.agg = AgglomerativeClustering(n_clusters=3).fit(self.data)
        print(self.agg.labels_)
        return self.agg.labels_

    def start_label_spreading(self):
        self.label_spread = LabelSpreading(kernel="knn",n_neighbors=2,alpha=0.01,max_iter=2000)
        self.label_spread.fit(self.data, self.labels)
        output_labels = self.label_spread.transduction_
        print("output_labels: ",output_labels)
        output_label_array = np.asarray(output_labels)
        return output_label_array

    def train_autoencoder(self):
        data_tensor = torch.tensor(self.data, dtype=torch.float32)
        labels_tensor = torch.tensor(self.labels, dtype=torch.long)
        dataset = TensorDataset(data_tensor, labels_tensor)
        data_loader = DataLoader(dataset, batch_size=200, shuffle=True)
        model = Autoencoder(self.data.shape[1])
        reconstruction_loss_fn = nn.MSELoss()
        classification_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        reconstruction_factor = 10
        classification_factor = 0.9
        for epoch in range(30):
            total_reconstruction_loss = 0
            total_classification_loss = 0
            total_loss = 0
            for i, batch in enumerate(data_loader):
                input_data, labels = batch
                decoded, classification = model(input_data)
                reconstruction_loss = reconstruction_loss_fn(decoded, input_data)
                classification_loss = classification_loss_fn(classification, labels)
                loss = reconstruction_factor * reconstruction_loss + classification_factor * classification_loss
                total_reconstruction_loss += reconstruction_loss.item()
                total_classification_loss += (classification_loss.item() if not math.isnan(classification_loss.item()) else 0)
                total_loss += (reconstruction_factor * reconstruction_loss.item() + classification_factor * (classification_loss.item() if not math.isnan(classification_loss.item()) else 0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            average_reconstruction_loss = total_reconstruction_loss / len(data_loader)
            average_classification_loss = total_classification_loss / len(data_loader)
            average_loss = total_loss / len(data_loader)
            print(f'Epoch {epoch+1}, '
                f'Average Reconstruction Loss: {average_reconstruction_loss:.6f}, '
                f'Average Classification Loss: {average_classification_loss:.4f}, '
                f'Average Total Loss: {average_loss:.4f}')
            self.autoencoder = model


    def dimensionality_reduction(self):
      reduced_data = self.autoencoder.encoder(torch.tensor(self.data, dtype=torch.float32)).detach().numpy()
      return reduced_data

    def map_labels(self, real, predicted):
        # predicted_labels are much bigger then real_labels
        predicted_labels = predicted[:num_of_labeled_data]
        real_labels = real
        max_labels = [-1,-1,-1]
        _0_counts = [0,0,0]
        _1_counts = [0,0,0]
        _2_counts = [0,0,0]
        for i in range(len(predicted_labels)):
          if predicted_labels[i] == 0:
            if real_labels[i] == 0:
              _0_counts[0] += 1
            if real_labels[i] == 1:
              _0_counts[1] += 1
            if real_labels[i] == 2:
              _0_counts[2] += 1
        for i in range(len(predicted_labels)):
          if predicted_labels[i] == 1:
            if real_labels[i] == 0:
              _1_counts[0] += 1
            if real_labels[i] == 1:
              _1_counts[1] += 1
            if real_labels[i] == 2:
              _1_counts[2] += 1
        for i in range(len(predicted_labels)):
          if predicted_labels[i] == 2:
            if real_labels[i] == 0:
              _2_counts[0] += 1
            if real_labels[i] == 1:
              _2_counts[1] += 1
            if real_labels[i] == 2:
              _2_counts[2] += 1
        max_counts = np.array([np.max(_0_counts), np.max(_1_counts), np.max(_2_counts)])
        counts = np.array([_0_counts,_1_counts,_2_counts])
        sorted_indices = np.argsort(max_counts)[::-1]  # Sort in descending order
        max_labels[sorted_indices[0]] = np.argmax(np.array(counts[sorted_indices[0]]))
        if np.argmax(np.array(counts[sorted_indices[0]])) == np.argmax(np.array(counts[sorted_indices[1]])):
          max_labels[sorted_indices[1]] = np.argsort(np.array(counts[sorted_indices[1]]))[-2]
        else:
          max_labels[sorted_indices[1]] = np.argmax(np.array(counts[sorted_indices[1]]))

        if np.argmax(np.array(counts[sorted_indices[2]])) == np.argmax(np.array(counts[sorted_indices[0]])) or np.argmax(np.array(counts[sorted_indices[2]])) == np.argmax(np.array(counts[sorted_indices[1]])):
          max_labels[sorted_indices[2]] = np.argsort(np.array(counts[sorted_indices[2]]))[-2]
        else:
          max_labels[sorted_indices[2]] = np.argmax(np.array(counts[sorted_indices[2]]))

        if max_labels[sorted_indices[2]] == np.argmax(np.array(counts[sorted_indices[0]])) or max_labels[sorted_indices[2]] == np.argmax(np.array(counts[sorted_indices[1]])):
          max_labels[sorted_indices[2]] = np.argsort(np.array(counts[sorted_indices[2]]))[-3]

        print(counts)
        final_labels = []
        for p in predicted:
          final_labels.append(max_labels[p])
        return np.hstack((real[:num_of_labeled_data], final_labels[num_of_labeled_data:]))

    def evaluate_clustering_results(self, real_labels, predicted_labels, label_mappings=None):
        assert len(real_labels) == len(predicted_labels)
        max_labels = [-1,-1,-1]
        _0_counts = [0,0,0]
        _1_counts = [0,0,0]
        _2_counts = [0,0,0]
        for i in range(len(predicted_labels)):
          if predicted_labels[i] == 0:
            if real_labels[i] == 0:
              _0_counts[0] += 1
            if real_labels[i] == 1:
              _0_counts[1] += 1
            if real_labels[i] == 2:
              _0_counts[2] += 1
        for i in range(len(predicted_labels)):
          if predicted_labels[i] == 1:
            if real_labels[i] == 0:
              _1_counts[0] += 1
            if real_labels[i] == 1:
              _1_counts[1] += 1
            if real_labels[i] == 2:
              _1_counts[2] += 1
        for i in range(len(predicted_labels)):
          if predicted_labels[i] == 2:
            if real_labels[i] == 0:
              _2_counts[0] += 1
            if real_labels[i] == 1:
              _2_counts[1] += 1
            if real_labels[i] == 2:
              _2_counts[2] += 1
        max_counts = np.array([np.max(_0_counts), np.max(_1_counts), np.max(_2_counts)])
        counts = np.array([_0_counts,_1_counts,_2_counts])
        # Combine and sort by the maximum counts
        sorted_indices = np.argsort(max_counts)[::-1]  # Sort in descending order
        max_labels[sorted_indices[0]] = np.argmax(np.array(counts[sorted_indices[0]]))
        if label_mappings is None:
          if np.argmax(np.array(counts[sorted_indices[0]])) == np.argmax(np.array(counts[sorted_indices[1]])):
            max_labels[sorted_indices[1]] = np.argsort(np.array(counts[sorted_indices[1]]))[-2]
            max_counts[sorted_indices[1]] = np.sort(np.array(counts[sorted_indices[1]]))[-2]
          else:
            max_labels[sorted_indices[1]] = np.argmax(np.array(counts[sorted_indices[1]]))
          if np.argmax(np.array(counts[sorted_indices[2]])) == np.argmax(np.array(counts[sorted_indices[0]])) or np.argmax(np.array(counts[sorted_indices[2]])) == np.argmax(np.array(counts[sorted_indices[1]])):
            max_labels[sorted_indices[2]] = np.argsort(np.array(counts[sorted_indices[2]]))[-2]
            max_counts[sorted_indices[2]] = np.sort(np.array(counts[sorted_indices[2]]))[-2]
          else:
            max_labels[sorted_indices[2]] = np.argmax(np.array(counts[sorted_indices[2]]))
          if max_labels[sorted_indices[2]] == np.argmax(np.array(counts[sorted_indices[0]])) or max_labels[sorted_indices[2]] == np.argmax(np.array(counts[sorted_indices[1]])):
            max_labels[sorted_indices[2]] = np.argsort(np.array(counts[sorted_indices[2]]))[-3]
            max_counts[sorted_indices[2]] = np.sort(np.array(counts[sorted_indices[2]]))[-3]
        else:
          max_labels=label_mappings
        return float((max_counts[0] + max_counts[1] + max_counts[2]) * 100 / len(real_labels))