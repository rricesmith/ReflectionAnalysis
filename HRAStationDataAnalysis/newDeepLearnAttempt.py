#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans
import os
import configparser
import gc

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

######################################################
# 1. Data Loading & Z–score Normalization
######################################################
# Load your data. Assumed shape: (n_events, 4, 256)
config = configparser.ConfigParser() 
config.read(os.path.join('HRAStationDataAnalysis', 'config.ini')) 
date = config['PARAMETERS']['date']
station_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'nurFiles', date)
station_id = 13
data = []
for file in os.listdir(station_data_folder):
    if file.startswith(f'{date}_Station{station_id}_Times'):
        file_path = os.path.join(station_data_folder, file)
        print(f"Loading file: {file_path}")
        events = np.load(file_path, allow_pickle=True)
        mask = not np.any(events, axis=0)  # Check if any event is all zero
        print(events[mask])
        print(events[mask].shape)
        data.concatenate(events[mask])
        del events
        del mask
        gc.collect()  # Free up memory if necessary

plot_folder = f'HRAStationDataAnalysis/plots/{date}/DeepLearning/'
os.makedirs(plot_folder, exist_ok=True)

# data = np.load('path_to_data.npy')  # Adjust the path to your data

# Rearrange so that the time steps come first: (n_events, 256, 4)
data = np.transpose(data, (0, 2, 1))

# Global z–score normalization (you could choose per-channel normalization if desired)
data_mean = np.mean(data)
data_std  = np.std(data)
data_norm = (data - data_mean) / data_std

# (Optional) Use a subset for training if your dataset is huge
n_train = 100000
if data_norm.shape[0] > n_train:
    data_norm = data_norm[:n_train]

######################################################
# 2. Build the Encoder Network (Latent dimension = 4)
######################################################
input_shape = (256, 4)  # 256 time steps & 4 channels
inp = Input(shape=input_shape)

# A simple convolutional encoder:
x = Conv1D(16, kernel_size=3, activation='relu', padding='same')(inp)
x = MaxPooling1D(pool_size=2, padding='same')(x)   # -> (128, 16)
x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
x = MaxPooling1D(pool_size=2, padding='same')(x)     # -> (64, 32)
x = Flatten()(x)
latent = Dense(4, activation=None)(x)   # Set latent dimension to 4

encoder = Model(inputs=inp, outputs=latent, name='encoder')
encoder.summary()

######################################################
# 3. Define a Custom Clustering Layer
######################################################
class ClusteringLayer(Layer):
    def __init__(self, n_clusters, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters

    def build(self, input_shape):
        latent_dim = input_shape[-1]
        # Initialize cluster centers with shape (n_clusters, latent_dim)
        self.clusters = self.add_weight(name='clusters',
                                        shape=(self.n_clusters, latent_dim),
                                        initializer='glorot_uniform',
                                        trainable=True)
        super(ClusteringLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs: (batch_size, latent_dim)
        # Expand dims to compute pairwise distances between inputs and cluster centers.
        expanded_inputs = tf.expand_dims(inputs, axis=1)          # (batch, 1, latent_dim)
        expanded_clusters = tf.expand_dims(self.clusters, axis=0)    # (1, n_clusters, latent_dim)
        # Compute squared Euclidean distances.
        distances = tf.reduce_sum(tf.square(expanded_inputs - expanded_clusters), axis=2)
        # Compute soft assignments using Student's t-distribution (with degree=1).
        q = 1.0 / (1.0 + distances)
        q = q / tf.reduce_sum(q, axis=1, keepdims=True)
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_clusters)

# Set the expected number of clusters (here 4).
n_clusters = 4

# Create the DEC model: encoder followed by the clustering layer.
clustering_output = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
dec_model = Model(inputs=encoder.input, outputs=clustering_output)
dec_model.summary()

######################################################
# 4. Initialize Cluster Centers with KMeans
######################################################
# Obtain latent representations from the encoder.
latent_features = encoder.predict(data_norm, batch_size=256)

# Run KMeans on the latent features to initialize cluster centers.
kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
y_pred_initial = kmeans.fit_predict(latent_features)

# Set the KMeans cluster centers to the clustering layer.
dec_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
print("Initialized cluster centers via KMeans.")

######################################################
# 5. Define the Clustering Loss and Training Procedure
######################################################
# Compute the auxiliary target distribution p.
def target_distribution(q):
    weight = q ** 2 / tf.reduce_sum(q, axis=0)
    return tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, axis=1))

optimizer = Adam(learning_rate=1e-3)
dec_model.compile(optimizer=optimizer, loss='kullback_leibler_divergence')

# Training parameters.
maxiter = 10000          # Total training iterations.
update_interval = 140    # Update the target distribution every update_interval iterations.
batch_size = 256
index_array = np.arange(data_norm.shape[0])

loss_history = []

######################################################
# 6. DEC Training Loop
######################################################
print("Starting DEC training ...")
for ite in range(maxiter):
    # Select a random batch.
    batch_index = np.random.choice(index_array, size=batch_size, replace=False)
    x_batch = data_norm[batch_index]

    # Every update_interval iterations, update the target distribution.
    if ite % update_interval == 0:
        q_all = dec_model.predict(data_norm, batch_size=batch_size)
        p_all = target_distribution(q_all).numpy()
        # (Optional) Plot histogram of cluster assignments every few updates.
        if ite % (update_interval * 5) == 0:
            plt.figure()
            plt.hist(np.argmax(q_all, axis=1), bins=n_clusters, rwidth=0.8)
            plt.title(f'Iter {ite}: Cluster Assignment Histogram')
            plt.xlabel('Cluster')
            plt.ylabel('Count')
            plt.savefig(f'{plot_folder}/cluster_histogram_iter_{ite}.png')
            plt.close()
            # plt.show()

    p_batch = p_all[batch_index]

    # Train on batch.
    loss = dec_model.train_on_batch(x_batch, p_batch)
    loss_history.append(loss)

    if ite % 500 == 0:
        print(f"Iter {ite}: loss = {loss:.4f}")

# Plot the training loss history.
plt.figure(figsize=(8,4))
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('KL Loss')
plt.title('DEC Training Loss History')
plt.savefig(f'{plot_folder}/dec_training_loss.png')
plt.close()
# plt.show()

######################################################
# 7. Visualizing the 4-Dimensional Latent Space Distribution
######################################################
# Get final soft assignments for all data.
q_final = dec_model.predict(data_norm, batch_size=batch_size)
cluster_assignments = np.argmax(q_final, axis=1)

# The latent features are 4-D. Create pairwise 2D scatter plots.
fig, axes = plt.subplots(2, 3, figsize=(15,10))
axes = axes.flatten()
dim_labels = ['f1', 'f2', 'f3', 'f4']
pair_idx = 0
for i in range(4):
    for j in range(i+1, 4):
        ax = axes[pair_idx]
        ax.scatter(latent_features[:, i], latent_features[:, j],
                   c=cluster_assignments, s=3, cmap='viridis', alpha=0.7)
        ax.set_xlabel(dim_labels[i])
        ax.set_ylabel(dim_labels[j])
        ax.set_title(f'{dim_labels[i]} vs {dim_labels[j]}')
        pair_idx += 1
plt.tight_layout()
plt.savefig(f'{plot_folder}/latent_space_scatter.png')
# plt.show()

######################################################
# 8. Plot Example Signals from Each Cluster
######################################################
# For each cluster, randomly pick a few events and plot their signals.
n_examples = 3  # number of examples per cluster
time_axis = np.arange(data_norm.shape[1])  # 256 timesteps

for cluster in range(n_clusters):
    indices = np.where(cluster_assignments == cluster)[0]
    print(f"Cluster {cluster} has {len(indices)} events.")
    # Plot a few example events.
    example_indices = np.random.choice(indices, size=min(n_examples, len(indices)), replace=False)
    
    for idx in example_indices:
        signal = data_norm[idx]  # shape: (256, 4)
        plt.figure(figsize=(10,4))
        # Plot each channel on the same plot.
        for ch in range(signal.shape[-1]):
            plt.plot(time_axis, signal[:, ch], label=f'Channel {ch}')
        plt.title(f'Cluster {cluster}, Event Index {idx}')
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Amplitude')
        plt.legend()
        plt.savefig(f'{plot_folder}/cluster_{cluster}_event_{idx}.png')
        # plt.show()

######################################################
# 9. Save the Indices of Events for Each Cluster
######################################################
save_dir = 'cluster_indices'
os.makedirs(save_dir, exist_ok=True)

for cluster in range(n_clusters):
    indices = np.where(cluster_assignments == cluster)[0]
    file_path = os.path.join(save_dir, f'cluster_{cluster}_indices.npy')
    np.save(file_path, indices)
    print(f"Saved {len(indices)} indices for cluster {cluster} to {file_path}")

print("Clustering and saving of event indices complete.")