import os
import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

class module(object):
    def __init__(self):
        # read X_pca
        try:
            self.X_pca= pd.read_csv('X_pca.csv')
            print('Read X_pca successfully:\n',self.X_pca.head())
            self.X_pca = self.X_pca.to_numpy() 
        except FileNotFoundError as e:
            print('Can not find X_pca.csv.')
            sys.exit(0)

        # read labels
        try:
            self.labels= pd.read_csv('labels.csv')
            print('Read labels successfully:\n',self.labels.head())
            self.labels = self.labels.to_numpy() 
        except FileNotFoundError as e:
            print('Can not find labels.csv.')
            sys.exit(0)

        # read origin dataset
        try:
            self.scaled_songs= pd.read_csv('scaled_songs.csv')
            print('Read scaled_songs successfully:\n',self.scaled_songs.head())
            self.scaled_songs = self.scaled_songs.to_numpy()
        except FileNotFoundError as e:
            print('Can not find scaled_songs.csv.')
            sys.exit(0)

        # read filter_id_name
        try:
            self.filtered_id_name = pd.read_csv('filtered_id_name.csv')
            print('Read filtered_id_name successfully:\n',self.filtered_id_name.head())
        except FileNotFoundError as e:
            print('Can not find filtered_id_name.csv.')
            sys.exit(0)

        # load PCA
        self.pca = PCA(n_components=5) 
        self.pca.fit_transform(self.scaled_songs[:,1:])    

        # load GMM
        self.gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)
        self.training()

    # training GMM model
    def training(self):
        print('trainning ....')
        self.gmm.fit(self.X_pca)
        print('trainning accomplished.')

    def predict(self,song_id):
        print('predicting ...\n')
        song_ids = self.scaled_songs[:, 0]
        indices = np.where(song_ids == song_id)[0]
        if len(indices) != 1:
            print('Sorry, this song is not in our database.')
            sys.exit(0)
        data = self.scaled_songs[indices[0],:]
        data = data[1:]
        data = data.reshape(1, -1)
        data_pca = self.pca.transform(data)
        predicted_label = self.gmm.predict(data_pca)[0]

        # select all data belong to this labels
        cluster_indices = np.where(self.labels == predicted_label)[0]
        cluster_data = self.X_pca[cluster_indices]

        # using nearest neighbor search
        nbrs = NearestNeighbors(n_neighbors=10, metric='euclidean')
        nbrs.fit(cluster_data)
        distances, local_indices = nbrs.kneighbors(data_pca)
        
        recommended_indices = cluster_indices[local_indices[0]]
        recommended_rows = self.filtered_id_name.iloc[recommended_indices]
        print('What I recommend are: ',recommended_rows)
        return None