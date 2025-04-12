# Collaborative and Content filtering for music recommendation from Spotify playlists
(Documentation updated April 10, 2025)

## Description
This is a Python project recommends music using collaborative filtering and content filtering. The input is one playlist of songs in any length and an output is one song recommended for the given playlist. This project started from SENG 474/CSC 503 Data Mining course at University of Victoria in 2025 Spring.

## Getting started
1. Run all cells in "data_merge.ipynb"
2. Run all cells in "Collaborative filtering.ipynb"
3. Run all cells in "Centent filtering masking.ipynb"
4. Run all cells in "Evaluation masking.ipynb"

## Dataset
2 dataset have been combined.

1. [Spotify Million Playlist Dataset Challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
2. [Spotify 1.2M+ Songs](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)

We merged two dataset data_merge.ipynb. The resulting merged data is new_features.csv. We merged so the merged data have song id, playlist id and features (tempo, danceability, energy...).

## Spotify
To understand collaborative filtering and content based filtering we need to understand the [Spotify](https://open.spotify.com/) platform. [Spotify](https://open.spotify.com/) is a music streaming platform with over 600 million (2024) users. Spotify is known for a feature "playlist". Playlist is list of songs. Users can create playlists and share the playlists online. Other users can listen to the playlists made by other users. Recently, recommendation algorithms have created significant impact on internet (TikTok, Instagram Reels, Youtube Shorts...). We believe we could incorporate these user created Spotify playlists to recommend music to the user base.

## Collaborative filtering
Collaborative filtering is if we integrate users' created playlists for a recommendation system, we are basically collborating with other users to create such trends in the music recommendation system. For exmaple, if we are to recommend musics just based on the characteristics of the musics such as tempo, danceability, energy, pitch, the recommendation logic will not understand a playlist of background musics for "Titanic". The songs in that playlist will have different characteristics and the recommended song will not relevent to the movie "Titanic". However, using playlists created by users, we can understand the concept of "Titanic" and recommend a music related to "Titanic". To do this we are using Gaussian Mixture Model (GMM). The reason why we are using Gaussian Mixture is because we want to cluster the musics but also want it to be a soft clustering to not strictly cluster musics into categories. Using Akaike Information Criterion (AIC) and Bayes Information Criterion (BIC) we can estimate how many clusters do we want for GMM which resulted in k=10.

1. Average the characteristics of all playlists
2. Use Gaussian Mixture to soft cluster all playlists

## Content filtering
Content filtering is recommending a song based on the content of the song. This means feature extraction such as tempo, danceability, energy, pitch... Aftering clustering the playlists, for an input playlist, we predict which cluster does the input playlist belong to, thereby scoping down the area for content filtering. Then we do content filtering using the songs in the cluster the input playlist belong to. For the content filtering we are using euclidean distance to score how related the songs are. 

1. List all the songs in the cluster predicted with input playlist
for song i in input playlist
    2. Use K-Nearest Neighbors (KNN) to recommend the best song for ith song
    3. Find a recommended song with the closest euclidean distance from the KNN
    4. Recommend that song

The current system is basically looking for the closest related song from any song in the playlist. The scoring system should be improved in the future considering when someone make a playlist, the next song they want to add to the playlist is usually most related to the song they added in the last.

## Evaluation
We are using masking for evaluation. First we mask the last song in the playlist. If the input playlist had N songs, it will now have N-1 songs. We run the collaborative filtering and the content filtering using the masked playlist. If the output recommended song match with the masked song, we count it as a hit. Otherwise, it is a miss. We do this for all the playlist in the dataset. Currently, the hit rate is 0.5%.

## Further documentation
The full report for this project can be found here.

## Who built this project
* David Kim
* Abhay Cheruthottathil
* Weiting Ye
* Yule Wang
* Adithyakrishna Arunkumar