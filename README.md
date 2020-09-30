# Face Detection and Clustering in Video

Given a video as input this algorithm aims at clustering faces, based on the FaceNet face
recognition model.


This is a TensorFlow implementation of the face recognizer described in the paper
["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832). The project also uses ideas from the paper ["A Discriminative Feature Learning Approach for Deep Face Recognition"](http://ydwen.github.io/papers/WenECCV16.pdf) as well as the paper ["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) from the [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) at Oxford.

## Abstract
Given a video as input the presented algorithm aims at clustering faces, based on the
FaceNet face recognition model.    
  
The first stage of this method consists of extracting specific frames from the video,
locating all faces contained in each frame and calculating the feature vectors from those
faces using FaceNet pretrained deep neural network.    
  
Afterwards, our method clusters the face vectors using K-means. The main disadvantage
of the K-means algorithm is that the number of clusters, K, must be supplied as a
parameter. In this paper we use a simple validity measure based on the intra-cluster and
inter-cluster distance measures, which allows the number of clusters to be determined
automatically.     
  
In the final stage, the method puts emphasis on outlier faces and treats them in a special
manner

## Requirements

Check [requirements.txt](https://github.com/harpap/video-face-clusters/blob/silhouette-manage-outliers/requirements.txt)

## Usage
Run from cmd or powershell to load GUI:  
`python src/compare.py`
