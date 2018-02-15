"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import time
import argparse
import facenet
import align.detect_face
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(args):
    #sys.stdout = open(os.path.dirname(os.path.realpath(__file__))+'/Distance.txt', 'w+') #redirect output
    output_dir_vid = os.path.expanduser(args.output_dir + '/video')
    if not os.path.exists(output_dir_vid):
        os.makedirs(output_dir_vid)
    #frameGetter('C:/Users/computer science/Downloads/Brad Pitt- Between Two Ferns with Zach Galifianakis.mp4',output_dir_vid)
    dataset = facenet.get_dataset(args.output_dir)
    images = load_and_align_data(dataset[0].image_paths, args.image_size, args.margin, args.gpu_memory_fraction)
    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(args.model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            '''
            print('embeddings:')
            print(emb)
            nrof_images = len(dataset[0].image_paths)

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, dataset[0].image_paths[i]))
            print('')
            
            # Print distance matrix
            print('Distance matrix')
            print('         ', end='')
            for i in range(nrof_images):
                print('   %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                    print('  %1.4f  ' % dist, end='')
                print('')'''

            #kmeans me sklearn
            kmeans = KMeans(n_clusters=args.clusters, random_state=33,max_iter=1000000000,n_init =30, init='random',tol=0.00000001).fit(emb)
            print (kmeans.labels_)
            nrof_images = len(dataset[0].image_paths)
            '''print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, dataset[0].image_paths[i]))
            print('')
            for i in range(kmeans.n_clusters):
                print ("omada : %d"% i)
                for j in range(nrof_images):
                    if (i==kmeans.labels_[j]):
                        print (dataset[0].image_paths[j])
            print (kmeans.inertia_)'''
            #silhouette
            silhouette_avg = silhouette_score(emb, kmeans.labels_)
            print(silhouette_avg)
            #----------
            range_n_clusters = [2, 3, 4, 5, 6] #tha ithela n dokimasw oso megalwnoun taclusters
                                            #an xeirotereuoun ta apotelesmata n stamataeiautomata
            for n_clusters in range_n_clusters:
                # Create a subplot with 1 row and 2 columns
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.set_size_inches(18, 7)

                # The 1st subplot is the silhouette plot
                ax1.set_xlim([-1, 1])
                # The (n_clusters+1)*10 is for inserting blank space between silhouette
                # plots of individual clusters, to demarcate them clearly.
                ax1.set_ylim([0, len(emb) + (n_clusters + 1) * 10])

                # Initialize the clusterer with n_clusters value and a random generator
                # seed of 10 for reproducibility.
                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                cluster_labels = clusterer.fit_predict(emb)

                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed
                # clusters
                silhouette_avg = silhouette_score(emb, cluster_labels)
                print("For n_clusters =", n_clusters,
                      "The average silhouette_score is :", silhouette_avg)

                # Compute the silhouette scores for each sample
                sample_silhouette_values = silhouette_samples(emb, cluster_labels)

                y_lower = 10
                for i in range(n_clusters):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = \
                        sample_silhouette_values[cluster_labels == i]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = cm.spectral(float(i) / n_clusters)
                    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                      0, ith_cluster_silhouette_values,
                                      facecolor=color, edgecolor=color, alpha=0.7)

                    # Label the silhouette plots with their cluster numbers at the middle
                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

                ax1.set_title("The silhouette plot for the various clusters.")
                ax1.set_xlabel("The silhouette coefficient values")
                ax1.set_ylabel("Cluster label")

                # The vertical line for average silhouette score of all the values
                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                ax1.set_yticks([])  # Clear the yaxis labels / ticks
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                # 2nd Plot showing the actual clusters formed
                colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
                ax2.scatter(emb[:, 0], emb[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                            c=colors, edgecolor='k')

                # Labeling the clusters
                centers = clusterer.cluster_centers_
                # Draw white circles at cluster centers
                ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                            c="white", alpha=1, s=200, edgecolor='k')

                for i, c in enumerate(centers):
                    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                s=50, edgecolor='k')

                ax2.set_title("The visualization of the clustered data.")
                ax2.set_xlabel("Feature space for the 1st feature")
                ax2.set_ylabel("Feature space for the 2nd feature")

                plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                              "with n_clusters = %d" % n_clusters),
                             fontsize=14, fontweight='bold')

                plt.show()

def frameGetter(vid,output_dir):
    frame_interval = 1000  # Number of frames after which to save
    frame_rate = 0
    frame_count = 0
    cap = cv2.VideoCapture(vid)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            if (frame_count % frame_interval) == 0:
                cv2.imwrite(output_dir+ "/frame-" + str(frame_count) + ".jpg", frame)
            frame_count+=1
        else:
            break
    # When everything is done, release the capture
    cap.release()


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    j=0
    pos=[]
    for i in range(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if bounding_boxes.size==0:
            print('image:'+image_paths[i]+'\n has not a detectable face')
            pos.append(i)
        else:
            det = np.squeeze(bounding_boxes[0,0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            '''r,g,b = cv2.split(prewhitened)
            img2 = cv2.merge([b,g,r])
            cv2.imshow('image'+str(j),img2)   
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
            img_list[j] = prewhitened
            j+=1
    for x in pos:
        image_paths.pop(x)
    img_list=[x for x in img_list if x is not None]
    images = np.stack(img_list)
    return images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('output_dir', type=str, help='output directory')
    parser.add_argument('clusters', type=int, help='number of faces')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
#apo ta palia i mikroteri apostasi =0.7680
#sto interview 0.2856
#me k-means (mathisi xwris epivlepsi dld den exw etiketes) sta emb na lew poses omades thelw kai na kanei clustering

#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
#http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html