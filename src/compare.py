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

from shutil import copy
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
    sys.stdout = open(os.path.dirname(os.path.realpath(__file__))+'/output.txt', 'w+') #redirect output
    output_dir_vid = os.path.expanduser(args.output_dir + '/video')
    if not os.path.exists(output_dir_vid):
        os.makedirs(output_dir_vid)
    video_path = 'C:/Users/Harris/Documents/GitHub/facenet-private/Cristiano Ronaldo.mp4'
    #frame_getter(video_path,output_dir_vid)
    dataset = facenet.get_dataset(args.output_dir)
    images = load_and_align_data(dataset[0].image_paths, args.image_size, args.margin, args.gpu_memory_fraction)
    emb = run_forward_pass(images, args.model)
    
    #silhouette
    #----------
    cluster_labels, silhouette_avg, sample_silhouette_values, best_cl = kMSilhouette(emb)
    #fix outliers
    outlier_kind, two_m_clusters = outliers(cluster_labels[best_cl], best_cl+2, sample_silhouette_values[best_cl], 
                                                                      silhouette_avg[best_cl], args.outlierConstant, video_path,
                                                                      output_dir_vid, dataset[0].image_paths)
    nrof_images = len(dataset[0].image_paths)
    if outlier_kind==1:
        dataset = facenet.get_dataset(args.output_dir)
        images = load_and_align_data(dataset[0].image_paths, args.image_size, args.margin, args.gpu_memory_fraction)
        emb = run_forward_pass(images, args.model)
        cluster_labels, silhouette_avg, sample_silhouette_values, best_cl = kMSilhouette(emb)
    elif two_m_clusters:        #an exei stoixeia mesa
        dataset = facenet.get_dataset(args.output_dir)
        images = load_and_align_data(dataset[0].image_paths, args.image_size, args.margin, args.gpu_memory_fraction)
        emb = run_forward_pass(images, args.model)
        #de douleuei o tropos p th thelame opote to paw opws prin
        cluster_labels, silhouette_avg, sample_silhouette_values, best_cl = kMSilhouette(emb)
        '''emb2=[]
        while two_m_clusters:
            two_m_cluster = two_m_clusters.pop()
            for j in range(nrof_images):
                if (two_m_cluster == cluster_labels[j]):   #palio label, xalaei logo twn kainouriwn eikonwn!!
                    emb2.append(emb[j])
            clusterer = KMeans(n_clusters=2, random_state=10)
            cluster_labels_2 = clusterer.fit_predict(emb2)
            for j in range(nrof_images):
                if (two_m_cluster == cluster_labels[j]):
                    if cluster_labels_2.pop(0) == 1:
                        cluster_labels[j] = best_cl+3    #allazw mono gia to 2o label gt to 1o de me noiazei n meinei idio
    '''
    
    output_dir_cluster = [None] * (best_cl+2)
    for i in range(best_cl+2):
        output_dir_cluster[i] = os.path.expanduser(args.output_dir + '/omada '+str(i))
        if not os.path.exists(output_dir_cluster[i]):
            os.makedirs(output_dir_cluster[i])
        if not os.path.exists(output_dir_cluster[i]+' (cropped)'):
            os.makedirs(output_dir_cluster[i]+' (cropped)')
    for j in range(nrof_images):
        r,g,b = cv2.split(images[j])
        img2 = cv2.merge([b*255,g*255,r*255])
        #path manipulation for imwrite
        outImWr=output_dir_cluster[cluster_labels[best_cl][j]]+' (cropped)'+'/'+os.path.basename(dataset[0].image_paths[j])
        cv2.imwrite(outImWr,img2)
        copy(dataset[0].image_paths[j],output_dir_cluster[cluster_labels[best_cl][j]])
    print(sample_silhouette_values[2])
    print('\n')
    print(cluster_labels[best_cl])



def frame_getter(vid, output_dir, frame):
    cap = cv2.VideoCapture(vid)
    if frame is None:
        frame_interval = 300  # Number of frames after which to save
        frame_count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
                if (frame_count % frame_interval) == 0:
                    cv2.imwrite(output_dir+ "/frame-" + str(frame_count) + ".jpg", frame)
                frame_count+=1
            else:
                break
    else:
        total_frames = cap.get(7) #to thelw g an einai sto telos
        cap.set(1, frame-2)
        for i in range(2):
            ret, frame_read = cap.read()
            cv2.imwrite(output_dir+ "/frame-" + str(int(cap.get(1)-1)) + ".jpg", frame_read)
        if total_frames > frame + 2:
            cap.read()              #apla proxwraei g na mn ksanagrapsei tin idia eikona
            for i in range(2):
                ret, frame_read = cap.read()
                cv2.imwrite(output_dir+ "/frame-" + str(int(cap.get(1)-1)) + ".jpg", frame_read)
            # kai meta ksanatrexw ooolo apo tin arxi g ti 1i periptwsi
    # When everything is done, release the capture
    cap.release()
    
def run_forward_pass(images, model):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            return sess.run(embeddings, feed_dict=feed_dict)

def kMSilhouette(emb):
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] #tha ithela n dokimasw oso megalwnoun taclusters
                                            #an xeirotereuoun ta apotelesmata n stamataeiautomata
    j=0
    cluster_labels = [None] * len(range_n_clusters)
    silhouette_avg = [None] * len(range_n_clusters)
    sample_silhouette_values = [None] * len(range_n_clusters)
    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels[j] = clusterer.fit_predict(emb)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg[j] = silhouette_score(emb, cluster_labels[j])
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg[j])

        # Compute the silhouette scores for each sample
        sample_silhouette_values[j] = silhouette_samples(emb, cluster_labels[j])
        j+=1
    #best cluster
    best_cl=silhouette_avg.index(max(silhouette_avg))
    print('best number of clusters: ',best_cl+2)
    return cluster_labels, silhouette_avg, sample_silhouette_values, best_cl
    
def outliers(cluster_labels, max_clusters, sample_silhouette_values, silhouette_avg, N, video_path, output_dir_vid, image_paths):
    nrof_images = len(cluster_labels)
    difference = [0] * (nrof_images)
    outl_sum = 0
    outl_cluster_sum = [0] * (max_clusters)   #auta
    sum_of_images_in_cluster = [0] * (max_clusters) #mallon de xreiazonte pinakes n einai
    return_val1 = 0
    return_val2 = []
    for i in range(nrof_images):
        if sample_silhouette_values[i] < silhouette_avg - N :
            difference[i] = silhouette_avg - sample_silhouette_values[i]
            outl_sum += 1
    
    for i in range(max_clusters):
        for j in range(nrof_images):
            if (i==cluster_labels[j]):
                sum_of_images_in_cluster[i] += 1
                if difference[j]!=0:
                    #vazw poia frames einai
                    _,video = image_paths[j].split('video')
                    _,start_of_frame = video.split('-')
                    frame_nr,_ = start_of_frame.split('.')
                    frame_getter(video_path, output_dir_vid, int(frame_nr)) #pare epipleon frames
                    outl_cluster_sum[i] += 1
        if (outl_cluster_sum[i]/sum_of_images_in_cluster[i]) >= 0.7:
            print ("periptwsi 1 gia to cluster: " + str(i))
            return_val1 = 1
        elif outl_cluster_sum[i]==1:
            return_val2.append(i)
            #2-means
        print (outl_cluster_sum[i])
        print (sum_of_images_in_cluster)
    return return_val1 ,return_val2


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 80 # minimum size of face
    threshold = [ 0.8, 0.9, 0.9 ]  # three steps's threshold
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
            pos.append(image_paths[i])
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
        image_paths.remove(x)
    img_list=[x for x in img_list if x is not None]
    images = np.stack(img_list)
    return images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('output_dir', type=str, help='output directory')
    parser.add_argument('--outlierConstant', type=float, help='Constant for fixing outliers. The bigger the harder to find outlier', default=0.45)
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

#de xreiazonte ta plots
#interface, database me polla video

#gia to detect_face:davidsandberg:
#There are five landmarks detected by MTCNN and these are left eye, right eye, nose, left mouth corner, and right mouth corner.
# It would not be straight forward to detect a larger number of landmarks.

#diorthwsa ta false detect k ta minsize
#PREPEI na to testarw k sta alla video
#na sigourepsw oti den mou aporriptei "kala" proswpa

#gia:   sample_silhouette_values[j] = silhouette_samples(emb, cluster_labels[j])
#The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.