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
from shutil import rmtree
from scipy import misc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances_argmin_min
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
from tkinter import filedialog
from tkinter import messagebox
from tkinter import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(args):
    global dataset2, cluster_labels, video_path, model, output_dir
    model = args.model
    output_dir = args.output_dir
    dataset2 =[]
    #sys.stdout = open(os.path.dirname(os.path.realpath(__file__))+'/output.txt', 'w+') #redirect output
    video_path = 'E:/OneDrive/Documents/videos/genimata.avi'
    gui()
    output_dir_vid = os.path.expanduser(output_dir + '/video')
    if not os.path.exists(output_dir_vid):
        os.makedirs(output_dir_vid)
    #frame_getter(video_path,output_dir_vid)
    dataset = facenet.get_dataset(output_dir)
    images = load_and_align_data(dataset[0].image_paths, args.image_size, args.margin, args.gpu_memory_fraction)
    emb = run_forward_pass(images, model)
    
    #silhouette
    #----------
    cluster_labels, sample_silhouette, best_cl, centers = kMSilhouette(emb)
    #fix outliers
    for i in range(3):
        outlier_kind, two_m_clusters = outliers(cluster_labels, best_cl+2, sample_silhouette, args.outlierConstant, video_path,
                                                                          output_dir_vid, dataset[0].image_paths)
        
        if outlier_kind:
            images2 = load_and_align_data(dataset2, args.image_size, args.margin, args.gpu_memory_fraction)
            emb2 = run_forward_pass(images2, model)
            dataset[0].image_paths = dataset[0].image_paths + dataset2
            images = np.concatenate((images, images2))
            emb = np.concatenate((emb, emb2))
            cluster_labels, sample_silhouette, best_cl, centers = kMSilhouette(emb)
            dataset2 = []
        else:
            break
    
    if two_m_clusters:        #an exei stoixeia mesa
        # (TO TSEKARW ME adonis.avi)
        images2 = load_and_align_data(dataset2, args.image_size, args.margin, args.gpu_memory_fraction)
        emb2 = run_forward_pass(images2, model)
        dataset[0].image_paths = dataset[0].image_paths + dataset2
        images = np.concatenate((images, images2))
        emb = np.concatenate((emb, emb2))
        nrof_images = len(dataset[0])
        while two_m_clusters:
            emb2 = []
            best_cl += 1
            two_m_cluster = two_m_clusters.pop()
            for j in range(nrof_images):
                if (two_m_cluster == cluster_labels[j]):
                    emb2.append(emb[j])
            clusterer = KMeans(n_clusters=2, random_state=10)
            cluster_labels_2 = clusterer.fit_predict(emb2)
            centers = np.concatenate((centers, clusterer.cluster_centers_))
            for j in range(nrof_images):
                if (two_m_cluster == cluster_labels[j]):  #oi kainouries oi eikones dn exoun two_m_cluster
                    first, cluster_labels_2 = cluster_labels_2[0], cluster_labels_2[1:]   #pop
                    if first == 1:
                        cluster_labels[j] = best_cl + 1    #allazw mono gia to 2o label gt to 1o de me noiazei n meinei idio
    
    nrof_images = len(dataset[0])
    output_dir_cluster = [None] * (best_cl+2)
    output_summary = [None] * (best_cl+2)
    closest, _ = pairwise_distances_argmin_min(centers, emb) #theseis [133  47 150 185 150]
    closest = np.unique(closest)
    for i in range(best_cl+2):
        output_dir_cluster[i] = os.path.expanduser(output_dir + '/omada '+str(i))
        output_summary[i] = os.path.expanduser(output_dir + '/SUMMARY/omada '+str(i))
        if not os.path.exists(output_dir_cluster[i]):
            os.makedirs(output_dir_cluster[i])
        if not os.path.exists(output_dir_cluster[i]+' (cropped)'):
            os.makedirs(output_dir_cluster[i]+' (cropped)')
        if not os.path.exists(output_summary[i]):
            os.makedirs(output_summary[i])
    for j in range(nrof_images):
        r,g,b = cv2.split(images[j])
        img2 = cv2.merge([b*255,g*255,r*255])
        #path manipulation for imwrite
        outImWr=output_dir_cluster[cluster_labels[j]]+' (cropped)'+'/'+os.path.basename(dataset[0].image_paths[j])
        cv2.imwrite(outImWr,img2)
        copy(dataset[0].image_paths[j],output_dir_cluster[cluster_labels[j]])
    for x in closest:
        r,g,b = cv2.split(images[x])
        img2 = cv2.merge([b*255,g*255,r*255])
        #path manipulation for imwrite
        outImWr=output_summary[cluster_labels[x]]+'/(cropped)'+os.path.basename(dataset[0].image_paths[x])
        cv2.imwrite(outImWr,img2)
        fig = plt.figure() 
        fig.canvas.set_window_title('cluster: ' + str(cluster_labels[x]))
        plt.imshow(images[x], interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        copy(dataset[0].image_paths[x],output_summary[cluster_labels[x]])
    show_del_gui()


def frame_getter(vid, output_dir, frame = None, cl = None):
    global cluster_labels
    cap = cv2.VideoCapture(vid)
    
    def write_img():
        ret, frame_read = cap.read()
        string = output_dir+ "/frame-" + str(int(cap.get(1)-1)) + ".jpg"
        cv2.imwrite(string, frame_read)
        dataset2.append(string)
    
    if frame is None:
        frame_interval = 50  # Number of frames after which to save
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
        if frame > 2:
            cap.set(1, frame-2)
        else:
            cap.set(1, frame+1)
            for i in range(2):
                write_img()
                cluster_labels = np.append(cluster_labels,cl)
        for i in range(2):
            write_img()
            cluster_labels = np.append(cluster_labels,cl)
        if total_frames > frame + 2:
            cap.read()              #apla proxwraei g na mn ksanagrapsei tin idia eikona
            for i in range(2):
                write_img()
                cluster_labels = np.append(cluster_labels,cl)
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
    clusterer = [None] * len(range_n_clusters)
    cluster_labels = [None] * len(range_n_clusters)
    silhouette_avg = [None] * len(range_n_clusters)
    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer[j] = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels[j] = clusterer[j].fit_predict(emb)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg[j] = silhouette_score(emb, cluster_labels[j])
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg[j])
        j+=1
        
    #best cluster
    best_cl=silhouette_avg.index(max(silhouette_avg))
    print('best number of clusters: ',best_cl+2)
    
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(emb, cluster_labels[best_cl])
    
    # Find variance 
    #var = np.amin(clusterer[best_cl].transform(emb), axis=1)      #i transform gurnaei apostaseis olwn twn kentrwn k pairnw
                                                                  #min wste na kratisw tin apostasi tou dikou tou kentrou
    
    return cluster_labels[best_cl], sample_silhouette_values, best_cl, clusterer[best_cl].cluster_centers_
    
def outliers(cluster_labels, max_clusters, sample_sil, N, video_path, output_dir_vid, image_paths):
    nrof_images = len(cluster_labels)
    outl_sum = 0
    outl_cluster_sum = 0
    sum_of_images_in_cluster = 0
    return_val1 = 0
    return_val2 = []
    for i in range(max_clusters):
        for j in range(nrof_images):
            if (i==cluster_labels[j]):
                sum_of_images_in_cluster += 1
                if sample_sil[j] <= N :
                    #vazw poia frames einai
                    _,video = image_paths[j].split('video')
                    _,start_of_frame = video.split('-')
                    frame_nr,_ = start_of_frame.split('.')
                    frame_getter(video_path, output_dir_vid, int(frame_nr), i) #pare epipleon frames
                    outl_cluster_sum += 1
                    outl_sum += 1
        if (outl_cluster_sum/sum_of_images_in_cluster) >= 0.4:
            print ("periptwsi 1 gia to cluster: " + str(i))
            return_val1 = 1
        elif outl_cluster_sum==1:
            return_val2.append(i)
            #2-means
        print ('posa outliers exw sto cluster ' + str(outl_cluster_sum) + ' vs')
        print ('sunolo eikonwn sauto to cluster ' + str(sum_of_images_in_cluster))
        print('\n')
        outl_cluster_sum = 0
        sum_of_images_in_cluster = 0
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
        elif bounding_boxes.shape[0]!=1:
            print('image:'+image_paths[i]+'\n has more than one face')
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
        try:
            os.remove(x)        #diagrafw ta frames xwris proswpa
            image_paths.remove(x)
        except FileNotFoundError: 
            print("warning: file not found (already deleted)")
    img_list=[x for x in img_list if x is not None]
    images = np.stack(img_list)
    return images

def gui():

    def BVideoFunction():
        global video_path
        video_path =  filedialog.askopenfilename(initialdir = os.path.dirname(video_path), title = "Choose Video source",filetypes = (("avi files","*.avi"),("mp4 files","*.mp4"),("all files","*.*")))

    def BOutpDirFunction():
        global output_dir
        output_dir =  filedialog.askdirectory(initialdir = output_dir)

    def BModelFunction():
        global model
        model =  filedialog.askopenfilename(initialdir = os.path.dirname(os.path.realpath(__file__)) ,title = "Give the path of the model",filetypes = (("pb files","*.pb"),("all files","*.*")))

    def BRunFunction():
        global video_path
        global output_dir
        try:
            print (video_path)
            print (output_dir)
        except NameError: 
            print("vale label")
        root.destroy()
        
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()
            exit()


    root = Tk(className = ' Finding Distinct Faces')
    root.configure(background='#A3CEDC')

    BVideo = Button(root, text ="    Choose Video source    ", command = BVideoFunction)
    BVideo.grid(ipadx=3, ipady=3, padx=4, pady=4)

    BOutpDir = Button(root, text ="Choose Output directory \n(Needs to be empty)", command = BOutpDirFunction)
    BOutpDir.grid(ipadx=2, ipady=2, padx=4, pady=4)

    LFrames = Label( root, text='Frames after which\n to exrtact image:' )
    LFrames.grid(column=2, row=0, ipadx=2, ipady=2, padx=4, pady=4)
    frames = StringVar()
    frames.set(50)
    EFrames = Entry(root, bd =5, textvariable = frames)
    EFrames.grid(column=3, row=0, ipadx=2, ipady=2, padx=4, pady=4)

    LConstant = Label( root, text='Silhouette constant for locating outliers\n(Recommendation: Do not modify):' )
    LConstant.grid(column=2, row=1, ipadx=1, ipady=1, padx=4, pady=4)
    constant = StringVar()
    constant.set(0.1)
    EConstant = Entry(root, bd =5, textvariable = constant)
    EConstant.grid(column=3, row=1, ipadx=2, ipady=2, padx=4, pady=4)
    
    BModel = Button(root, text ="Give the path of the model protobuf (.pb) file", command = BModelFunction)
    BModel.grid(row=3, column=0, ipadx=2, ipady=2, padx=4, pady=4)

    BRun = Button(root, text ="RUN", command = BRunFunction)
    BRun.grid(column=2, row=4, ipadx=3, ipady=3, padx=4, pady=4)
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
    
def show_del_gui():
    def BShowFunction():
        global output_dir
        filedialog.askopenfile(mode="r", initialdir = output_dir)
        
    def BDelFunction():
        global output_dir
        if messagebox.askokcancel("Delete", "Are you sure you want to delete all results?"):
            for the_file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, the_file)
                try:
                    if os.path.isdir(file_path): rmtree(file_path)
                except Exception as e:
                    print(e)
        
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()
            exit()

    root = Tk(className = ' Finding Distinct Faces')
    root.configure(background='#A3CEDC')

    BShow = Button(root, text ="Show results", command = BShowFunction)
    BShow.grid(ipadx=3, ipady=3, padx=4, pady=4)
    
    BDel = Button(root, text ="Delete results", command = BDelFunction)
    BDel.grid(ipadx=2, ipady=2, padx=4, pady=4)
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file', default='~/models/facenet/20170512-110547/20170512-110547.pb')
    parser.add_argument('--output_dir', type=str, help='output directory', default='C:/Users/Harris/Documents/GitHub/facenet-private/frames')
    parser.add_argument('--outlierConstant', type=float, help='Constant for fixing outliers. The smaller the harder to find outlier', default=0.1)
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

#fakelo summary me proswpo pio konta sto meso oro
#16/9 na mn ksexasw na svinei ta epipleon frames p pairnw
