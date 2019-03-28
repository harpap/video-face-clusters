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

import PIL
from PIL import ImageTk
from shutil import copy, rmtree, move
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
import argparse
import facenet
import align.detect_face
from tkinter import filedialog, messagebox
from tkinter import *
from functools import partial
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(args):

    class Data:
        def __init__(self, image_path, image, cluster_label, emb, outlier):
            self.image_path = image_path
            self.image = image
            self.cluster_label = cluster_label
            self.emb = emb
            self.outlier = outlier
            
        def print_data(self):
            print(self.image_path)
            print(self.image)
            print(self.cluster_label)
            print(self.emb)
       
      
    def gui(video_path, output_dir, model, outl_const):

        def BVideoFunction(video_path):
            video_path =  filedialog.askopenfilename(initialdir = os.path.dirname(video_path), title = "Choose Video source",filetypes = (("avi files","*.avi"),("mp4 files","*.mp4"),("all files","*.*")))
            retArgs[0] = video_path
            
        def BOutpDirFunction(output_dir):
            output_dir =  filedialog.askdirectory(initialdir = output_dir)
            retArgs[1] = output_dir

        def BModelFunction(model):
            model =  filedialog.askopenfilename(initialdir = os.path.dirname(os.path.realpath(__file__)) ,title = "Give the path of the model",filetypes = (("pb files","*.pb"),("all files","*.*")))
            retArgs[2] = model

        def BRunFunction():
            retArgs[3] = int(EFrames.get())
            retArgs[4] = float(EConstant.get())
            root.destroy()
            
        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                root.destroy()
                exit()
        
        frame_interval = 50   # Number of frames after which to save
        retArgs = [video_path, output_dir, model, frame_interval, outl_const]
        root = Tk(className = ' Finding Distinct Faces')
        root.configure(background='#A3CEDC')

        BVideoFuncArg = partial(BVideoFunction, video_path)
        BVideo = Button(root, text ="    Choose Video source    ", command = BVideoFuncArg)
        BVideo.grid(ipadx=3, ipady=3, padx=4, pady=4)

        BOutpDirFuncArg = partial(BOutpDirFunction, output_dir)
        BOutpDir = Button(root, text ="Choose Output directory \n(Needs to be empty)", command = BOutpDirFuncArg)
        BOutpDir.grid(ipadx=2, ipady=2, padx=4, pady=4)

        LFrames = Label( root, text='Frames after which\n to exrtact image:' )
        LFrames.grid(column=2, row=0, ipadx=2, ipady=2, padx=4, pady=4)
        frames = StringVar()
        frames.set(retArgs[3])
        EFrames = Entry(root, bd =5, textvariable = frames)
        EFrames.grid(column=3, row=0, ipadx=2, ipady=2, padx=4, pady=4)

        LConstant = Label( root, text='Silhouette constant for locating outliers\n(Recommendation: Do not modify):' )
        LConstant.grid(column=2, row=1, ipadx=1, ipady=1, padx=4, pady=4)
        constant = StringVar()
        constant.set(outl_const)
        EConstant = Entry(root, bd =5, textvariable = constant)
        EConstant.grid(column=3, row=1, ipadx=2, ipady=2, padx=4, pady=4)
        
        BModelFuncArg = partial(BModelFunction, model)
        BModel = Button(root, text ="Give the path of the model protobuf (.pb) file", command = BModelFuncArg)
        BModel.grid(row=3, column=0, ipadx=2, ipady=2, padx=4, pady=4)

        BRun = Button(root, text ="RUN", command = BRunFunction)
        BRun.grid(column=2, row=4, ipadx=3, ipady=3, padx=4, pady=4)
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
        return tuple(retArgs)

    def frame_getter(frame_interval, frame = None, cl = None):
        cap = cv2.VideoCapture(video_path)
        
        def write_img(cl):
            ret, frame_read = cap.read()
            path = output_dir_vid+ "/frame-" + str(int(cap.get(1)-1)) + ".jpg"
            if not (os.path.isfile(path)):
                cv2.imwrite(path, frame_read)
                temp_data_list.append(Data(image_path = path, image = None, cluster_label = cl, emb = None, outlier = False))
        
        if frame is None:
            frame_count = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret==True:
                    if (frame_count % frame_interval) == 0:
                        path = output_dir_vid+ "/frame-" + str(frame_count) + ".jpg"
                        cv2.imwrite(path, frame)
                        data_list.append(Data(image_path = path, image = None, cluster_label = None, emb = None, outlier = False))
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
                    write_img(cl)
            for i in range(2):
                write_img(cl)
            if total_frames > frame + 2:
                cap.read()              #apla proxwraei g na mn ksanagrapsei tin idia eikona
                for i in range(2):
                    write_img(cl)
                # kai meta ksanatrexw ooolo apo tin arxi g ti 1i periptwsi
        # When everything is done, release the capture
        cap.release()

    def load_and_align_data(dl):
        minsize = 80 # minimum size of face
        threshold = [ 0.8, 0.9, 0.9 ]  # three steps's threshold
        factor = 0.709 # scale factor
        
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
      
        nrof_samples = len(dl)
        #nrof_samples_dif = nrof_samples # to thelw g ta multiple faces (wste n mn ksanampainei se epeksergasmena dl to main loop tou while)
        i=0
        while i < nrof_samples:
            img = misc.imread(os.path.expanduser(dl[i].image_path), mode='RGB')
            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            if bounding_boxes.size==0:
                print('image:'+dl[i].image_path+'\n has not a detectable face')
                try:
                    os.remove(dl[i].image_path)        #diagrafw ta frames xwris proswpa
                    del dl[i]
                except FileNotFoundError: 
                    print("warning: file not found (already deleted)")
                nrof_samples-=1
                i-=1
            elif bounding_boxes.shape[0]!=1:
                for j in range(nrof_faces):
                    det = np.squeeze(bounding_boxes[j,0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-args.margin/2, 0)
                    bb[1] = np.maximum(det[1]-args.margin/2, 0)
                    bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    aligned = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                    prewhitened = facenet.prewhiten(aligned)
                    dl.append(Data(image_path = dl[i].image_path, image = prewhitened, cluster_label = dl[i].cluster_label, emb = dl[i].emb, outlier = dl[i].outlier))
                print('image:'+dl[i].image_path+'\n has more than one face')
                del dl[i]           #diagrafw to arxiko datalist, evala alla pio panw
                nrof_samples-=1
                i-=1
                #edw t palia sto example
                
            else:
                det = np.squeeze(bounding_boxes[0,0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-args.margin/2, 0)
                bb[1] = np.maximum(det[1]-args.margin/2, 0)
                bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                aligned = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                prewhitened = facenet.prewhiten(aligned)
                '''r,g,b = cv2.split(prewhitened)
                img2 = cv2.merge([b,g,r])
                cv2.imshow('image'+str(i),img2)   
                cv2.waitKey(0)
                cv2.destroyAllWindows()'''
                dl[i].image = prewhitened
            i+=1
    
    def run_forward_pass(dl, model):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # Load the model
                facenet.load_model(model)
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                # Extract images from dl
                images = [x.image for x in dl]
                # Run forward pass to calculate embeddings
                feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                embs = sess.run(embeddings, feed_dict=feed_dict)
                # Integrate embeddings in dl
                for i,emb in enumerate(embs): dl[i].emb = emb
                
    def kMSilhouette(dl):
        range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] #tha ithela n dokimasw oso megalwnoun taclusters
                                                #an xeirotereuoun ta apotelesmata n stamataeiautomata
        j=0
        clusterer = [None] * len(range_n_clusters)
        cluster_labels = [None] * len(range_n_clusters)
        silhouette_avg = [None] * len(range_n_clusters)
        # Get the embeddings from dl
        emb = [x.emb for x in dl]
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
        
        # Integrate cluster_labels in dl
        for i,cll in enumerate(cluster_labels[best_cl]): dl[i].cluster_label = cll
        
        return sample_silhouette_values, best_cl, clusterer[best_cl].cluster_centers_
    
    def outliers(dl, max_clusters, sample_sil, N, video_path, output_dir_vid):
        nrof_images = len(dl)
        outl_sum = 0
        outl_cluster_sum = 0
        sum_of_images_in_cluster = 0
        return_val1 = -1
        return_val2 = []
        for i in range(max_clusters):
            for j in range(nrof_images):
                if (i==dl[j].cluster_label):
                    sum_of_images_in_cluster += 1
                    if sample_sil[j] <= N :
                        dl[j].outlier = True
                        #vazw poia frames einai
                        _,video = dl[j].image_path.split('video')
                        _,start_of_frame = video.split('-')
                        frame_nr,_ = start_of_frame.split('.')
                        frame_getter(None, int(frame_nr), i) #pare epipleon frames
                        outl_cluster_sum += 1
                        outl_sum += 1
            if (outl_cluster_sum/sum_of_images_in_cluster) >= 0.4:
                print ("REDO gia to cluster: " + str(i))
                return_val1 = i
            elif outl_cluster_sum==1:
                return_val2.append(i)
                #2-means
            print ('posa outliers exw sto cluster '+str(i)+':  ' + str(outl_cluster_sum) + ' vs')
            print ('sunolo eikonwn sauto to cluster ' + str(sum_of_images_in_cluster))
            print('\n')
            outl_cluster_sum = 0
            sum_of_images_in_cluster = 0
        return return_val1 ,return_val2
    
    def dispose_outliers(): #isws to xrisimopoiisw de kserw akoma (k to outlier_index_list tou outliers)
        print (outlier_index_list)
        outlier_index_list.sort()
        print (outlier_index_list)
        index_fix = 0
        for i in outlier_index_list:
            print(data_list[i - index_fix].cluster_label)
            del data_list[i - index_fix] #tha ta metakinw anti g del
            index_fix += 1
    
    def show_del_gui(output_dir, output_summary):
    
        def BShowFunction(output_dir):
            filedialog.askopenfile(mode="r", initialdir = output_dir)
            
        def BDelFunction(output_dir):
            if messagebox.askokcancel("Delete", "Are you sure you want to delete all results?"):
                for the_file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, the_file)
                    try:
                        if os.path.isdir(file_path): rmtree(file_path)
                    except Exception as e:
                        print(e)
            
        def myfunction(event):
            canvas.configure(scrollregion=canvas.bbox("all"),width=320,height=500)
        
        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                root.destroy()
                exit()

        root = Tk(className = ' Finding Distinct Faces')
        frame=Frame(root)                       #added
        
        Grid.rowconfigure(root, 0, weight=1)
        Grid.columnconfigure(root, 0, weight=1)
        frame.grid(row=0, column=0, sticky=N+S+E+W)
        grid=Frame(frame)
        grid.grid(sticky=N+S+E+W, column=0, row=7, columnspan=2)
        Grid.rowconfigure(frame, 7, weight=1)
        Grid.columnconfigure(frame, 0, weight=1)#------
        
        root.configure(background='#A3CEDC')
        
        canvas=Canvas(grid)
        frame=Frame(canvas)
        myscrollbar=Scrollbar(grid,orient="vertical",command=canvas.yview)
        canvas.configure(yscrollcommand=myscrollbar.set)
        
        myscrollbar.pack(side="right",fill="y")
        canvas.pack(side="left")
        canvas.create_window((0,0),window=frame,anchor='nw')
        frame.bind("<Configure>",myfunction)
        
        for i,outSum in enumerate(output_summary):
            print (i)
            print (outSum)
            c=i%2
            imgSum = os.listdir(outSum)
            path=outSum+'/'+imgSum[0]
            image = PIL.Image.open(path)
            if image.size>(500,300):
                image=image.resize((500,300))
            photo = ImageTk.PhotoImage(image)
            label = Label(frame, image=photo)
            label.image = photo
            label.grid(row=i-c, column=c, sticky=N+S+E+W)
            
        BShowFuncArg = partial(BShowFunction, output_dir)
        BShow = Button(root, text ="Show results", command = BShowFuncArg)
        BShow.grid(ipadx=3, ipady=3, padx=4, pady=4)
        
        BDelFuncArg = partial(BDelFunction, output_dir)
        BDel = Button(root, text ="Delete results", command = BDelFuncArg)
        BDel.grid(ipadx=2, ipady=2, padx=4, pady=4)        
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
    
    # "image_path image cluster_label emb") ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!!!!!
    data_list = []
    #sys.stdout = open(os.path.dirname(os.path.realpath(__file__))+'/output.txt', 'w+') #redirect output
    video_path = 'D:/mpoulas.mp4'
    outl_const = args.outlierConstant
    video_path, output_dir, model, frame_interval, outl_const = gui(video_path, args.output_dir, args.model, outl_const)
    output_dir_vid = os.path.expanduser(output_dir + '/video')
    if not os.path.exists(output_dir_vid):
        os.makedirs(output_dir_vid)
    #na to svisw sto telos telos to kanw g n mn trexei to frame_getter kathe fora (isws to enswmatwsw sto GUI)
    if False:
        frame_getter(frame_interval)
    else:
        dataset = facenet.get_dataset(output_dir)
        for path in dataset[0].image_paths: data_list.append(Data(image_path = path, image = None, cluster_label = None, emb = None, outlier = False))
    
    temp_data_list = data_list
    #fix outliers
    for i in range(3):
        load_and_align_data(temp_data_list)
        run_forward_pass(temp_data_list, model)
        data_list = data_list + temp_data_list
        sample_silhouette, best_cl, centers = kMSilhouette(data_list)
        temp_data_list = []
        redo, two_m_clusters = outliers(data_list, best_cl+2, sample_silhouette, outl_const, video_path, output_dir_vid)
        
        if redo == -1:  # if redo==False
            break
    
    if two_m_clusters:        #an exei stoixeia mesa
        load_and_align_data(temp_data_list)   #den esvina ta cllabels p kanei o framgetter kai den itn 1:1 me t dataset
        run_forward_pass(temp_data_list, model)
        data_list = data_list + temp_data_list
        nrof_images = len(data_list)
        while two_m_clusters:
            emb2 = []
            best_cl += 1
            two_m_cluster = two_m_clusters.pop()
            for j in range(nrof_images):
                if (two_m_cluster == data_list[j].cluster_label):
                    emb2.append(data_list[j].emb)
            clusterer = KMeans(n_clusters=2, random_state=10)
            cluster_labels_2 = clusterer.fit_predict(emb2)
            centers[two_m_cluster] = clusterer.cluster_centers_[0]
            centers=np.vstack((centers, clusterer.cluster_centers_[1]))
            for j in range(nrof_images):
                if (two_m_cluster == data_list[j].cluster_label):
                    first, cluster_labels_2 = cluster_labels_2[0], cluster_labels_2[1:]   #pop
                    if first == 1:
                        data_list[j].cluster_label = best_cl + 1    #allazw mono gia to 2o label gt to 1o de me noiazei n meinei idio
                        
    
    nrof_images = len(data_list)
    outl_dir = os.path.expanduser(output_dir + '/outliers')
    if not os.path.exists(outl_dir):
        os.makedirs(outl_dir)
    i=0
    while i < nrof_images:
        if data_list[i].cluster_label == redo or data_list[i].outlier:
            r,g,b = cv2.split(data_list[i].image)
            img2 = cv2.merge([b*255,g*255,r*255])
            #path manipulation for imwrite
            outImWr=outl_dir+'/'+str(data_list[i].cluster_label)+' (cropped)'+os.path.basename(data_list[i].image_path)
            cv2.imwrite(outImWr,img2)
            copy(data_list[i].image_path,outl_dir)
            del data_list[i]
            nrof_images-=1
            i-=1
        i+=1
    
    nrof_images = len(data_list)
    output_dir_cluster = [None] * (best_cl+2)
    output_summary = [None] * (best_cl+2)
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
        r,g,b = cv2.split(data_list[j].image)
        img2 = cv2.merge([b*255,g*255,r*255])
        #path manipulation for imwrite
        outImWr = output_dir_cluster[data_list[j].cluster_label]+' (cropped)'+'/'+str(j)+' '+os.path.basename(data_list[j].image_path)
        cv2.imwrite(outImWr,img2)
        copy(data_list[j].image_path, output_dir_cluster[data_list[j].cluster_label])
        
    for i in range(best_cl+2):
        emb2 = []
        dataset3 = []
        if not redo == i:
            for j in range(nrof_images):
                if (i==data_list[j].cluster_label):
                    emb2.append(data_list[j].emb)
                    dataset3.append(j)
            Ccenter=[centers[i],centers[i]]
            closest, _ = pairwise_distances_argmin_min(Ccenter, emb2)
            copy(data_list[dataset3[closest[0]]].image_path,output_summary[i])
            
            r,g,b = cv2.split(data_list[dataset3[closest[0]]].image)
            img2 = cv2.merge([b*255,g*255,r*255])
            #path manipulation for imwrite
            outImWr=output_summary[i]+'/(cropped)'+os.path.basename(data_list[dataset3[closest[0]]].image_path)
            cv2.imwrite(outImWr,img2)
            
    if not redo == -1:  # if redo==True
        os.rmdir(output_dir_cluster[redo])  # delete these because it will be empty
        os.rmdir(output_dir_cluster[redo]+' (cropped)')
        os.rmdir(output_summary[redo])
        del output_summary[redo]
    
    show_del_gui(output_dir, output_summary)
    



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

#de xreiazonte ta plots
#interface, database me polla video

#gia to detect_face:davidsandberg:
#There are five landmarks detected by MTCNN and these are left eye, right eye, nose, left mouth corner, and right mouth corner.
# It would not be straight forward to detect a larger number of landmarks.

#gia:   sample_silhouette_values[j] = silhouette_samples(emb, cluster_labels[j])
#The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.

#16/9 na mn ksexasw na svinei ta epipleon frames p pairnw
