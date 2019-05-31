from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import Facenet.facenet as facenet
import Facenet.detect_face as detect_face
import os
import time
import pickle
import sys

sess = None
pnet= None
rnet= None
onet= None
embedding_size= None
images_placeholder= None
phase_train_placeholder= None
embeddings= None
model= None
HumanNames= None

def Init(): 
 global sess,pnet,rnet,onet,embedding_size,images_placeholder,embeddings,model,HumanNames,phase_train_placeholder
 modeldir = 'Facenet/model/20180408-102900.pb'
 classifier_filename = 'Facenet/class/classifier.pkl'
 npy='Facenet/npy'
 train_img="align"

 with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        HumanNames = os.listdir(train_img)
        HumanNames.sort()
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        # video_capture = cv2.VideoCapture("akshay_mov.mp4") 


def predict(image1):
        global sess,pnet,rnet,onet,embedding_size,imagesS_placeholder,embeddings,model,HumanNames,phase_train_placeholder
        minsize = 20  # minimum size of face
        threshold = [0.8, 0.9, 0.9]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        batch_size = 1000
        image_size = 182
        input_image_size = 160
      

        if (True):
            find_results = []
            
            emb_array = np.zeros((1, embedding_size))
            # frame = frame[:, :, 0:3]
            # bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            # nrof_faces = bounding_boxes.shape[0] # số detec
            # print('Face Detected: %d' % nrof_faces)
            nrof_faces =1
            if nrof_faces > 0:
                begin=time.time()
                cropped = []
                scaled = []
                scaled_reshape = []
                i=0
                if(True):
                    cropped.append(image1)
                    cropped[i] = facenet.flip(cropped[i], False)
                    scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                    scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                           interpolation=cv2.INTER_CUBIC)
                    scaled[i] = facenet.prewhiten(scaled[i])
                    scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                    predictions = model.predict_proba(emb_array )
                    best_class_indices = np.argmax(predictions, axis=1)
                    end=time.time()
                    print("predict :" +str(end-begin))
                    if(predictions[0][best_class_indices[0]]>=0.5):
                        return HumanNames[best_class_indices[0]]
                    else:
                        return "Nguoi la"
