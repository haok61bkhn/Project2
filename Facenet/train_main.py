from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from Facenet.classifier import training
from Facenet.preprocess import *

def train():
  datadir = 'align/'
  modeldir = "Facenet/model/20180408-102900.pb"
  classifier_filename = 'Facenet/class/classifier.pkl'
  print ("Training Start")
  obj=training(datadir,modeldir,classifier_filename)
  get_file=obj.main_train() 
  print('Saved classifier model to file "%s"' % get_file)
