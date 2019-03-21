import re, gzip, pickle, time
from collections import Counter
from multiprocessing import Queue, Lock
import threading
from nltk.corpus import stopwords
from nltk import sent_tokenize
# from gensim.models.word2vec import Text8Corpus
from nltk.stem import WordNetLemmatizer
from nltk.parse.corenlp import CoreNLPParser
import codecs
import cython
import numpy as np
import pyximport
from scipy import sparse
import logging
from glove_train import *
pyximport.install(setup_args={"include_dirs": np.get_include()})
from glove_cython_D import train_glove

try:
    # Python 2 compat
    import cPickle as pickle
except ImportError:
    import pickle


# Set parameters
windowSize = 10
embedSize = 2
xmax = 2
alpha = 0.75
batchSize = 50
learningRate = 0.05
numEpochs = 10
vectorDimension = 200