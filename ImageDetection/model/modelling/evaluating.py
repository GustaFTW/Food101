import tensorflow as tf
import sys
import sklearn.metrics as skm
sys.path.insert(0, "C:\\Users\\Gusta\\Documents\\CodeStuff\\Machine_Learning\\Practice\\project\\ImageDetection\\model\\data\\")
sys.path.insert(0, "C:\\Users\\Gusta\\Documents\\CodeStuff\\Machine_Learning\\Practice\\project\\ImageDetection\\helper_functions\\")
from helper_functions import make_confusion_matrix

import data_pipeline

model_foodvision = tf.keras.models.load_model("C:\\Users\\Gusta\\Documents\\CodeStuff\\Machine_Learning\\Practice\\project\\ImageDetection\\model\\modelling\\model_save\\")

# Load in our data and our model
test_data = data_pipeline.test_data

# Evaluate the model
model_foodvision.evaluate(test_data)
