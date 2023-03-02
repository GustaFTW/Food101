import tensorflow_datasets as tfds
import tensorflow as tf
import sys
sys.path.insert(0, "C:\\Users\\Gusta\\Documents\\CodeStuff\\Machine_Learning\\Practice\\project\\ImageDetection\\helper_functions")
from helper_functions import preprocess_img

# Now, we download our data into the separated folders
(train_data, test_data), ds_info = tfds.load(name="food101",
                                             split=["train", "validation"],
                                             shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True)

# Defining the class names
class_names = ds_info.features["label"].names
# print(len(class_names))

# Now batching and preparing our data
train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Fit the test data
test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)
# Now to see what they look like
# print(f"Our test data contains {len(test_data)} batches.\nOur train data contains {len(train_data)} batches.\n")
# print("-" * 20)
imageDataset = train_data, test_data