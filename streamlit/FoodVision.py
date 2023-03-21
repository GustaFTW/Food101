import streamlit as st
from PIL import Image

hamburguer = Image.open('C:\\Users\\Gusta\\Documents\\CodeStuff\\Machine_Learning\\Practice\\project\\ImageDetection\\streamlit\\pics\\230446.jpg')
ice_cream = Image.open('C:\\Users\\Gusta\\Documents\\CodeStuff\\Machine_Learning\\Practice\\project\\ImageDetection\\streamlit\\pics\\321321.jpg')
pizza = Image.open('C:\\Users\\Gusta\\Documents\\CodeStuff\\Machine_Learning\\Practice\\project\\ImageDetection\\streamlit\\pics\\pizza.jpg')
base_model_structure = Image.open("C:\\Users\\Gusta\\Documents\\CodeStuff\\Machine_Learning\\Practice\\project\\ImageDetection\\streamlit\\pics\\base_model_structure.png")
model_structure = Image.open("C:\\Users\\Gusta\\Documents\\CodeStuff\\Machine_Learning\\Practice\\project\\ImageDetection\\streamlit\\pics\\model_structure.png")


st.title("Food Recognition")
st.text("""This is a project that can identify 101 types of different food, 
it was made using Tensorflow Food101 dataset, all the code can be found on
github.com/gustaftw/ 
with the EfficientNetB0 base model,
and the custom top layers for feacture extraction for 5 epochs, 
then, finally fine tuned (base model layers unfrozen) for more 3 epochs
With this model (and only on 8 epochs) 
it achieved a 75% accuracy on 101 classes! 
For the amount of training time these are not so bad results.""")
st.header("_Food101 Dataset_")
st.text("""As mentioned above, this project used the Food101 dataset 
from Tensorflowdatasets (https://www.tensorflow.org/datasets/api_docs/python/tfds), 
after the data pipeline (the code can be found in the github) the model was fitted to the data""")
st.text("""The first thing needed to be done in the data was
split it into train and test set, which was easy 
using the tensorflow.dataset library. After that, it was used
tfds.prefetch (so that our code uses as many threads in the cpu
to preload the data into the model) and tfds.batch (which splits the data
into batches of size 32). Here's three samples of our dataset: """)

col1, col2, col3 = st.columns(3)

with col1:
   st.header("Hamburguer")
   st.image(hamburguer)

with col2:
   st.header("Ice cream")
   st.image(ice_cream)
 
with col3:
   st.header("Pizza")
   st.image(pizza)

st.title("The model")
st.header("_Structure of our model_")
st.text("""As previously instantiated, the model is a feacture extraction/fine tuned 
model with EfficientNetB0 as the base. Which led to a high performance model with 
only a few epochs training, the base layers were to frozen on the first 5 epochs,
(for feature extraction), and unfrozen for the further 3 epochs.
The model was saved on launch so that it doesn't have
to fit it on deploy, saving time in production.
The code that generated the model:
""")
st.code("""# Creating the model
inputs = tf.keras.layers.Input(INPUT_SHAPE, name="input_layer")
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(101, activation="softmax", name="output_layer")(x)
model_foodvision = tf.keras.Model(inputs, outputs)""")
st.text("""Notice that it was set a GlobalAveragePooling2D layer after the base model 
to reduce the amount of parameters that it had. For the ouput layer it was used 
a fully connected (dense) layer with 101 (quantity of labels) neurons and a 
activation of softmax. Softmax is a good choice for the output of a neural 
network because it can normalize the output into a probability distribution 
of possible outcomes. Heres the structure of the model, and base model (EfficientNetB0)""")
col1, col2 = st.columns(2)

with col1:
   st.header("Feature estructure model")
   st.image(model_structure)

with col2:
   st.header("EfficientNetB0")
   st.image(base_model_structure, caption="Only the last layers because the models is to big to be shown entirely")

st.title("Evaluating the model's predictions")
st.text("""As mentioned early the model achieved a accuracy of 75%, which is a very
good number for not that many epochs. But there are some other values that deserves 
attention, they are precision, recall and f1-score metrics. What do they mean, briefely?
Precision calculates how many times our model predicted a label for positive 
(1 for pizza for example) and it got correct, recall takes all the predictions 
of positive and check to see how many are actually correct. What can be taken
from that is, if you don't want your model to have false positives
(predicting 1 when actually should be 0) you should try to increase
precision, and if you don't want your model to have false negatives
(predicting 0 when it should be 1)  you should try to increase your recall 
(please note that if you increase recall, you get a lower precision
and vice-versa, for more look into precision recall tradeoff)""")
st.title("Conclusion")
st.text("""The model achieved good results with limited training time for the number of samples
and labels (101 is quite a lot!), so it can easily be launched it into production 
for many applications where it's needed to classify/identify any type of food.""")
