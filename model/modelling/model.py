import sys
import tensorflow as tf
sys.path.insert(0, "C:\\Users\\Gusta\\Documents\\CodeStuff\\Machine_Learning\\Practice\\project\\ImageDetection\\model\\data\\")
import data_pipeline
INPUT_SHAPE = (224, 224, 3)


train_data, test_data = data_pipeline.imageDataset

# Now we have everything we need to try and create our model
# We are going to create a feature extraction model based on EfficientNetB0

base_model = tf.keras.applications.EfficientNetB0(include_top=False) # because we are not using the top layers, we're creating our owns
base_model.trainable = False # due to only feature extraction, we don't their previously learned patterns to change

# Creating the model
inputs = tf.keras.layers.Input(INPUT_SHAPE, name="Input_layer")
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(101, activation="softmax", name="Output_layer")(x)
model_foodvision = tf.keras.Model(inputs, outputs, name="Food_vision_model")

# Compile our  model
model_foodvision.compile(loss="sparse_categorical_crossentropy",
                         optimizer="Adam",
                         metrics="accuracy")

# Check the summary of our model
# print("\n" * 5)
print(model_foodvision.summary())

# Instantiate a callback before fitting the model
path_dir = "C:\\Users\\Gusta\\Documents\\CodeStuff\\Machine_Learning\\Practice\\project\\ImageDetection\\model\\modelling\checkpoints\\checkpoints\\"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(path_dir,
                                                         monitor="accuracy",
                                                         save_best_only = True,
                                                         save_weights_only = True)


# Now fit the model
print("\n" * 5)
history_feature_extraction = model_foodvision.fit(train_data,
                                                  epochs=5,
                                                  steps_per_epoch=len(train_data),
                                                  validation_data=test_data,
                                                  validation_steps=int(len(test_data) * 0.01),
                                                  callbacks=checkpoint_callback)

# Reset the base model layers back to trainable, so we can fine-tuned it our data
base_model.trainable = True
# Refreeze the layers except for the top 5 layers
for layer in base_model.layers[:-5]:
    layer.trainable = False

# Recompile the model lowering the learning rate
model_foodvision.compile(loss="sparse_categorical_crossentropy",
                         optimizer=tf.keras.optimizers.Adam(lr=10e-5),
                         metrics=["accuracy"])

# Fine-tune it
history_fine_tune = model_foodvision.fit(train_data,
                                         epochs=3,
                                         steps_per_epoch=len(train_data),
                                         validation_data=test_data,
                                         validation_steps=int(len(test_data) * 0.01),
                                         callbacks=checkpoint_callback)


tf.keras.utils.plot_model(base_model,
                          to_file="base_model_structure.png")

tf.keras.utils.plot_model(model_foodvision,
                          to_file="model_structure.png")

model_foodvision.save("C:\\Users\\Gusta\\Documents\\CodeStuff\\Machine_Learning\\Practice\\project\\ImageDetection\\model\\modelling\\model_save\\")
