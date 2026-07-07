import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

IMAGE_SIZE = (224,224)
BATCH_SIZE = 32

train_path = "dataset/asl_alphabet_train/asl_alphabet_train"

if not os.path.exists(train_path):
    print("Dataset folder not found!")
    exit()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)

validation_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42
)

print(train_generator.class_indices)