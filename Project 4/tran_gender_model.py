import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt 

# Veri seti dizini
train_dir = "dataset"

# Veri artırma (Data Augmentation)
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,  # Hata düzeltilmiş
    validation_split=0.2,  # Doğrulama seti için %20 ayır
)

# Eğitim verilerini yükleme
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=5,
    class_mode="binary",
    subset="training"
)

# Doğrulama verilerini yükleme
val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=5,
    class_mode="binary",
    subset="validation"
)

# CNN Modeli oluşturma
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.5),  # Overfitting'i azaltmak için
    tf.keras.layers.Dense(1, activation="sigmoid")  # İkili sınıflandırma için sigmoid aktivasyon
])

# Modeli derleme
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Modeli eğitme
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Modeli kaydetme
model.save("gender_model.h5")

# Eğitim grafiğini çizdirme
plt.plot(history.history["accuracy"], label="Eğitim Doğruluğu")
plt.plot(history.history["val_accuracy"], label="Doğrulama Doğruluğu")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
