import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 🔧 FIXED PARAMETERS (FAST)
EPOCHS = 5
INIT_LR = 1e-3
BS = 16
default_image_size = (128, 128)
directory_root = 'PlantVillage'
width = 128
height = 128
depth = 3

# 🔧 FUNCTION
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, default_image_size)
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error: {e}")
        return None

# 📥 LOAD DATA
image_list, label_list = [], []

print("[INFO] Loading images...")

for disease_folder in listdir(directory_root):
    if disease_folder == ".DS_Store":
        continue

    print(f"[INFO] Processing {disease_folder} ...")
    disease_path = f"{directory_root}/{disease_folder}"
    
    for image_name in listdir(disease_path)[:500]: # limit for speed
        image_path = f"{disease_path}/{image_name}"
        
        if image_path.endswith(".jpg") or image_path.endswith(".JPG"):
            image_list.append(convert_image_to_array(image_path))
            label_list.append(disease_folder)

print("[INFO] Image loading completed")

# 🏷 LABEL ENCODING
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer, open('label_transform.pkl', 'wb'))

n_classes = len(label_binarizer.classes_)

# 🔧 NORMALIZATION FIX
np_image_list = np.array(image_list, dtype=np.float16) / 255.0

# 🔀 TRAIN TEST SPLIT
print("[INFO] Splitting data...")
x_train, x_test, y_train, y_test = train_test_split(
    np_image_list, image_labels, test_size=0.2, random_state=42
)

# 🔄 AUGMENTATION
aug = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# 🧠 MODEL
model = Sequential()

inputShape = (height, width, depth)
chanDim = -1

model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(n_classes))
model.add(Activation("softmax"))

# 🔧 OPTIMIZER FIX
opt = Adam(learning_rate=INIT_LR)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 🚀 TRAIN
print("[INFO] Training network...")

history = model.fit(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    verbose=1
)

# 📊 PLOT
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title("Accuracy")
plt.show(block=False)
plt.pause(3)
plt.close()

# 📊 EVALUATE
print("[INFO] Evaluating model...")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100:.2f}%")

# 💾 SAVE MODEL
model.save("plant_model.h5")
print("[INFO] Model saved as plant_model.h5")
# 🌿 REMEDY SYSTEM
remedies = {
    "Apple___Black_rot": "Remove infected leaves and apply fungicide.",
    "Tomato___Late_blight": "Use copper-based fungicide and avoid overhead watering.",
    "Potato___Early_blight": "Crop rotation and fungicide recommended.",
    "Apple___healthy": "No disease detected. Maintain care."
}

# 🔍 TEST PREDICTION
idx = np.argmax(model.predict(x_test[:1]))
predicted_class = label_binarizer.classes_[idx]

print("\n🌿 Prediction Result")
print("Disease:", predicted_class)
print("Remedy:", remedies.get(predicted_class, "Consult expert"))
