import tensorflow as tf
import numpy as np
import cv2
import pickle

# load model + labels
model = tf.keras.models.load_model("plant_model.h5")
label_binarizer = pickle.load(open("label_transform.pkl", 'rb'))

# remedies
remedies = {
    "Tomato_Bacterial_spot": "Use disease-free seeds, avoid overhead watering, apply copper-based bactericide.",
    "Tomato_Early_blight": "Remove infected leaves, apply fungicide, ensure crop rotation.",
    "Tomato_Late_blight": "Use copper fungicide, avoid wet leaves, improve airflow.",
    "Tomato_Leaf_Mold": "Reduce humidity, ensure ventilation, apply fungicide.",
    "Tomato_Septoria_leaf_spot": "Remove infected foliage, avoid splashing water, apply fungicide.",
    "Tomato__Target_Spot": "Use resistant varieties, apply fungicide, maintain spacing.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use miticide, increase humidity, wash leaves.",
    "Tomato__Tomato_mosaic_virus": "Remove infected plants, sanitize tools.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies, remove infected plants.",
    "Tomato_healthy": "No disease detected. Maintain regular care.",
    # add others if needed
}

# input image
img_path = input("Enter image path: ")

img = cv2.imread(img_path)
img = cv2.resize(img, (128,128))
img = img / 255.0
img = np.reshape(img, (1,128,128,3))

# predict
pred = model.predict(img)
idx = np.argmax(pred)
disease = label_binarizer.classes_[idx]

print("\nDisease:", disease)
print("Remedy:", remedies.get(disease, "Consult expert"))
