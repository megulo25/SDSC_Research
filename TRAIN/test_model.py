from keras.models import model_from_json
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np

validation_dir = os.path.join(os.getcwd(), 'data', 'nabirds_9', 'test')

# Import model
with open('model_architecture_1.json', 'r') as f:
    model = model_from_json(f.read())

model.load_weights('model_weights_1.h5')


# Validation Generator
test_datagen = ImageDataGenerator(
    rescale=1./255
)

validation_generator = test_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(299,299),
    batch_size=1,
    class_mode='categorical'
)

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = list(validation_generator.class_indices.keys())
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))