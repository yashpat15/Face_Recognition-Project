import cv2
import os
import numpy as np
from PIL import Image

def create_training_data(directory):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    face_samples = []
    ids = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]

                pil_image = Image.open(path).convert('L') 
                image_array = np.array(pil_image, 'uint8')
                faces = face_cascade.detectMultiScale(image_array)

                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    face_samples.append(roi)
                    ids.append(id_)

    recognizer.train(face_samples, np.array(ids))
    recognizer.save('models/face-trainer.yml')

if __name__ == "__main__":
    create_training_data('data/images')
