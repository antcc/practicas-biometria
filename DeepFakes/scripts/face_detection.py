#!/usr/bin/env python3


import os
from IPython import display
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN

# Shape of face to save
image_size = (160, 160)

# Reading path
path = ""
# Writting path
new_path = ""


# Face detection function using MTCNN
def face_detection(img):
      img = Image.fromarray(np.uint8(img))
      results = detector.detect_faces(np.asarray(img))

      # Show images with no face detected.
      if results == []:
        plt.imshow(img)
        plt.show()
        return False

      x1, y1, width, height = results[0]['box']
      x1, y1 = abs(x1), abs(y1)
      x2, y2 = x1 + width, y1 + height

      image = np.asarray(img)
      face = image[y1:y2, x1:x2]
      face = Image.fromarray(face)
      face = face.resize(image_size)
      return face



for subdir, dirs, files in os.walk(path):
    for file in files:
        path = os.path.join(subdir, file)
        img = Image.open(path)
        face = face_detection(img)
        if face:
            # Save reading path but changing root
            new_path = new_path + "/".join(path.split("/")[1:])
            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path))
            face.save(new_path)
