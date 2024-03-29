import cv2 as cv
import numpy as np
import os
from mtcnn.mtcnn import MTCNN
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FACELOADING:
    def __init__(self, directory, min_face_size=20, max_face_size=None):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()
        self.min_face_size = min_face_size
        self.max_face_size = max_face_size


    def extract_face(self, filename, output_dir='output_2'):
        os.makedirs(output_dir, exist_ok=True)
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Resize image if needed
        if max(img.shape[:2]) > 1000:
            img = cv.resize(img, (1000, 1000))

        height, width, channels = img.shape
        # Check if the image is too small to process
        if min(height, width) < self.min_face_size:
            print(f"Skipping {filename} due to small image size")
            return

        start_time = time.time()
        faces = self.detector.detect_faces(img)
        end_time = time.time()
        print(f"{end_time - start_time:.2f} seconds")

        if faces:
            # Sort faces based on x-coordinate
            faces.sort(key=lambda x: x['box'][0])
            # Extract the face with the lowest x-value
            x, y, w, h = faces[0]['box']
            x, y = abs(x), abs(y)
            answer = self.yolo_normalize(x, y, w, h, width, height)
            # Save the new image with bounding box
            base_filename = os.path.splitext(os.path.basename(filename))[0]
            txt_filename = os.path.join(output_dir, base_filename + '.txt')
            # Save x, y, w, h coordinates to text file
            print(answer)
            with open(txt_filename, 'w') as f:
                f.write(f'x: {answer[0]}, y: {answer[1]}, w: {answer[2]}, h: {answer[3]}')

    def load_images(self):
        image_dir = os.path.join(os.getcwd(), self.directory)  # Get the full path of the directory
        for im_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, im_name)
            self.extract_face(image_path)

    def yolo_normalize(self, x, y, w, h, img_width, img_height):
        # Convert inputs to a numpy array
        data = np.array([x, y, w, h])

        # Normalize x and w relative to image width
        x_normalized = x / img_width
        w_normalized = w / img_width

        # Normalize y and h relative to image height
        y_normalized = y / img_height
        h_normalized = h / img_height

        return [x_normalized, y_normalized, w_normalized, h_normalized]


faceloading = FACELOADING('./valid/Raafay')
faceloading.load_images()
