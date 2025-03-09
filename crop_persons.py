import cv2
import os
from ultralytics import YOLO

# Load trained person detection model
person_model = YOLO("weights/person_detection.pt")

# Input and output directories
input_folder = "datasets/images"
output_folder = "datasets/cropped_persons"

os.makedirs(output_folder, exist_ok=True)

def crop_persons(image_path, output_folder):
    image = cv2.imread(image_path)
    results = person_model(image)

    for i, r in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, r)
        person_crop = image[y1:y2, x1:x2]
        cv2.imwrite(f"{output_folder}/cropped_{i}.jpg", person_crop)

# Process all images
for img_file in os.listdir(input_folder):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):
        crop_persons(os.path.join(input_folder, img_file), output_folder)

print("✅ Cropped person images saved in datasets/cropped_persons")
