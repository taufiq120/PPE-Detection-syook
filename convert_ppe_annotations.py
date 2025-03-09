import os
import xml.etree.ElementTree as ET
import cv2
from ultralytics import YOLO

# Load person detection model
person_model = YOLO("weights/person_detection.pt")

# Input folders
image_folder = "datasets/images"
label_folder = "datasets/labels"

# Output folders for cropped images and new labels
cropped_image_folder = "datasets/cropped_persons/train"
cropped_label_folder = "datasets/cropped_persons/train/labels"
os.makedirs(cropped_image_folder, exist_ok=True)
os.makedirs(cropped_label_folder, exist_ok=True)

# Define PPE class names
PPE_CLASSES = ["hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]

def convert_ppe_annotations(image_name, full_img_width, full_img_height, person_x1, person_y1, person_x2, person_y2, objects):
    """ Convert full-image PPE annotations to cropped-person coordinates """
    new_annotations = []
    
    for obj in objects:
        cls_name = obj.find("name").text
        if cls_name not in PPE_CLASSES:
            continue  # Skip non-PPE objects
        
        cls_id = PPE_CLASSES.index(cls_name)
        bbox = obj.find("bndbox")

        try:
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
        except AttributeError:
            print(f"⚠️ Skipping object in {image_name} due to missing bounding box.")
            continue  # Skip if any bounding box data is missing

        # Check if PPE is inside the cropped person bounding box
        if xmax < person_x1 or xmin > person_x2 or ymax < person_y1 or ymin > person_y2:
            continue

        # Adjust coordinates relative to cropped person
        new_xmin = max(xmin - person_x1, 0)
        new_ymin = max(ymin - person_y1, 0)
        new_xmax = min(xmax - person_x1, person_x2 - person_x1)
        new_ymax = min(ymax - person_y1, person_y2 - person_y1)

        # Normalize coordinates for YOLO format
        x_center = (new_xmin + new_xmax) / 2 / (person_x2 - person_x1)
        y_center = (new_ymin + new_ymax) / 2 / (person_y2 - person_y1)
        width = (new_xmax - new_xmin) / (person_x2 - person_x1)
        height = (new_ymax - new_ymin) / (person_y2 - person_y1)

        new_annotations.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return new_annotations

def process_images():
    """ Process each image and extract cropped persons with adjusted PPE annotations """
    for image_name in os.listdir(image_folder):
        if not image_name.endswith(".jpg") and not image_name.endswith(".png"):
            continue

        image_path = os.path.join(image_folder, image_name)
        label_path = os.path.join(label_folder, image_name.replace(".jpg", ".xml"))

        if not os.path.exists(label_path):
            print(f"⚠️ Warning: No XML annotation found for {image_name}. Skipping.")
            continue

        # Load image and annotation
        image = cv2.imread(image_path)
        tree = ET.parse(label_path)
        root = tree.getroot()

        full_img_width = int(root.find("size/width").text)
        full_img_height = int(root.find("size/height").text)

        # Detect persons in the image
        results = person_model(image)

        if len(results[0].boxes.xyxy) == 0:
            print(f"⚠️ No person detected in {image_name}, skipping.")
            continue

        for i, person_box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
            person_x1, person_y1, person_x2, person_y2 = map(int, person_box)

            # Crop person from the image
            cropped_image = image[person_y1:person_y2, person_x1:person_x2]
            cropped_image_path = os.path.join(cropped_image_folder, f"{image_name.replace('.jpg', '')}_person_{i}.jpg")
            cv2.imwrite(cropped_image_path, cropped_image)

            # Convert PPE annotations to cropped person coordinates
            objects = root.findall("object")
            new_annotations = convert_ppe_annotations(image_name, full_img_width, full_img_height, person_x1, person_y1, person_x2, person_y2, objects)

            # Save the new annotations
            if new_annotations:
                cropped_label_path = os.path.join(cropped_label_folder, f"{image_name.replace('.jpg', '')}_person_{i}.txt")
                with open(cropped_label_path, "w") as f:
                    f.write("\n".join(new_annotations))
                print(f"✅ Saved {cropped_label_path} with {len(new_annotations)} annotations.")
            else:
                print(f"⚠️ No PPE annotations found for {cropped_image_path}. Skipping label creation.")

# Run the process
process_images()
print("✅ All persons cropped and PPE annotations converted.")
