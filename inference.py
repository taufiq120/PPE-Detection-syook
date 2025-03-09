import cv2
import os
from ultralytics import YOLO

# Load trained models
person_model = YOLO("weights/person_detection.pt")
ppe_model = YOLO("weights/ppe_detection.pt")

# Directories
input_folder = "datasets/images"
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# Define PPE class names
PPE_CLASSES = ["hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]

def detect_ppe(image_path):
    image = cv2.imread(image_path)

    # Step 1: Detect persons
    person_results = person_model(image)
    
    if len(person_results[0].boxes) == 0:
        print(f"⚠️ No person detected in {image_path}. Skipping...")
        return  # Skip if no person detected

    for result in person_results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Convert to NumPy array
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            # Draw bounding box around person
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Crop the detected person
            cropped_image = image[y1:y2, x1:x2].copy()

            # Step 2: Detect PPE on cropped person
            ppe_results = ppe_model(cropped_image)

            if len(ppe_results[0].boxes) == 0:
                print(f"⚠️ No PPE detected on person {i} in {image_path}.")
                continue  # Skip if no PPE detected

            for ppe_result in ppe_results:
                ppe_boxes = ppe_result.boxes.xyxy.cpu().numpy()
                ppe_classes = ppe_result.boxes.cls.cpu().numpy()

                for j, ppe_box in enumerate(ppe_boxes):
                    px1, py1, px2, py2 = map(int, ppe_box)

                    # Adjust PPE coordinates relative to full image
                    px1 += x1
                    py1 += y1
                    px2 += x1
                    py2 += y1

                    class_id = int(ppe_classes[j])
                    label = PPE_CLASSES[class_id] if class_id < len(PPE_CLASSES) else "PPE"

                    # Draw PPE bounding box
                    cv2.rectangle(image, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.putText(image, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save the output image
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"✅ Processed: {image_path} -> {output_path}")

# Process all images in the input directory
for img_file in os.listdir(input_folder):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):
        detect_ppe(os.path.join(input_folder, img_file))

print("✅ Inference Complete! Results saved in 'results/'")
