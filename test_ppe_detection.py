import cv2
import os
from ultralytics import YOLO

# Load trained PPE detection model
ppe_model = YOLO("weights/ppe_detection.pt")

# Input and output folders
input_folder = "datasets/cropped_persons/train/images"  # Adjusted path
output_folder = "static/ppe_test"
os.makedirs(output_folder, exist_ok=True)

# Define PPE class names
PPE_CLASSES = ["hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]

def detect_ppe(image_path):
    image = cv2.imread(image_path)

    # Run inference on PPE model
    results = ppe_model(image)

    if len(results[0].boxes) == 0:
        print(f"⚠️ No PPE detected in {image_path}. Skipping...")
        return  # Skip if no PPE detected

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(classes[i])
            label = PPE_CLASSES[class_id] if class_id < len(PPE_CLASSES) else "PPE"

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save the output image
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"✅ PPE detected in {image_path}. Saved to {output_path}")

# Process all images in the cropped directory
for img_file in os.listdir(input_folder):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):
        detect_ppe(os.path.join(input_folder, img_file))

print("✅ PPE detection test complete! Results saved in 'static/ppe_test/'")
