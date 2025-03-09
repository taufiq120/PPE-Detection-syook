import cv2
from ultralytics import YOLO

# Load trained PPE model
ppe_model = YOLO("weights/ppe_detection.pt")

# Load a cropped person image
image_path = "datasets/cropped_persons/train/cropped_0.jpg"  # Replace with an actual cropped image
image = cv2.imread(image_path)

# Run PPE detection
results = ppe_model(image)

# Draw bounding boxes
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = PPE_CLASSES[int(classes[i])]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Save and show the output
cv2.imwrite("results/test_ppe.jpg", image)
print("✅ Test PPE detection complete. Check 'results/test_ppe.jpg'")
