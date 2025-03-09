from ultralytics import YOLO
from argparse import ArgumentParser
import os

def get_latest_model():
    runs_dir = "runs/detect"
    if not os.path.exists(runs_dir):
        return None

    subfolders = [f for f in os.listdir(runs_dir) if f.startswith("person_detection")]

    if not subfolders:
        return None

    # Extract only the numeric part safely
    def extract_number(folder_name):
        num_part = folder_name.replace("person_detection", "").strip()
        return int(num_part) if num_part.isdigit() else 0  # Default to 0 if empty

    # Sort folders by extracted number
    latest_folder = sorted(subfolders, key=extract_number)[-1]
    
    return os.path.join(runs_dir, latest_folder, "weights", "best.pt")

def main():
    parser = ArgumentParser(description="Train YOLOv8 model for person detection.")
    parser.add_argument("--data", default="person_dataset.yaml", help="Path to dataset YAML file.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    args = parser.parse_args()

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")  # Use a pre-trained YOLOv8 model

    # Train the model
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="person_detection"
    )

    # Ensure weights directory exists
    os.makedirs("weights", exist_ok=True)

    # Get the latest trained model path
    best_model = get_latest_model()

    if best_model and os.path.exists(best_model):
        print(f"✅ Model training complete! Saved as {best_model}")
        os.rename(best_model, "weights/person_detection.pt")
    else:
        print("❌ Error: Trained model not found!")

if __name__ == "__main__":
    main()
