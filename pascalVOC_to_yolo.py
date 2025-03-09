import os
import xml.etree.ElementTree as ET
from argparse import ArgumentParser

def convert_annotation(xml_file, output_dir, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    yolo_annotations = []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Convert to YOLO format (normalized)
        x_center = (xmin + xmax) / (2 * width)
        y_center = (ymin + ymax) / (2 * height)
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        yolo_annotations.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    # Save YOLO annotation file
    output_file = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".txt")
    with open(output_file, "w") as f:
        f.write("\n".join(yolo_annotations))

def main():
    parser = ArgumentParser(description="Convert PascalVOC annotations to YOLOv8 format.")
    parser.add_argument("input_dir", help="Path to the directory containing PascalVOC annotations.")
    parser.add_argument("output_dir", help="Path to save YOLOv8 annotations.")
    args = parser.parse_args()

    # Load class names
    classes_file = os.path.join(args.input_dir, "classes.txt")
    if not os.path.exists(classes_file):
        print(f"❌ Error: {classes_file} not found!")
        return

    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all XML files in the "labels" folder
    labels_folder = os.path.join(args.input_dir, "labels")
    if not os.path.exists(labels_folder):
        print(f"❌ Error: Labels folder '{labels_folder}' not found!")
        return

    for xml_file in os.listdir(labels_folder):
        if xml_file.endswith(".xml"):
            convert_annotation(
                os.path.join(labels_folder, xml_file),
                args.output_dir,
                classes
            )

    print("✅ Conversion completed! YOLOv8 labels saved in", args.output_dir)

if __name__ == "__main__":
    main()
