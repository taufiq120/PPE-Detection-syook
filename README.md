🚀 AI-Powered Person & PPE Detection System

👷‍♂️ Because Safety Comes First!

Welcome to the ultimate AI-driven safety assistant! This project isn't just another object detection tool—it’s a smart system designed to detect people and analyze their safety gear (hard hats, gloves, masks, vests, etc.), ensuring workplace compliance and accident prevention.

It leverages the power of YOLOv8 for lightning-fast detection and comes with a sleek, interactive interface built using Streamlit & OpenCV for an effortless user experience.

📸 See It in Action!

![Screenshot 2025-03-09 234933](https://github.com/user-attachments/assets/2fcfe642-bbe6-4659-8e89-450887793dc1)

![Screenshot 2025-03-09 234426](https://github.com/user-attachments/assets/2e4d4f32-6669-404c-bd22-725fffa3b7bc)

![Screenshot 2025-03-09 234251](https://github.com/user-attachments/assets/55a6fc07-7baf-46bf-a4e6-a43bb955c7f2)

![Screenshot 2025-03-09 234208](https://github.com/user-attachments/assets/ad84fcf9-d847-4c26-8da9-a432fc5ad615)

![Screenshot 2025-03-09 234335](https://github.com/user-attachments/assets/4d663048-636e-4796-a6df-40096c54db43)

![Screenshot 2025-03-09 234317](https://github.com/user-attachments/assets/1af13be8-d001-432c-9c60-c119fc337773)


![Screenshot 2025-03-09 234449](https://github.com/user-attachments/assets/865e3d4a-3460-425f-84ea-da65bd371461)

===============================================================================

🔥 Features That Make It Stand Out

✔️ Detects persons in an image with precision

✔️ Identifies PPE items like helmets, gloves, masks, vests, boots, and more

✔️ Processes cropped person images to refine PPE detection accuracy

✔️ Web-based Interface—built with Streamlit for easy interaction

✔️ Real-time image processing powered by OpenCV

✔️ Modern UI with a clean, responsive design
-------------------------------------------------------------------------------------------

🛠️ What’s Under the Hood?

Technology	Role

🐍 Python	Backend processing

⚡ YOLOv8	Object detection

🎥 OpenCV	Image handling & visualization

🌍 Streamlit	Interactive web app

🎨 HTML, CSS, JS	Custom UI enhancements
------------------------------------------------------------------------------------------------

📚 How We Built the Models

To make this system accurate and efficient, we trained two models:

1️⃣ Person Detection Model - Finds humans in an image

2️⃣ PPE Detection Model - Scans the cropped person images for safety gear
-------------------------------------------------------------------------------------------------

📌 Dataset & Training Steps

✔ Dataset Format: Pascal VOC XML annotations → YOLO format

✔ Convert annotations using convert_ppe_annotations.py

✔ Train models using train_person_detection.py and train_ppe_detection.py

✔ Run detection with inference.py

==================================================================================================

🚀 Getting Started

📥 Clone the Repository

git clone https://github.com/mars2812/Persons-and-PPE-Detection-using-AI.git

cd PPE-Detection

📦 Install Dependencies

pip install -r requirements.txt

📥 Download YOLOv8

pip install ultralytics

🎯 Running the System

1️⃣ Run Model Inference

python inference.py

This will process images and detect persons and PPE.

2️⃣ Launch the Streamlit App

streamlit run app.py

This will start the Streamlit-based web UI, allowing you to upload images and view results interactively.
----------------------------------------------------------------------------------------------------

💡 Future Improvements

🔹 Add bounding box colors to differentiate PPE items clearly

🔹 Improve model accuracy with larger datasets

🔹 Integrate a database to store and review detections

🔹 Implement real-time video streaming
---------------------------------------------------------------------------------------------------
👨‍💻 Developed By
Taufiq Ahmed I

----------------------------------------------x-----------------------------------------------x--------
