🛣️ AURA: Smart City Infrastructure AI

AURA (Automated Urban Road Assessment) is an automated deep learning pipeline and interactive web dashboard designed to process street-level dashcam imagery, dynamically flagging and mapping severe pavement distress. Built by Team 14D for the AI4ALL Ignite Program.

🎯 The Objective

Current civil maintenance frameworks rely on slow, manual, and reactive inspections. AURA transitions this to a proactive, data-driven workflow. By detecting and classifying structural failures (like potholes and alligator cracking) in real-time, municipalities can optimize repair budgets and navigation apps can reroute commuters around dangerous hazards.

✨ Key Features

Real-Time Object Detection: Utilizes a custom-trained YOLOv8s Convolutional Neural Network to classify 4 specific road hazards:

D00: Longitudinal Cracks

D10: Transverse Cracks

D20: Alligator Cracking

D40: Potholes

Geospatial EXIF Mapping: The Python backend automatically extracts hidden GPSInfo metadata from uploaded smartphone images, instantly dropping a maintenance pin on an interactive map.

Dynamic Analysis Reports: Translates raw YOLO tensor outputs into human-readable maintenance summaries.

Advanced Bias Mitigation: Trained using Negative Sampling (images of healthy roads, shadows, and tar lines) and Synthetic Weather Augmentation (Albumentations) to drastically reduce false positives.

🚀 How to Run Locally

Clone the repository:

git clone https://github.com/zheng-anthony/AURA.git
cd AURA


Install the required dependencies:

pip install ultralytics opencv-python streamlit pandas pillow


Run the Streamlit Dashboard:

streamlit run ai4all_dashboard.py


Upload a photo of a pothole (preferably taken on a smartphone with location services enabled) to see the AI inference and map marker in action!

📊 Model Evaluation & Metrics

Our model was rigorously trained for 50 Epochs using a merged dataset of RDD2022 and high-resolution street-level pothole imagery. Below are the finalized validation metrics proving the efficacy of our bias mitigation strategy.

1. Training Results & Loss Curves

Notice the steep decline in train/cls_loss and the lock-in of generalizable weights during the final "Mosaic Close" phase (Epochs 40-50).


2. Confusion Matrix (Error Analysis)

Our Negative Sampling strategy successfully taught the AI to ignore severe visual noise (shadows/manholes) in the "Background" class, while achieving near-perfect separation between Potholes (D40) and Transverse Cracks (D10).


3. F1-Score & Precision-Recall

Road data is heavily imbalanced. Despite this, the model achieved a highly robust harmonic mean (F1) of 0.62, surging past 0.73 mAP@0.5 for severe, structural D40 Potholes.

