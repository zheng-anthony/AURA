# 🛣️ AURA: Smart City Infrastructure AI

AURA (Automated Urban Road Assessment) is a deep-learning pipeline and Streamlit dashboard that analyzes uploaded street-level images and recorded videos for pavement distress. It was built by Team 14D for the AI4ALL Ignite Program.

## 🎯 Objective

Road inspections are often slow, manual, and reactive. AURA demonstrates how computer vision can identify and map pavement damage so municipalities can prioritize maintenance work.

## ✨ Key features

- **Image and recorded-video detection:** A custom-trained YOLOv8s model detects four road-hazard classes in uploaded photos and recorded video frames:
  - D00: Longitudinal cracks
  - D10: Transverse cracks
  - D20: Alligator cracking
  - D40: Potholes
- **Geospatial mapping:** AURA reads GPS EXIF data from supported images and one recording-level GPS location from supported video metadata. If metadata is absent or was removed during sharing, the dashboard uses the latitude and longitude entered in the sidebar.
- **Analysis reports:** Model predictions are converted into readable hazard summaries.
- **Bias mitigation:** Training incorporated negative samples such as healthy roads, shadows, and tar lines, plus synthetic weather augmentation with Albumentations.

## 📤 Supported uploads

- Images: JPG, JPEG, and PNG
- Recorded videos: MP4, MOV, and M4V
- Video limits: 30 seconds and 100 MB per upload

Video detection totals are **frame-level counts**, not counts of unique physical hazards. The same pothole or crack may therefore be counted in multiple frames. Annotated output videos are generated without an audio track.

On some phones, the upload control may offer **Record Video** or a camera option through the device's native file picker. This behavior depends on the phone, browser, and permissions; AURA does not provide continuous live-camera streaming. A recorded clip must finish before it can be uploaded and analyzed.

## 🚀 Run locally

Clone the repository:

```bash
git clone https://github.com/zheng-anthony/AURA.git
cd AURA
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the dashboard:

```bash
python -m streamlit run src/app.py
```

Open the displayed local URL, upload a supported image or recorded video, and review the annotated detections and map. Original phone files are most likely to retain GPS metadata; messaging, social-media, and editing applications may strip it.

## 📊 Model evaluation

The model was trained for 50 epochs using a merged dataset of RDD2022 and high-resolution street-level pothole imagery.

1. **Training results and loss curves:** Training classification loss declined sharply before the final mosaic-close phase in epochs 40–50.
2. **Confusion matrix:** Negative sampling helped the model distinguish road damage from background features such as shadows and manholes.
3. **F1-score and precision-recall:** The model achieved an F1 score of 0.62 and more than 0.73 mAP@0.5 for D40 potholes.
