
# Image Captioning & Instance Segmentation



This repo integrates **Mask R-CNN** (instance segmentation) and a **BLIP** captioner to produce overlays and captions.

---

## Quick Setup & Run

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Download NLTK resources
python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('wordnet', quiet=True)"

# Create outputs folder
mkdir -p outputs

# Run Streamlit demo
streamlit run app/streamlit_app.py

or Simply Run i have also created 
```bash
python main.py
```bash
Image Captioning + Instance Segmentation
 Overview
This project combines Instance Segmentation (detecting objects and their masks)
with Image Captioning (describing images in natural language) using PyTorch,
Torchvision, and HuggingFace Transformers.

The app runs on Streamlit with an easy-to-use interface —
upload any image, see detected objects, their masks, and automatically generated captions
for the entire image and for each detected object.

 Features
Instance Segmentation using Mask R-CNN (torchvision.models.detection)

Image Captioning using BLIP (Salesforce/blip-image-captioning-base)

Interactive UI with Streamlit sliders for:

Detection confidence threshold

Number of objects shown (top_k)

Per-object cropping with captions

Runs locally or in GitHub Codespaces (CPU mode supported)

End-to-end: just pip install -r requirements.txt + python main.py

 Project Structure

image-caption-seg/
├── app/
│   └── streamlit_app.py         # Streamlit UI
├── src/
│   ├── models/
│   │   ├── captioning.py        # CaptioningModel (BLIP)
│   │   └── segmentation.py      # SegmentationModel (Mask R-CNN)
│   ├── utils.py                 # Helper functions
├── data/                        # (Optional: store sample images here)
├── outputs/                     # Saved results
├── main.py                      # Launcher for Streamlit app
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
 Installation
1️⃣ Clone the Repository
bash
git clone https://github.com/<your-username>/image-caption-seg.git
cd image-caption-seg
2️⃣ Install Dependencies
bash
pip install -r requirements.txt
 requirements.txt
txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
pillow>=9.0.0
numpy>=1.23.0
streamlit>=1.25.0
▶️ Running the App
Option A: Local PC
python main.py
Then open the link in your browser (e.g., http://localhost:8501).

Option B: GitHub Codespaces
Open repo in Codespaces

Install dependencies:
bash
pip install -r requirements.txt
Start app:

bash

python main.py
In the Ports tab:

Make Port 8501 Public

Open the provided URL (e.g., https://<codespace-id>-8501.app.github.dev)

## --> How to Use
Upload an image (.jpg, .jpeg, .png, .webp)

Adjust:

Detection confidence

Number of top objects to show

View:

Overlay with bounding boxes & masks

Global caption for full image

Per-object crops with captions

## --> Screenshots
![alt text](image-2.png)
Main UI

Per-object Captions

## --> Tech Stack
Language: Python 3.9+

Frameworks: PyTorch, Streamlit

Models:

Segmentation: Mask R-CNN (torchvision)

Captioning: BLIP (transformers)

Tools: Pillow, NumPy

## --> Credits
PyTorch

Torchvision

HuggingFace Transformers

Salesforce BLIP

Streamlit

## --> License
MIT License.
Feel free to fork and improve!

## --> Future Improvements

Allow multiple image uploads

Save & export captions + masks

Add multilingual captioning


