# E-Waste Classification AI Assistant ♻️

An AI-powered chatbot that helps identify e-waste items and provides proper disposal guidance with links to authorized collection centers.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)

## Features

- 🔍 **Image Classification**: Upload images to identify e-waste items
- 📋 **Disposal Instructions**: Get proper disposal guidelines for each item
- 🗺️ **Collection Centers**: Find authorized e-waste recycling centers
- ⚠️ **Hazard Warnings**: Special alerts for hazardous materials
- 💬 **Chat Interface**: Easy-to-use conversational UI

## Supported E-Waste Items

| Item | Hazard Level |
|------|--------------|
| Cables | Low |
| CDs/DVDs | Low |
| TV Remotes | Low |
| Batteries | ⚠️ High |
| Motherboards | ⚠️ High |

## Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   

2. **Create a virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   
   The app will automatically open at `http://localhost:8501`

## Usage

1. Open the application in your browser
2. Upload an image of an electronic item using the file uploader
3. The AI will identify the item and provide:
   - Classification result with confidence score
   - Disposal instructions
   - Tips for proper handling
   - Links to authorized collection centers
4. Upload more images to identify additional items
5. Use "Clear Chat" button in sidebar to start fresh

## Project Structure

```
ewaste_ai_project/
├── app.py                  # Main Streamlit application
├── model_unquant.tflite    # TensorFlow Lite model
├── labels.txt              # Class labels
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Model Information

- **Model Type**: TensorFlow Lite (floating point)
- **Input Size**: 224x224 RGB images
- **Training**: Google Teachable Machine
- **Classes**: 6 (Not e-waste, TV Remote, Cable, Motherboard, Battery, CD)

## Technologies Used

- **Frontend**: Streamlit
- **ML Framework**: TensorFlow Lite
- **Image Processing**: Pillow, NumPy
- **Model Training**: Google Teachable Machine

---

**Note**: Always dispose of e-waste responsibly at authorized collection centers to protect the environment! 🌍


