import streamlit as st
import numpy as np
from PIL import Image

# Disable GPU and suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# Page config
st.set_page_config(page_title="E-Waste Assistant", page_icon="recycle", layout="centered")

# Load TFLite model
@st.cache_resource
def load_model():
    try:
        interp = tf.lite.Interpreter(model_path="model_unquant.tflite")
        interp.allocate_tensors()
        return interp
    except Exception as e:
        return None

interpreter = load_model()
input_details = interpreter.get_input_details() if interpreter else None
output_details = interpreter.get_output_details() if interpreter else None

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[-1] for line in f.readlines()]

# E-waste data with disposal instructions and map links
ewaste_data = {
    "CD": {
        "category": "Mixed E-Waste / Plastic",
        "instruction": "CDs should not be burned or thrown in household waste. They contain plastic and metal layers. Please take them to authorized mixed e-waste recycling centers for safe processing.",
        "map_link": "https://www.google.com/maps/search/authorized+e-waste+recycling+centers+Chennai",
        "hazard": False
    },
    "Battery": {
        "category": "Hazardous E-Waste",
        "instruction": "Batteries contain hazardous chemicals and heavy metals. Do not dispose of them in regular trash. Store safely and hand over only to authorized battery or e-waste collection centers.",
        "map_link": "https://www.google.com/maps/search/battery+e-waste+collection+Chennai",
        "hazard": True
    },
    "Tv Remote": {
        "category": "Consumer Electronics",
        "instruction": "Remove the batteries before disposal. Batteries must be disposed of separately in battery collection bins. The remote body should be handed over to an authorized e-waste recycler.",
        "map_link": "https://www.google.com/maps/search/e-waste+collection+centers+Chennai",
        "hazard": False
    },
    "Motherboard": {
        "category": "Hazardous Electronic Waste",
        "instruction": "Motherboards contain lead, mercury, and valuable metals. Do not dismantle at home. Hand over only to certified e-waste recyclers who safely extract materials.",
        "map_link": "https://www.google.com/maps/search/authorized+electronics+recyclers+Chennai",
        "hazard": True
    },
    "Cable": {
        "category": "Mixed E-Waste / Plastic",
        "instruction": "Cables contain copper and plastic materials. Please take them to authorized e-waste recycling centers for proper recycling and material recovery.",
        "map_link": "https://www.google.com/maps/search/e-waste+collection+centers+Chennai",
        "hazard": False
    },
    "Not a e-waste": {
        "category": "General Waste / Non-Electronic",
        "instruction": "Oops! This item does not appear to be electronic waste. No special e-waste disposal is required. Please dispose of it according to your local municipal waste guidelines.",
        "map_link": "https://www.google.com/maps/search/municipal+waste+collection+Chennai",
        "hazard": False
    }
}

# Prediction function
def predict(image):
    if interpreter is None:
        return None, 0
    
    # Resize to 224x224 as Teachable Machine expects
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    # Teachable Machine normalization: (pixel / 127.5) - 1 gives range [-1, 1]
    img_array = (img_array / 127.5) - 1
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # DEBUG: Print all predictions to terminal
    print("\n" + "="*50)
    print("MODEL PREDICTION DEBUG")
    print("="*50)
    for i, (label, prob) in enumerate(zip(labels, predictions[0])):
        print(f"  {i}: {label:20s} -> {prob*100:.2f}%")
    
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    
    print("-"*50)
    print(f"  WINNER: {labels[predicted_idx]} ({confidence*100:.2f}%)")
    print("="*50 + "\n")
    
    return labels[predicted_idx], confidence

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I am your **E-Waste Assistant**.\n\nUpload an image of any electronic item and I will help you:\n- Identify if it is e-waste\n- Tell you how to dispose it properly\n- Find authorized collection centers\n\n**Upload an image below to get started!**"
    })

# UI Header
st.title("E-Waste Assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "image" in message:
            st.image(message["image"], width=200)
        st.markdown(message["content"], unsafe_allow_html=True)
        # Show map preview if available
        if "map_link" in message:
            search_query = message["map_link"].split("search/")[1] if "search/" in message["map_link"] else "e-waste+collection+Chennai"
            map_url = f"https://www.google.com/maps?q={search_query}&output=embed"
            st.components.v1.iframe(map_url, height=300)

# Image upload
uploaded_file = st.file_uploader("Upload an image of e-waste", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_id = uploaded_file.name + str(uploaded_file.size)
    
    if "last_image_id" not in st.session_state or st.session_state.last_image_id != image_id:
        st.session_state.last_image_id = image_id
        
        st.session_state.messages.append({
            "role": "user",
            "content": "Can you identify this item?",
            "image": image
        })
        
        if interpreter:
            label, confidence = predict(image)
            info = ewaste_data.get(label, ewaste_data["Not a e-waste"])
            
            response = f"### 🔍 Detection Result\n\n"
            response += f"**Identified Item:** {label}\n\n"
            response += f"**Category:** {info['category']}\n\n"
            response += f"**Confidence:** {confidence*100:.1f}%\n\n"
            response += "---\n\n"
            response += f"### ♻️ Disposal Instruction\n\n"
            response += f"{info['instruction']}\n\n"
            
            if info['hazard']:
                response += "\n⚠️ **WARNING:** This is hazardous waste. Please handle with care and dispose only through authorized channels!\n\n"
            
            response += "---\n\n"
            response += "### 📍 Find Authorized E-Waste Collection Centers\n\n"
            response += f"� **[Click here to find nearby collection centers on Google Maps]({info['map_link']})**\n\n"
            
            # Add embedded map preview
            map_embed_link = info['map_link'].replace("search/", "embed/v1/search?key=&q=")
            response += f'<iframe width="100%" height="300" style="border:0; border-radius:10px;" loading="lazy" allowfullscreen referrerpolicy="no-referrer-when-downgrade" src="https://www.google.com/maps/embed/v1/search?key=AIzaSyBFw0Qbyq9zTFTd-tUY6dZWTgaQzuU17R8&q={info["map_link"].split("search/")[1]}"></iframe>\n\n'
            
            response += "\n---\n*Upload another image to identify more items!*"
            
            st.session_state.messages.append({"role": "assistant", "content": response, "map_link": info['map_link']})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Error: Model not loaded."})
        
        st.rerun()

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This AI assistant helps you identify e-waste and provides proper disposal guidance.
    
    **Supported Items:**
    - Cables
    - CDs/DVDs
    - Batteries
    - TV Remotes
    - Motherboards
    """)
    st.divider()
    st.header("Useful Links")
    st.markdown("""
    - [E-Waste Guide India](https://cpcb.nic.in/e-waste/)
    - [CPCB E-Waste Rules](https://cpcb.nic.in/e-waste-rules/)
    - [Find E-Waste Centers](https://www.google.com/maps/search/authorized+e-waste+recycling+centers/)
    """)
    st.divider()
    if st.button("Clear Chat", key="clear_chat_btn"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your **E-Waste Assistant**.\n\nUpload an image to get started!"}]
        if "last_image_id" in st.session_state:
            del st.session_state.last_image_id
        st.rerun()
