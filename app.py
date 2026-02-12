import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(page_title="Messy Desk Scanner", page_icon="🧹")

# Load YOLOv8 model (cached to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define object weights for scoring
weights = {
    "pen": 1, "paper": 2, "mug": 3, "trash": 5, "book": 2, "cable": 3, "bottle": 4, "clothes": 3,
}

def compute_chaos_score(results):
    score = 0
    # results is a list, we only process the first image (results[0])
    for box in results[0].boxes:
        label_idx = int(box.cls)
        # Check if the class index is within the names dictionary
        if label_idx in results[0].names:
            label = results[0].names[label_idx]

            if label == "person":
                continue
            
            # Add to score if the label is in our weights dictionary, default to 1 if not found but detected?
            # Creating a slightly more robust scoring:
            # If it's a known 'messy' item, use its weight.
            # If it's not in the list but not a person, maybe give it a default low score?
            # The original code used `weights.get(label, 1)`, so we'll stick to that.
            score += weights.get(label, 1)
        
    return score

def get_chaos_level(score):
    if score == 0:
        return "✨ Spotless! No clutter detected.", "success"
    elif score <= 5:
        return "🧼 Very tidy desk. Keep it up!", "success"
    elif score <= 10:
        return "🗂️ Slightly messy, but manageable.", "info"
    elif score <= 20:
        return "📚 Messy — might need some cleaning.", "warning"
    else:
        return "🚨 Disaster zone! Time to declutter.", "error"

st.title("🧹 AI Messy Desk Scanner")
st.write("Upload a photo or use your webcam to analyze how messy your desk is!")

# Input method selection
option = st.radio("Choose Input Method", ("Camera", "Upload Image"))

image_input = None

if option == "Camera":
    image_input = st.camera_input("Take a picture")
else:
    image_input = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_input:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Run inference
    results = model(img)
    
    # Calculate score
    chaos_score = compute_chaos_score(results)
    chaos_message, status = get_chaos_level(chaos_score)
    
    # Display results
    st.subheader("Analysis Results")
    
    # Display the chaos level with appropriate styling
    if status == "success":
        st.success(f"**Chaos Score: {chaos_score}**\n\n{chaos_message}")
    elif status == "info":
        st.info(f"**Chaos Score: {chaos_score}**\n\n{chaos_message}")
    elif status == "warning":
        st.warning(f"**Chaos Score: {chaos_score}**\n\n{chaos_message}")
    else:
        st.error(f"**Chaos Score: {chaos_score}**\n\n{chaos_message}")

    # Plot results on the image
    res_plotted = results[0].plot()
    
    # Convert BGR to RGB for Streamlit display
    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    st.image(res_plotted_rgb, caption="Detected Objects", use_column_width=True)

    # Optional: Display detected items list
    with st.expander("See detected items"):
        detected_items = {}
        for box in results[0].boxes:
            label_idx = int(box.cls)
            label = results[0].names[label_idx]
            if label != "person":
                 detected_items[label] = detected_items.get(label, 0) + 1
        
        if detected_items:
            for item, count in detected_items.items():
                st.write(f"- {item}: {count}")
        else:
            st.write("No typical desk items detected.")

