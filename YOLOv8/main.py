import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image

# FIXME: ignore warnings, should be removed in the future
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Poker Card Recognition",
    page_icon="🃏",
    layout="wide"
)

# Cache model loading
@st.cache_resource
def load_model(model_name):
    """Load the trained YOLO model"""
    model_path = os.path.join(".", model_name, "weights", "best.pt")
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    try:
        # Try using the old version of torch.load
        import torch
        old_load = torch.load
        def safe_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return old_load(*args, **kwargs)
        torch.load = safe_load
        
        model = YOLO(model_path)
        
        # Restore the original torch.load
        torch.load = old_load
        
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def get_available_models():
    """Get the list of available models"""
    available_models = []
    model_folders = ["Model-35e", "Model-50e", "Model-65e"]
    
    for folder in model_folders:
        model_path = os.path.join(".", folder, "weights", "best.pt")
        if os.path.exists(model_path):
            available_models.append(folder)
    
    return available_models

def predict_cards(image, model, confidence_threshold=0.5):
    """
    Predict poker cards in the image
    
    Args:
        image: Input image
        model: YOLO model
        confidence_threshold: Confidence threshold
    
    Returns:
        processed_image: Processed image (with annotations)
        predictions: List of prediction results
    """
    if model is None:
        return None, []
    
    # Perform prediction
    results = model(image, conf=confidence_threshold, verbose=False)[0]
    
    # Copy image for drawing
    processed_image = image.copy()
    predictions = []
    
    if results.boxes is not None and len(results.boxes) > 0:
        # Get prediction data
        boxes_data = results.boxes.data.cpu().numpy()
        
        for box_data in boxes_data:
            x1, y1, x2, y2, confidence, class_id = box_data
            
            if confidence >= confidence_threshold:
                # Get class name
                class_name = results.names[int(class_id)]
                
                # Draw bounding box
                cv2.rectangle(processed_image, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Draw label background
                cv2.rectangle(processed_image,
                            (int(x1), int(y1) - label_size[1] - 10),
                            (int(x1) + label_size[0], int(y1)),
                            (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(processed_image, label,
                          (int(x1), int(y1) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                          (0, 0, 0), 2, cv2.LINE_AA)
                
                # Save prediction result
                predictions.append({
                    'card': class_name,
                    'confidence': float(confidence),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
    
    return processed_image, predictions

def main():
    """Main function"""
    # Title
    st.title("🃏 Poker Card Recognition System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Get available models
    available_models = get_available_models()
    if not available_models:
        st.error("No available model files found!")
        return
    
    # Model selection dropdown
    model_display_names = {
        "Model-35e": "35 Epochs Training Model (Model-35e)", 
        "Model-50e": "50 Epochs Training Model (Model-50e)",
        "Model-65e": "65 Epochs Training Model (Model-65e)"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=available_models,
        format_func=lambda x: model_display_names.get(x, x),
        index=0,
        help="Select the training model to use. Different models may have different accuracy and performance."
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="A higher threshold will reduce false positives but may miss some cards."
    )
    
    # Load model
    with st.spinner(f"Loading model {model_display_names.get(selected_model, selected_model)}..."):
        model = load_model(selected_model)
    
    if model is None:
        st.error("Model loading failed, please check if the model file exists.")
        return
    
    st.success(f"✅ Model loaded successfully! Currently using: {model_display_names.get(selected_model, selected_model)}")
    
    # File upload
    st.header("📁 Upload Image")
    uploaded_file = st.file_uploader(
        "Select an image containing poker cards",
        type=['png', 'jpg', 'jpeg'],
        help="Supports PNG, JPG, JPEG formats"
    )
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        
        # Convert to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Create two-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Perform prediction
        with st.spinner("Recognizing poker cards..."):
            processed_image, predictions = predict_cards(
                image_cv, model, confidence_threshold
            )
        
        if processed_image is not None:
            with col2:
                st.subheader("Prediction Results")
                # Convert back to RGB format for display
                processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                st.image(processed_image_rgb, use_container_width=True)
            
            # Display prediction details
            st.header("🎯 Recognition Details")
            
            if predictions:
                st.success(f"Detected {len(predictions)} poker cards")
                
                # Create prediction results table
                prediction_data = []
                for i, pred in enumerate(predictions, 1):
                    prediction_data.append({
                        "Index": i,
                        "Card": pred['card'],
                        "Confidence": f"{pred['confidence']:.3f}",
                        "Bounding Box": f"({pred['bbox'][0]}, {pred['bbox'][1]}) - ({pred['bbox'][2]}, {pred['bbox'][3]})"
                    })
                
                st.table(prediction_data)
                
                # Statistics
                st.subheader("📊 Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Number of Card Corners Detected", len(predictions))
                
                with col2:
                    avg_confidence = np.mean([pred['confidence'] for pred in predictions])
                    st.metric("Average Confidence", f"{avg_confidence:.3f}")
                
                with col3:
                    max_confidence = max([pred['confidence'] for pred in predictions])
                    st.metric("Highest Confidence", f"{max_confidence:.3f}")
                
                
            else:
                st.warning("No poker cards detected, please try:")
                st.write("- Lowering the confidence threshold")
                st.write("- Ensuring there are clear poker cards in the image")
                st.write("- Checking the image quality and lighting conditions")
        
        else:
            st.error("Image processing failed")
    
    else:
        # Display usage instructions
        st.info("👆 Please upload an image containing poker cards to start recognition")
        
        # Display model information
        if model is not None:
            st.header("ℹ️ Model Information")
            st.write(f"- **Current Model**: {model_display_names.get(selected_model, selected_model)}")
            st.write(f"- **Model Type**: YOLOv8")
            st.write(f"- **Number of Supported Classes**: 52 (Standard Poker Cards)")
            st.write(f"- **Supported Suits**: ♠️ ♥️ ♦️ ♣️")
            st.write(f"- **Supported Ranks**: 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A")
            
            # Display model training information
            model_info = {
                "Model-35e": "35 Epochs Training - Quick Training Version",
                "Model-50e": "50 Epochs Training - Balanced Version", 
                "Model-65e": "65 Epochs Training - High Precision Version"
            }
            if selected_model in model_info:
                st.write(f"- **Model Description**: {model_info[selected_model]}")

if __name__ == "__main__":
    main()
