import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Detection System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 20px;
        color: #558B2F;
        text-align: center;
        padding-bottom: 30px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #F1F8E9;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and training history
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('trained_plant_disease_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_training_history():
    try:
        with open('training_hist.json', 'r') as f:
            history = json.load(f)
        return history
    except Exception as e:
        st.error(f"Error loading training history: {e}")
        return None

# Class names for the 38 plant disease categories
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def predict_disease(model, image):
    """Predict plant disease from image"""
    # Resize image to model input size
    img = image.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class] * 100)
    
    return CLASS_NAMES[predicted_class], confidence, predictions[0]

def format_disease_name(disease_name):
    """Format disease name for better display"""
    parts = disease_name.replace('_', ' ').split('   ')
    if len(parts) == 2:
        plant = parts[0].strip()
        disease = parts[1].strip()
        return plant, disease
    return disease_name, ""

def main():
    # Header
    st.markdown('<p class="main-header">🌿 Plant Disease Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Deep Learning Solution for Agricultural Disease Diagnosis</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("⚠️ Failed to load the model. Please ensure 'trained_plant_disease_model.keras' is in the directory.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/leaf.png", width=80)
        st.title("Navigation")
        page = st.radio("Go to", ["🏠 Home", "🔬 Disease Detection", "📊 Model Performance", "ℹ️ About"])
        
        st.markdown("---")
        st.markdown("### Project Info")
        st.markdown("""
        **Student:** Subhash Halder  
        **Guide:** Ayan Pal  
        **Institution:** Amity University Online  
        **Program:** MCA (Machine Learning)
        """)
    
    # Main content based on page selection
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔬 Disease Detection":
        show_detection_page(model)
    elif page == "📊 Model Performance":
        show_performance_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Display home page"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## Welcome to Plant Disease Detection System")
        st.markdown("""
        ### 🎯 Purpose
        This system uses advanced **Convolutional Neural Networks (CNN)** to identify plant diseases 
        from leaf images, helping farmers and agricultural professionals make quick, informed decisions.
        
        ### 🌟 Key Features
        - **38 Disease Classes**: Detects diseases across multiple plant species
        - **High Accuracy**: Achieves 98.35% training accuracy and 95.57% validation accuracy
        - **Real-time Detection**: Upload an image and get instant results
        - **Deep Learning Powered**: Built with TensorFlow and Keras
        
        ### 🌱 Supported Plants
        """)
        
        plants_col1, plants_col2, plants_col3 = st.columns(3)
        with plants_col1:
            st.markdown("- 🍎 Apple\n- 🫐 Blueberry\n- 🍒 Cherry\n- 🌽 Corn")
        with plants_col2:
            st.markdown("- 🍇 Grape\n- 🍊 Orange\n- 🍑 Peach\n- 🫑 Pepper")
        with plants_col3:
            st.markdown("- 🥔 Potato\n- 🍓 Strawberry\n- 🍅 Tomato\n- 🫛 Soybean")
    
    with col2:
        st.image("https://img.icons8.com/fluency/240/000000/plant-under-sun.png")
        
        st.markdown("### 📈 Quick Stats")
        st.metric("Training Images", "70,295")
        st.metric("Validation Images", "17,572")
        st.metric("Disease Classes", "38")
        st.metric("Model Accuracy", "95.57%")

def show_detection_page(model):
    """Display disease detection page"""
    st.markdown("## 🔬 Plant Disease Detection")
    st.markdown("Upload an image of a plant leaf to detect potential diseases.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a plant leaf image (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a plant leaf for disease detection"
    )
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        with col1:
            st.markdown("### 📸 Uploaded Image")
            st.image(image, use_container_width=True)
        
        # Make prediction
        with st.spinner('🔍 Analyzing image...'):
            disease_name, confidence, all_predictions = predict_disease(model, image)
            plant, disease = format_disease_name(disease_name)
        
        with col2:
            st.markdown("### 🎯 Prediction Results")
            
            # Display prediction
            if "healthy" in disease_name.lower():
                st.success(f"✅ **Plant Status:** Healthy")
                st.markdown(f"**Plant Type:** {plant}")
            else:
                st.warning(f"⚠️ **Disease Detected:** {disease}")
                st.markdown(f"**Plant Type:** {plant}")
            
            st.markdown(f"**Confidence:** {confidence:.2f}%")
            
            # Progress bar for confidence
            st.progress(float(confidence / 100))
            
            # Recommendation
            st.markdown("---")
            st.markdown("### 💡 Recommendation")
            if confidence > 85:
                st.info("High confidence prediction. The diagnosis is reliable.")
            elif confidence > 70:
                st.warning("Moderate confidence. Consider consulting an expert for confirmation.")
            else:
                st.error("Low confidence. Please upload a clearer image or consult an expert.")
        
        # Top 5 predictions
        st.markdown("---")
        st.markdown("### 📊 Top 5 Predictions")
        
        # Get top 5 predictions
        top_5_idx = np.argsort(all_predictions)[-5:][::-1]
        top_5_diseases = [CLASS_NAMES[i] for i in top_5_idx]
        top_5_confidences = [all_predictions[i] * 100 for i in top_5_idx]
        
        # Create dataframe
        df = pd.DataFrame({
            'Disease': top_5_diseases,
            'Confidence (%)': top_5_confidences
        })
        
        # Display as bar chart
        fig = px.bar(df, x='Confidence (%)', y='Disease', orientation='h',
                     color='Confidence (%)', color_continuous_scale='Greens')
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("👆 Please upload an image to begin detection")
        
        # Show example
        st.markdown("---")
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Click on the **Browse files** button above
        2. Select a clear image of a plant leaf
        3. Wait for the AI model to analyze the image
        4. View the prediction results and recommendations
        
        **Tips for best results:**
        - Use clear, well-lit images
        - Focus on the affected area of the leaf
        - Avoid blurry or low-resolution images
        - Ensure the leaf fills most of the frame
        """)

def show_performance_page():
    """Display model performance metrics"""
    st.markdown("## 📊 Model Performance Metrics")
    
    # Load training history
    history = load_training_history()
    
    if history is None:
        st.error("Failed to load training history.")
        return
    
    # Display key metrics
    st.markdown("### 🎯 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Training Accuracy", f"{history['accuracy'][-1]*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Validation Accuracy", f"{history['val_accuracy'][-1]*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Training Loss", f"{history['loss'][-1]:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Validation Loss", f"{history['val_loss'][-1]:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Accuracy plot
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Accuracy Over Epochs")
        epochs = list(range(1, len(history['accuracy']) + 1))
        
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=epochs, y=[acc*100 for acc in history['accuracy']], 
                                     mode='lines+markers', name='Training Accuracy',
                                     line=dict(color='#4CAF50', width=3)))
        fig_acc.add_trace(go.Scatter(x=epochs, y=[acc*100 for acc in history['val_accuracy']], 
                                     mode='lines+markers', name='Validation Accuracy',
                                     line=dict(color='#2196F3', width=3)))
        fig_acc.update_layout(xaxis_title='Epoch', yaxis_title='Accuracy (%)',
                             hovermode='x unified', height=400)
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        st.markdown("### 📉 Loss Over Epochs")
        
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=history['loss'], 
                                      mode='lines+markers', name='Training Loss',
                                      line=dict(color='#FF5722', width=3)))
        fig_loss.add_trace(go.Scatter(x=epochs, y=history['val_loss'], 
                                      mode='lines+markers', name='Validation Loss',
                                      line=dict(color='#FF9800', width=3)))
        fig_loss.update_layout(xaxis_title='Epoch', yaxis_title='Loss',
                              hovermode='x unified', height=400)
        st.plotly_chart(fig_loss, use_container_width=True)
    
    # Model architecture
    st.markdown("---")
    st.markdown("### 🏗️ Model Architecture")
    
    arch_col1, arch_col2 = st.columns([2, 1])
    
    with arch_col1:
        st.markdown("""
        **Convolutional Neural Network (CNN) Architecture:**
        
        - **Input Layer:** 128 x 128 x 3 (RGB images)
        - **Convolutional Block 1:** 2 Conv2D layers (32 filters) + MaxPooling
        - **Convolutional Block 2:** 2 Conv2D layers (64 filters) + MaxPooling
        - **Convolutional Block 3:** 2 Conv2D layers (128 filters) + MaxPooling
        - **Convolutional Block 4:** 2 Conv2D layers (256 filters) + MaxPooling
        - **Convolutional Block 5:** 2 Conv2D layers (512 filters) + MaxPooling
        - **Dropout Layer:** 25% dropout for regularization
        - **Flatten Layer:** Convert 2D features to 1D
        - **Dense Layer:** 1500 neurons with ReLU activation
        - **Dropout Layer:** 40% dropout to prevent overfitting
        - **Output Layer:** 38 neurons with Softmax activation
        
        **Training Configuration:**
        - Optimizer: Adam (learning rate: 0.0001)
        - Loss Function: Categorical Crossentropy
        - Batch Size: 32
        - Epochs: 10
        """)
    
    with arch_col2:
        st.markdown("**Dataset Statistics:**")
        st.markdown(f"- Training samples: 70,295")
        st.markdown(f"- Validation samples: 17,572")
        st.markdown(f"- Total classes: 38")
        st.markdown(f"- Image size: 128x128")
        st.markdown(f"- Color mode: RGB")
        
        st.markdown("---")
        st.markdown("**Training Time:**")
        st.markdown("- ~130 seconds/epoch")
        st.markdown("- Total: ~22 minutes")

def show_about_page():
    """Display about page"""
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    ### 📚 Project Overview
    
    The **Plant Disease Detection System** is an advanced machine learning application developed as part 
    of the MCA (Master of Computer Applications) curriculum at Amity University Online. This project 
    demonstrates the practical application of deep learning in agriculture for automated disease diagnosis.
    
    ### 🎓 Academic Information
    
    - **Project Title:** Plant Disease Detection Using Convolutional Neural Networks
    - **Student Name:** Subhash Halder
    - **Enrollment No:** A9929724000690(el)
    - **Program:** MCA (Machine Learning Specialization)
    - **Semester:** 4th Semester, Year 2025
    - **Institution:** Amity University Online
    - **Project Guide:** Ayan Pal (MTech in Computer Science)
    - **Guide Designation:** Senior Engineering Manager, Walmart
    - **Guide Experience:** 15 years
    
    ### 🔬 Technical Details
    
    **Dataset:**
    - Source: Kaggle - New Plant Diseases Dataset
    - Total Images: 87,867 (70,295 training + 17,572 validation)
    - Classes: 38 plant disease categories
    - Plants Covered: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, 
      Soybean, Squash, Strawberry, Tomato
    
    **Technology Stack:**
    - **Deep Learning Framework:** TensorFlow 2.16.2, Keras 3.12.0
    - **Programming Language:** Python 3.12
    - **Web Framework:** Streamlit 1.51.0
    - **Data Processing:** NumPy, Pandas
    - **Visualization:** Matplotlib, Plotly, Seaborn
    - **Image Processing:** OpenCV, PIL
    - **Hardware Acceleration:** Apple Metal GPU (M4 Pro)
    
    **Model Architecture:**
    - Custom CNN with 5 convolutional blocks
    - Progressive filter increase: 32 → 64 → 128 → 256 → 512
    - Dropout regularization (25% and 40%)
    - Dense layer with 1,500 neurons
    - Softmax output layer for 38 classes
    
    ### 🎯 Project Objectives
    
    1. Develop an accurate deep learning model for plant disease classification
    2. Achieve >90% accuracy in disease detection
    3. Create a user-friendly web interface for real-world deployment
    4. Contribute to precision agriculture and early disease diagnosis
    5. Demonstrate practical application of CNN in image classification
    
    ### 📊 Achievements
    
    - ✅ Training Accuracy: **98.35%**
    - ✅ Validation Accuracy: **95.57%**
    - ✅ Low validation loss: **0.1747**
    - ✅ Robust model with minimal overfitting
    - ✅ Real-time prediction capability
    - ✅ Interactive web application
    
    ### 🌍 Real-World Applications
    
    - Early detection of plant diseases in agricultural fields
    - Support for farmers in making informed treatment decisions
    - Reduction in crop losses through timely intervention
    - Cost-effective alternative to manual inspection
    - Scalable solution for large-scale farming operations
    - Educational tool for agricultural students and professionals
    
    ### 🔮 Future Enhancements
    
    - Mobile application for on-field diagnosis
    - Integration with IoT devices for automated monitoring
    - Expansion to more plant species and diseases
    - Multi-language support for global accessibility
    - Cloud deployment for wider reach
    - Integration with agricultural advisory systems
    - Real-time disease severity assessment
    
    ### 📄 License & Usage
    
    This project is developed for academic purposes as part of MCA curriculum. 
    The model and code can be used for educational and research purposes with proper attribution.
    
    ### 📞 Contact
    
    For questions or collaboration opportunities, please contact:
    - **Student:** Subhash Halder
    - **Email:** subhash23@amityonline.com
    - **Project Guide:** Ayan Pal, Walmart
    - **GitHub:** [https://github.com/subhash-halder/plant-disease-detection-system](https://github.com/subhash-halder/plant-disease-detection-system)
    
    ---
    
    **Acknowledgments:** Special thanks to Ayan Pal for guidance, Amity University Online for 
    academic support, and the Kaggle community for providing the dataset.
    """)

if __name__ == "__main__":
    main()

