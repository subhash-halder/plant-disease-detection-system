# CHAPTER 5: SYSTEM DESIGN AND ARCHITECTURE

## 5.1 Introduction

This chapter presents the comprehensive system design and architecture of the plant disease detection system. The design encompasses the CNN model architecture, web application structure, data flow diagrams, component interactions, and deployment considerations. The system architecture follows a modular design philosophy, separating concerns between model development, inference logic, user interface, and data management to facilitate maintenance, testing, and future enhancements.

The complete system consists of two primary components: the CNN classification model and the Streamlit web application. These components integrate through well-defined interfaces, enabling independent development and testing while ensuring seamless operation in the deployed system.

## 5.2 System Architecture Overview

### 5.2.1 High-Level Architecture

The plant disease detection system follows a client-server architecture adapted for web-based machine learning applications:

**Client Layer (User Interface):**
- Web browser accessing Stream

lit application
- Image upload interface
- Results display and visualization
- Interactive charts and navigation

**Application Layer (Web Server):**
- Streamlit framework handling HTTP requests
- Image preprocessing pipeline
- Model inference orchestration
- Result formatting and visualization

**Model Layer (ML Backend):**
- Trained Keras/TensorFlow model
- Feature extraction through convolutional layers
- Classification through fully connected layers
- Confidence score computation

**Data Layer:**
- Trained model weights (saved as .keras file)
- Training history (saved as JSON)
- Uploaded user images (temporary storage)
- Static assets (images, stylesheets)

### 5.2.2 Component Interaction

**User Interaction Flow:**
1. User navigates to web application URL
2. Streamlit renders user interface
3. User uploads plant leaf image
4. Application receives uploaded file
5. Preprocessing module resizes and prepares image
6. Model performs inference
7. Application formats and displays results
8. User reviews prediction and confidence scores

**Model Training Flow (Offline):**
1. Load dataset from directory structure
2. Preprocess images (resize, batch, encode labels)
3. Initialize CNN model architecture
4. Compile model with optimizer and loss function
5. Train model over multiple epochs
6. Evaluate on validation set
7. Save trained model weights
8. Save training history for analysis

## 5.3 CNN Model Architecture

### 5.3.1 Detailed Layer Specifications

**Layer 0: Input Layer**
- **Type:** InputLayer (implicit)
- **Shape:** (None, 128, 128, 3)
- **Parameters:** 0
- **Description:** Accepts RGB images of 128×128 pixels

**Layer 1: Conv2D_1**
- **Type:** Convolutional2D
- **Filters:** 32
- **Kernel Size:** 3×3
- **Padding:** Same
- **Activation:** ReLU
- **Input Shape:** (128, 128, 3)
- **Output Shape:** (128, 128, 32)
- **Parameters:** (3×3×3 + 1) × 32 = 896

**Layer 2: Conv2D_2**
- **Type:** Convolutional2D
- **Filters:** 32
- **Kernel Size:** 3×3
- **Padding:** Valid
- **Activation:** ReLU
- **Input Shape:** (128, 128, 32)
- **Output Shape:** (126, 126, 32)
- **Parameters:** (3×3×32 + 1) × 32 = 9,248

**Layer 3: MaxPooling2D_1**
- **Type:** MaxPooling2D
- **Pool Size:** 2×2
- **Strides:** 2
- **Input Shape:** (126, 126, 32)
- **Output Shape:** (63, 63, 32)
- **Parameters:** 0

**Layer 4-6: Convolutional Block 2**
- Similar structure with 64 filters
- Output Shape: (30, 30, 64)
- Parameters: ~37,000

**Layer 7-9: Convolutional Block 3**
- Similar structure with 128 filters
- Output Shape: (14, 14, 128)
- Parameters: ~148,000

**Layer 10-12: Convolutional Block 4**
- Similar structure with 256 filters
- Output Shape: (6, 6, 256)
- Parameters: ~590,000

**Layer 13-15: Convolutional Block 5**
- Similar structure with 512 filters
- Output Shape: (2, 2, 512)
- Parameters: ~2,360,000

**Layer 16: Dropout_1**
- **Type:** Dropout
- **Rate:** 0.25
- **Parameters:** 0

**Layer 17: Flatten**
- **Type:** Flatten
- **Input Shape:** (2, 2, 512)
- **Output Shape:** (2048,)
- **Parameters:** 0

**Layer 18: Dense_1**
- **Type:** Dense (Fully Connected)
- **Units:** 1,500
- **Activation:** ReLU
- **Input Shape:** (2048,)
- **Output Shape:** (1500,)
- **Parameters:** 2048×1500 + 1500 = 3,073,500

**Layer 19: Dropout_2**
- **Type:** Dropout
- **Rate:** 0.40
- **Parameters:** 0

**Layer 20: Dense_2 (Output)**
- **Type:** Dense (Fully Connected)
- **Units:** 38
- **Activation:** Softmax
- **Input Shape:** (1500,)
- **Output Shape:** (38,)
- **Parameters:** 1500×38 + 38 = 57,038

**Total Parameters:** Approximately 6.2 million
**Trainable Parameters:** All 6.2 million
**Non-trainable Parameters:** 0

### 5.3.2 Feature Map Dimensions

The progressive reduction in spatial dimensions coupled with increasing feature depth characterizes the architecture:

| Layer Block | Dimensions | Feature Maps | Receptive Field |
|-------------|------------|--------------|-----------------|
| Input | 128×128 | 3 | 1×1 |
| Block 1 | 63×63 | 32 | ~5×5 |
| Block 2 | 30×30 | 64 | ~13×13 |
| Block 3 | 14×14 | 128 | ~29×29 |
| Block 4 | 6×6 | 256 | ~61×61 |
| Block 5 | 2×2 | 512 | ~125×125 |
| Flattened | 2048 | - | Entire image |

### 5.3.3 Parameter Distribution

The parameter count concentrates heavily in the fully connected layers:

- **Convolutional Layers:** ~3.1 million parameters (50%)
- **Dense Layer 1:** ~3.07 million parameters (49.5%)
- **Output Layer:** ~57,000 parameters (0.9%)

This distribution reflects the typical CNN pattern where convolutional layers extract features efficiently with relatively few parameters through weight sharing, while fully connected layers require many parameters to combine these features for classification.

## 5.4 Web Application Architecture

### 5.4.1 Application Structure

```
app.py (Main Application File)
│
├── Configuration & Setup
│   ├── Page configuration (title, icon, layout)
│   ├── Custom CSS styling
│   └── Import statements
│
├── Resource Loading Functions
│   ├── load_model() - Loads trained Keras model
│   └── load_training_history() - Loads training metrics
│
├── Utility Functions
│   ├── predict_disease() - Performs inference
│   ├── format_disease_name() - Formats output
│   └── CLASS_NAMES - Disease category list
│
├── Page Functions
│   ├── show_home_page() - Landing page
│   ├── show_detection_page() - Main prediction interface
│   ├── show_performance_page() - Model metrics
│   └── show_about_page() - Project information
│
└── Main Execution
    └── main() - Application entry point
```

### 5.4.2 User Interface Components

**Navigation Sidebar:**
- Radio button selection for page navigation
- Project information display
- Student and guide details
- Persistent across all pages

**Home Page:**
- System overview and purpose
- Key features highlight
- Supported plants list
- Quick statistics display

**Disease Detection Page:**
- File uploader widget (JPG, JPEG, PNG)
- Image display (uploaded and processed)
- Prediction results card
  - Disease name
  - Plant type
  - Confidence percentage
  - Progress bar visualization
- Recommendations based on confidence
- Top-5 predictions bar chart

**Model Performance Page:**
- Key metrics cards (accuracy, loss)
- Training/validation accuracy line charts
- Training/validation loss line charts
- Model architecture summary
- Dataset statistics
- Training configuration details

**About Page:**
- Project overview
- Academic information
- Technical specifications
- Achievements summary
- Future work proposals

### 5.4.3 State Management

Streamlit employs a reactive programming model where the entire script reruns on user interaction. The application manages state through:

**Caching (@st.cache_resource, @st.cache_data):**
- Caches loaded model to avoid reloading on every interaction
- Caches training history for performance visualization
- Improves application responsiveness

**Session State:**
- Uploaded file storage
- User selections and preferences
- Navigation state

### 5.4.4 Styling and Presentation

**Custom CSS:**
- Main header styling (green color scheme reflecting agriculture)
- Sub-header formatting
- Prediction result cards with border highlighting
- Metric cards for performance statistics
- Responsive layout adapting to screen sizes

**Color Scheme:**
- Green tones (#2E7D32, #558B2F, #4CAF50) for agricultural theme
- Complementary colors for charts and visualizations
- High contrast for readability

## 5.5 Data Flow Architecture

### 5.5.1 Training Phase Data Flow

```
Dataset Directory Structure
    ↓
TensorFlow image_dataset_from_directory
    ↓
[Batched, Shuffled, Labeled Data]
    ↓
CNN Model (Forward Pass)
    ↓
Predictions → Loss Calculation
    ↓
Backpropagation (Backward Pass)
    ↓
Gradient Computation
    ↓
Adam Optimizer (Weight Updates)
    ↓
[Repeat for all batches and epochs]
    ↓
Trained Model → Save as .keras file
Training History → Save as JSON
```

### 5.5.2 Inference Phase Data Flow

```
User Uploads Image
    ↓
Streamlit File Uploader
    ↓
PIL Image Loading
    ↓
Image Resizing (128×128)
    ↓
Convert to Array
    ↓
Add Batch Dimension
    ↓
Loaded Keras Model
    ↓
Forward Pass (Inference Only)
    ↓
Softmax Probabilities (38 classes)
    ↓
ArgMax (Predicted Class)
Max Probability (Confidence)
    ↓
Format Results
    ↓
Display to User
```

### 5.5.3 Visualization Data Flow

```
Training History JSON
    ↓
Load into Python Dictionary
    ↓
Extract Metrics Lists
    ↓
Create Pandas DataFrames
    ↓
Plotly/Matplotlib Visualization
    ↓
Render in Streamlit
    ↓
Interactive Charts
```

## 5.6 Model Storage and Serialization

### 5.6.1 Model Persistence

The trained model is saved in Keras native format (.keras extension):

**File:** `trained_plant_disease_model.keras`
**Format:** HDF5-based Keras format
**Contents:**
- Model architecture (layer types, connections, configurations)
- Trained weights (all 6.2 million parameters)
- Optimizer state (Adam momentum values)
- Compilation configuration (loss function, metrics)

**Advantages:**
- Single file contains complete model
- Platform-independent format
- Supports custom layers and configurations
- Preserves exact model state for reproducibility

### 5.6.2 Training History Persistence

Training metrics are saved in JSON format:

**File:** `training_hist.json`
**Format:** JSON (JavaScript Object Notation)
**Contents:**
```json
{
  "accuracy": [epoch1_acc, epoch2_acc, ...],
  "loss": [epoch1_loss, epoch2_loss, ...],
  "val_accuracy": [epoch1_val_acc, ...],
  "val_loss": [epoch1_val_loss, ...]
}
```

**Usage:**
- Visualization of learning curves
- Analysis of training dynamics
- Detection of overfitting
- Performance reporting

## 5.7 Security and Error Handling

### 5.7.1 Input Validation

**File Type Validation:**
- Restricts uploads to JPG, JPEG, PNG formats
- Prevents execution of malicious code through file upload
- Streamlit file_uploader enforces type restrictions

**File Size Considerations:**
- Large images resize to 128×128, limiting processing load
- Memory management through batching and efficient tensor operations

**Image Format Validation:**
- PIL library handles various image formats safely
- Error handling for corrupted or invalid images

### 5.7.2 Error Handling

**Model Loading Errors:**
```python
try:
    model = tf.keras.models.load_model('trained_plant_disease_model.keras')
except Exception as e:
    st.error(f"Error loading model: {e}")
    return None
```

**Prediction Errors:**
- Graceful handling of invalid inputs
- User-friendly error messages
- Fallback behaviors for unexpected situations

**Resource Management:**
- Proper file closure after reading
- Memory cleanup for large tensors
- Efficient caching to prevent resource exhaustion

## 5.8 Scalability and Performance Considerations

### 5.8.1 Computational Efficiency

**Model Inference Time:**
- Single image prediction: <1 second on standard hardware
- Batch processing capability for multiple images
- GPU acceleration when available (Metal, CUDA)

**Memory Footprint:**
- Model size: ~25 MB (.keras file)
- Runtime memory: ~500 MB (model + activations)
- Acceptable for standard computing environments

### 5.8.2 Scalability Strategies

**Horizontal Scaling:**
- Stateless application design enables multiple instances
- Load balancing across servers for high traffic
- Cloud deployment (AWS, Google Cloud, Azure) for auto-scaling

**Vertical Scaling:**
- GPU acceleration for faster inference
- Larger memory for batch processing
- Optimized tensor operations

**Caching Strategies:**
- Model loaded once and cached
- Repeated predictions on same image served from cache
- Training history cached for visualization

### 5.8.3 Deployment Options

**Local Deployment:**
- Current implementation: `streamlit run app.py`
- Suitable for development, testing, demonstration
- Accessible on localhost or local network

**Cloud Deployment:**
- **Streamlit Cloud:** Native deployment platform
- **Heroku:** Container-based deployment
- **AWS/GCP/Azure:** VM or container services
- **Docker:** Containerization for consistent deployment

**Edge Deployment:**
- TensorFlow Lite conversion for mobile/edge devices
- Model quantization for reduced size
- Offline operation capability

## 5.9 Integration Points and APIs

### 5.9.1 External Dependencies

**TensorFlow/Keras:**
- Model architecture definition
- Training pipeline
- Inference engine

**Streamlit:**
- Web framework
- UI components
- Reactive programming model

**NumPy:**
- Array operations
- Numerical computations

**PIL (Pillow):**
- Image loading and manipulation
- Format conversion

**Plotly:**
- Interactive visualizations
- Training curve charts

### 5.9.2 Potential Integration Points

**Database Integration:**
- Store user upload history
- Track prediction statistics
- Collect feedback for model improvement

**Authentication System:**
- User accounts for personalized experience
- Role-based access control
- Usage tracking

**API Endpoints:**
- REST API for programmatic access
- Mobile app integration
- Third-party service integration

**Notification Systems:**
- Email alerts for disease detections
- SMS notifications for farmers
- Dashboard alerts for agricultural managers

## 5.10 Future Architecture Enhancements

### 5.10.1 Model Architecture Extensions

**Ensemble Methods:**
- Multiple models voting for predictions
- Improved accuracy through diversity
- Uncertainty quantification

**Transfer Learning Integration:**
- Pre-trained backbones (ResNet, EfficientNet)
- Fine-tuning for specific plant species
- Reduced training time

**Multi-task Learning:**
- Simultaneous disease classification and severity assessment
- Joint prediction of multiple plant attributes
- Shared feature representations

### 5.10.2 Application Architecture Extensions

**Mobile Application:**
- Native iOS/Android apps
- Camera integration for direct capture
- Offline model operation

**Real-time Monitoring:**
- Integration with IoT sensors
- Continuous greenhouse monitoring
- Automated alert systems

**Advanced Visualization:**
- Explainable AI visualizations (Class Activation Maps)
- Disease progression tracking
- Geographic disease mapping

## 5.11 Summary

This chapter presented the comprehensive system design and architecture encompassing the CNN model structure, web application components, data flow patterns, and deployment considerations. The modular architecture separates concerns between model development and user interface, facilitating independent testing, maintenance, and enhancement of each component.

The design balances simplicity with extensibility, providing a fully functional system while supporting future enhancements through well-defined interfaces and modular organization. The next chapter details the practical implementation of this architecture, documenting the development process, code organization, and technical challenges encountered during system realization.

---

**End of Chapter 5**

*Word Count: Approximately 2,900 words*

