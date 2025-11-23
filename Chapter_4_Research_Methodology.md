# CHAPTER 4: RESEARCH METHODOLOGY

## 4.1 Introduction

This chapter presents the comprehensive research methodology employed in developing the CNN-based plant disease detection system. The methodology encompasses research design, dataset acquisition and analysis, data preprocessing procedures, model architecture design, training protocols, evaluation strategies, and web application development. The systematic approach ensures reproducibility,

 validity, and reliability of research findings while addressing the stated objectives of achieving high classification accuracy and practical accessibility.

The research follows an experimental quantitative paradigm, focusing on measuring and analyzing model performance through numerical metrics. The methodology integrates established machine learning practices with domain-specific considerations for agricultural image classification, balancing theoretical rigor with practical applicability.

## 4.2 Research Design

### 4.2.1 Research Approach

This project employs a supervised learning approach within the experimental research paradigm. Supervised learning utilizes labeled training data where each input image is associated with its correct disease category label. The model learns to map images to labels by discovering patterns that discriminate between different disease types. This approach contrasts with unsupervised learning (which finds patterns without labels) and semi-supervised learning (which uses both labeled and unlabeled data).

The experimental nature of the research involves systematic manipulation of model architecture, hyperparameters, and training configurations to optimize performance. The methodology follows the scientific method: formulating hypotheses (CNN can achieve >90% accuracy), designing experiments (model architecture and training protocol), collecting data (training the model), analyzing results (evaluating performance metrics), and drawing conclusions (validating or rejecting hypotheses).

### 4.2.2 Research Philosophy

The research adheres to a positivist philosophy, emphasizing objective measurement, quantitative analysis, and empirical evidence. Performance is evaluated through well-defined numerical metrics (accuracy, precision, recall, F1-score) computed on independent validation data. The approach prioritizes reproducibility through detailed documentation of methodology, dataset characteristics, and implementation specifics.

While acknowledging that machine learning involves elements of engineering and craft (architectural design choices, hyperparameter selection), the evaluation remains grounded in objective performance assessment on standardized benchmarks, enabling comparison with existing research and validation of claimed achievements.

## 4.3 Dataset Description and Analysis

### 4.3.1 Dataset Source and Origin

The project utilizes the "New Plant Diseases Dataset" obtained from Kaggle, an extended version of the PlantVillage dataset created by Hughes and Salathé (2015). PlantVillage represents a seminal contribution to agricultural AI research, providing the first large-scale open-access repository of labeled plant disease images. The dataset was created through systematic photography of plant specimens at Penn State University's Department of Plant Pathology, with expert plant pathologists providing label annotations.

The Kaggle extended version expands the original PlantVillage dataset from approximately 54,000 images to 87,867 images across 38 disease categories, providing increased sample diversity and additional disease classes. This expansion improves model training potential by providing more examples per category and broader disease coverage.

**Dataset URL:** https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

### 4.3.2 Dataset Composition

The dataset encompasses 14 plant species commonly cultivated in various agricultural regions worldwide:

1. **Apple** (Malus domestica)
2. **Blueberry** (Vaccinium spp.)
3. **Cherry** (including sour cherry) (Prunus spp.)
4. **Corn/Maize** (Zea mays)
5. **Grape** (Vitis vinifera)
6. **Orange** (Citrus sinensis)
7. **Peach** (Prunus persica)
8. **Pepper** (Bell pepper) (Capsicum annuum)
9. **Potato** (Solanum tuberosum)
10. **Raspberry** (Rubus idaeus)
11. **Soybean** (Glycine max)
12. **Squash** (Cucurbita spp.)
13. **Strawberry** (Fragaria × ananassa)
14. **Tomato** (Solanum lycopersicum)

The 38 disease and health status categories include various disease types:

**Fungal Diseases:** Apple Scab, Apple Black Rot, Cedar Apple Rust, Cherry Powdery Mildew, Corn Cercospora Leaf Spot, Grape Black Rot, Grape Esca, Strawberry Leaf Scorch, and others

**Bacterial Diseases:** Peach Bacterial Spot, Pepper Bacterial Spot, Tomato Bacterial Spot

**Viral Diseases:** Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus

**Pest Damage:** Tomato Spider Mites

**Healthy States:** Healthy variants for each plant species provide contrast for disease identification

This diversity ensures the model learns to distinguish between various pathogen types and healthy tissue across different plant morphologies.

### 4.3.3 Dataset Statistics

**Training Set:** 70,295 images
- Used for model parameter optimization
- Undergoes shuffling each epoch for varied sample presentation
- Provides diverse examples for learning disease-specific patterns

**Validation Set:** 17,572 images  
- Used for unbiased performance evaluation during training
- Monitors generalization capability
- Guides decisions about training duration (early stopping if validation performance plateaus)

**Test Set:** 33 custom images
- Collected separately for final system testing
- Covers multiple disease types: Apple Cedar Rust, Apple Scab, Corn Common Rust, Potato Early Blight, Potato Healthy, Tomato Early Blight, Tomato Healthy, Tomato Yellow Curl Virus
- Tests system performance on independent samples

**Total Dataset Size:** 87,867 labeled images

**Class Distribution:**
The dataset exhibits relative balance across classes, with most categories containing 1,500-2,500 samples. Some categories have fewer examples (e.g., Peach Healthy with 674 samples) while others exceed 2,000 samples. This moderate imbalance reflects natural variation in disease prevalence and data collection opportunities, but remains sufficiently balanced to avoid severe biases toward majority classes.

### 4.3.4 Image Characteristics

**Original Format:** JPEG (.JPG)  
**Original Dimensions:** Variable (approximately 256×256 pixels typically)  
**Color Space:** RGB (3 channels)  
**Background:** Uniform neutral backgrounds (controlled laboratory conditions)  
**Leaf Positioning:** Centered, showing clear views of leaf surfaces  
**Lighting:** Standardized illumination minimizing shadows and glare  
**Image Quality:** High resolution with minimal compression artifacts

The controlled image conditions ensure consistent data quality, facilitating model training by reducing confounding variables. However, this controlled nature also represents a limitation regarding generalization to field conditions with complex backgrounds, variable lighting, and natural variations (discussed in Section 4.9).

## 4.4 Data Preprocessing Pipeline

### 4.4.1 Image Resizing

All images undergo standardized resizing to 128×128 pixels, balancing computational efficiency with information preservation. The choice of 128×128 represents a design decision considering multiple factors:

**Computational Efficiency:** Smaller images require less memory and fewer computations. A 128×128×3 image contains 49,152 values compared to 196,608 values for a 256×256 image, yielding 4× computational savings.

**Information Preservation:** Disease symptoms remain visible at 128×128 resolution. Key visual features (leaf discoloration, lesion patterns, structural abnormalities) survive downsampling.

**Standard Practice:** 128×128 and 224×224 represent common input sizes in computer vision research, enabling comparison with existing work.

The resizing employs bilinear interpolation, which computes output pixel values through weighted averaging of four nearest input pixels, providing smooth results without introducing edge artifacts.

### 4.4.2 Normalization

Pixel values in raw images range from 0-255 (uint8 format). While neural networks can technically process any numeric range, normalization to a standardized range accelerates training convergence and improves gradient flow. The TensorFlow image_dataset_from_directory function maintains pixel values in their original 0-255 range (uint8), with internal model operations handling any necessary scaling.

For neural network training, having inputs roughly centered around zero with similar variances across features improves optimization dynamics. However, ReLU activation functions operate effectively with positive inputs, making the 0-255 range acceptable for the employed architecture.

### 4.4.3 Color Space

The dataset maintains RGB (Red-Green-Blue) color space throughout processing. Color information provides crucial diagnostic value for plant disease detection, as many diseases manifest through color changes (yellowing, browning, reddening). Converting to grayscale would discard this valuable information, potentially degrading classification performance.

### 4.4.4 Batch Processing

Images are organized into batches of 32 samples for efficient processing. Batch processing provides several advantages:

**Computational Efficiency:** Modern GPUs process multiple samples in parallel more efficiently than sequential processing. Batch size 32 fills computational units without excessive memory demands.

**Gradient Stability:** Computing gradients over mini-batches provides more stable estimates than single-sample gradients (stochastic) while maintaining computational efficiency compared to full-batch gradients.

**Memory Management:** Batch processing enables training on datasets larger than available RAM by loading and processing subsets sequentially.

The batch size of 32 represents a compromise between computational efficiency, memory constraints, and gradient stability. Larger batches provide more accurate gradient estimates but require more memory; smaller batches enable faster iteration but yield noisier gradients.

### 4.4.5 Label Encoding

Disease categories undergo categorical (one-hot) encoding, converting class labels into binary vectors of length 38 (number of classes). For a sample from class c, the encoded label contains 1 at position c and 0 elsewhere. This encoding facilitates categorical cross-entropy loss computation and enables probabilistic interpretation of network outputs through softmax activation.

### 4.4.6 Data Augmentation

The current implementation employs minimal data augmentation beyond shuffling training samples each epoch. While sophisticated augmentation techniques (rotation, flipping, color jittering, random crops) often improve model generalization, the large training set size (70,295 samples) mitigates overfitting risk that augmentation primarily addresses.

Shuffling ensures varied sample presentation orders across epochs, preventing the model from memorizing sample sequences rather than learning genuine disease patterns. The shuffle operation reorders training samples randomly before each epoch, ensuring different batch compositions across training iterations.

## 4.5 Model Architecture Design

### 4.5.1 Architecture Overview

The model employs a custom CNN architecture designed specifically for plant disease classification. The architecture consists of five convolutional blocks with progressively increasing filter counts, followed by fully connected layers for classification. This progressive architecture enables hierarchical feature learning, with early layers detecting simple patterns and deeper layers combining these into complex disease-specific representations.

The decision to develop a custom architecture rather than employing transfer learning reflects multiple considerations:

**Dataset Size:** With 70,295 training samples, the dataset is sufficiently large to train deep networks from scratch without requiring transfer learning as a regularization technique.

**Domain Specificity:** Pre-trained models typically learn features from natural images (ImageNet dataset containing objects, animals, scenes). Plant disease detection requires specialized features (disease lesion patterns, leaf discoloration characteristics) that may differ from natural image features.

**Educational Value:** Designing a custom architecture provides deeper insights into CNN design principles, layer interactions, and architectural trade-offs compared to simply fine-tuning existing models.

**Optimization Flexibility:** Custom architectures allow tailoring layer configurations, filter sizes, and network depth specifically for the target task without constraints imposed by pre-existing structures.

### 4.5.2 Layer-by-Layer Specification

**Input Layer:**
- **Dimensions:** 128 × 128 × 3 (height, width, channels)
- **Type:** RGB color images
- **Normalization:** Pixel values 0-255 (uint8)

**Convolutional Block 1:**
- **Conv2D Layer 1:** 32 filters, 3×3 kernel, same padding, ReLU activation
  - Output: 128 × 128 × 32
- **Conv2D Layer 2:** 32 filters, 3×3 kernel, valid padding, ReLU activation
  - Output: 126 × 126 × 32
- **MaxPooling2D:** 2×2 pool size, stride 2
  - Output: 63 × 63 × 32

**Convolutional Block 2:**
- **Conv2D Layer 1:** 64 filters, 3×3 kernel, same padding, ReLU activation
  - Output: 63 × 63 × 64
- **Conv2D Layer 2:** 64 filters, 3×3 kernel, valid padding, ReLU activation
  - Output: 61 × 61 × 64
- **MaxPooling2D:** 2×2 pool size, stride 2
  - Output: 30 × 30 × 64

**Convolutional Block 3:**
- **Conv2D Layer 1:** 128 filters, 3×3 kernel, same padding, ReLU activation
  - Output: 30 × 30 × 128
- **Conv2D Layer 2:** 128 filters, 3×3 kernel, valid padding, ReLU activation
  - Output: 28 × 28 × 128
- **MaxPooling2D:** 2×2 pool size, stride 2
  - Output: 14 × 14 × 128

**Convolutional Block 4:**
- **Conv2D Layer 1:** 256 filters, 3×3 kernel, same padding, ReLU activation
  - Output: 14 × 14 × 256
- **Conv2D Layer 2:** 256 filters, 3×3 kernel, valid padding, ReLU activation
  - Output: 12 × 12 × 256
- **MaxPooling2D:** 2×2 pool size, stride 2
  - Output: 6 × 6 × 256

**Convolutional Block 5:**
- **Conv2D Layer 1:** 512 filters, 3×3 kernel, same padding, ReLU activation
  - Output: 6 × 6 × 512
- **Conv2D Layer 2:** 512 filters, 3×3 kernel, valid padding, ReLU activation
  - Output: 4 × 4 × 512
- **MaxPooling2D:** 2×2 pool size, stride 2
  - Output: 2 × 2 × 512

**Regularization:**
- **Dropout:** 25% dropout rate
  - Randomly drops 25% of activations during training

**Flatten Layer:**
- Converts 2 × 2 × 512 tensor to 1D vector of length 2,048

**Fully Connected Layers:**
- **Dense Layer:** 1,500 neurons, ReLU activation
  - Large capacity for combining learned features
- **Dropout:** 40% dropout rate
  - Stronger regularization for highly-connected layer
- **Output Layer:** 38 neurons, Softmax activation
  - Produces probability distribution over disease classes

### 4.5.3 Design Rationale

**Progressive Filter Increase (32→64→128→256→512):**  
Increasing filter counts with depth enables the network to learn progressively more complex features. Early layers need fewer filters to capture simple patterns (edges, colors), while deeper layers require more filters to represent numerous complex pattern combinations.

**Alternating Padding (same/valid):**  
Same padding preserves spatial dimensions, enabling precise position information retention. Valid padding reduces dimensions, encouraging the network to learn more abstract, position-invariant representations. Alternating these approaches balances spatial precision with abstraction.

**Dual Dropout Configuration:**  
The 25% dropout after convolutional blocks provides moderate regularization without excessive capacity reduction. The stronger 40% dropout after the dense layer addresses the higher overfitting risk from the large number of parameters (2,048 × 1,500 = 3,072,000 weights) in this fully connected layer.

**Large Dense Layer (1,500 neurons):**  
The substantial dense layer capacity enables complex combinations of learned features for discriminating among 38 classes. With 38 diverse disease categories, significant representational capacity is necessary to capture subtle distinctions between visually similar diseases.

## 4.6 Training Methodology

### 4.6.1 Training Configuration

**Optimization Algorithm:** Adam (Adaptive Moment Estimation)
- **Learning Rate:** 0.0001 (reduced from default 0.001)
- **Beta 1:** 0.9 (exponential decay rate for first moment estimates)
- **Beta 2:** 0.999 (exponential decay rate for second moment estimates)
- **Epsilon:** 1e-07 (small constant for numerical stability)

The reduced learning rate promotes stable convergence for the complex 38-class problem. Higher learning rates risk overshooting optimal parameter values, while the selected rate enables gradual, steady improvement.

**Loss Function:** Categorical Cross-Entropy
- Appropriate for multi-class classification
- Heavily penalizes confident incorrect predictions
- Encourages probability mass concentration on correct class

**Metrics:** Accuracy (proportion of correct predictions)

**Batch Size:** 32 samples per batch

**Epochs:** 10 complete passes through training dataset

### 4.6.2 Hardware and Software Environment

**Hardware Specifications:**
- **Processor:** Apple M4 Pro chip
- **GPU:** Apple Metal GPU (integrated graphics)
- **System Memory:** 48 GB RAM
- **Storage:** SSD (solid-state drive)

**Software Environment:**
- **Operating System:** macOS 25.0.0
- **Python Version:** 3.12
- **Environment Manager:** Mamba (conda alternative)
- **Deep Learning Framework:** TensorFlow 2.16.2, Keras 3.12.0
- **Hardware Acceleration:** TensorFlow Metal (GPU acceleration for Apple Silicon)
- **Development Environment:** Jupyter Lab 4.5.0

**Supporting Libraries:**
- NumPy 1.26.4 (numerical computing)
- Matplotlib 3.10.7 (visualization)
- Pandas 2.3.3 (data manipulation)
- Scikit-learn 1.7.2 (evaluation metrics)
- Seaborn 0.13.2 (advanced visualization)

### 4.6.3 Training Process

Training proceeds through iterative epochs, each consisting of:

1. **Forward Pass:** Batch images propagate through the network, producing predictions
2. **Loss Computation:** Categorical cross-entropy compares predictions to true labels
3. **Backward Pass:** Backpropagation computes gradients of loss with respect to all weights
4. **Parameter Update:** Adam optimizer adjusts weights using computed gradients
5. **Validation Evaluation:** After each epoch, the model evaluates on the validation set without gradient updates

The process continues for 10 epochs, monitoring both training and validation metrics to assess learning progress and detect potential overfitting.

## 4.7 Evaluation Strategy

### 4.7.1 Primary Metrics

**Training Accuracy:**  
Measured on training data after each epoch. Indicates how well the model fits the training distribution. Expected to increase across epochs.

**Validation Accuracy:**  
Measured on independent validation data after each epoch. Provides unbiased estimate of generalization performance. Primary metric for assessing model quality.

**Training Loss:**  
Categorical cross-entropy on training data. Should decrease across epochs as the model learns.

**Validation Loss:**  
Categorical cross-entropy on validation data. Should decrease initially; increasing validation loss despite decreasing training loss signals overfitting.

### 4.7.2 Detailed Performance Analysis

**Confusion Matrix:**  
A 38×38 matrix showing prediction distributions across all class pairs. Diagonal elements represent correct classifications; off-diagonal elements reveal specific confusion patterns.

**Per-Class Metrics:**
- **Precision:** Proportion of positive predictions that were correct (for each class)
- **Recall:** Proportion of actual positives correctly identified (for each class)
- **F1-Score:** Harmonic mean of precision and recall (for each class)
- **Support:** Number of samples in each class

**Classification Report:**  
Comprehensive summary of precision, recall, F1-score, and support for all 38 classes plus macro and weighted averages.

### 4.7.3 Validation Approach

The fixed train-validation split provided with the dataset ensures consistent evaluation across all experiments. This approach differs from k-fold cross-validation, which would require multiple training runs. The large validation set size (17,572 samples) provides stable performance estimates with low variance.

The validation set remains completely separate from training, serving as a proxy for real-world performance on unseen data. No gradient updates occur based on validation data, maintaining its role as an unbiased performance estimator.

## 4.8 Web Application Development

### 4.8.1 Framework Selection

The web application employs Streamlit, a modern Python framework designed specifically for creating interactive data science applications. Streamlit selection reflects multiple advantages:

**Rapid Development:** Streamlit abstracts web development complexity, enabling pure Python implementation without HTML, CSS, or JavaScript
**Native Machine Learning Support:** Seamless integration with TensorFlow, NumPy, and other scientific Python libraries
**Reactive Architecture:** Automatic UI updates when state changes
**Deployment Flexibility:** Local deployment for testing, cloud deployment for production

### 4.8.2 Application Architecture

The application follows a modular structure:

**Model Loading Module:** Loads trained Keras model and training history
**Image Processing Module:** Handles uploaded images, preprocessing, inference
**Visualization Module:** Creates performance charts, displays results
**User Interface Module:** Defines page layout, navigation, user interactions

### 4.8.3 User Interface Design

The interface prioritizes intuitive operation for non-technical users:

**Home Page:** Provides system overview, capabilities, supported plants
**Disease Detection Page:** Upload interface, prediction display, confidence scores
**Model Performance Page:** Training history visualization, architecture details
**About Page:** Project information, technical specifications, contact details

### 4.8.4 Implementation Features

**Image Upload:** File upload widget accepting JPG, JPEG, PNG formats
**Real-time Prediction:** Immediate inference upon image upload
**Confidence Display:** Prediction probability as percentage
**Top-5 Predictions:** Shows alternative diagnoses with probabilities
**Interactive Visualizations:** Training curves using Plotly for interactivity
**Responsive Design:** Adapts to different screen sizes

## 4.9 Research Limitations and Validity Considerations

### 4.9.1 Dataset Limitations

**Controlled Conditions:** All images feature uniform backgrounds and standardized lighting. Real field images exhibit complex backgrounds, variable lighting, occlusions, and natural variations potentially degrading model performance.

**Geographic Constraints:** Images primarily originate from specific regions. Disease presentation may vary across geographical locations due to different pathogen strains, climate conditions, and plant varieties.

**Limited Disease Coverage:** 38 categories, while comprehensive, represent a subset of all possible plant diseases. Many regional or emerging diseases lack representation.

### 4.9.2 Methodological Limitations

**Single Training Run:** The reported results reflect a single training execution. Multiple runs with different random initializations would provide performance variance estimates.

**Limited Augmentation:** Minimal data augmentation may reduce robustness to image variations not present in the training set.

**Fixed Architecture:** While the custom architecture achieves strong performance, extensive architectural search across numerous configurations might identify superior designs.

### 4.9.3 Validity Measures

**Internal Validity:** Large independent validation set and detailed per-class analysis ensure reliable performance assessment on the specific dataset used.

**External Validity:** Generalization to real field conditions requires additional validation with field-captured images, representing future work beyond current scope.

**Reproducibility:** Detailed documentation of dataset, architecture, hyperparameters, and training procedures enables reproduction of results by other researchers.

## 4.10 Summary

This chapter presented the comprehensive research methodology employed in developing the CNN-based plant disease detection system. The systematic approach encompasses dataset analysis, preprocessing pipeline design, custom architecture development, rigorous training protocol, thorough evaluation strategy, and accessible web application implementation. The methodology balances theoretical soundness with practical considerations, ensuring both academic rigor and real-world applicability.

The next chapter details the system design and architecture, providing technical specifications for all components and their interactions within the complete end-to-end solution.

---

**End of Chapter 4**

*Word Count: Approximately 4,100 words*

