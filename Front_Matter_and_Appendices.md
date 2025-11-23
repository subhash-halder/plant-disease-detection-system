# FRONT MATTER

## Title Page

**PLANT DISEASE DETECTION USING CONVOLUTIONAL NEURAL NETWORKS**

A Project Report  
Submitted in Partial Fulfillment of the Requirements  
for the Degree of  
**MASTER OF COMPUTER APPLICATIONS**  
(Machine Learning Specialization)

by

**SUBHASH HALDER**  
Enrollment Number: A9929724000690(el)

Under the Guidance of  
**Mr. Ayan Pal**  
M.Tech (Computer Science)  
Senior Engineering Manager, Walmart  
(15 Years of Experience)

**AMITY UNIVERSITY ONLINE**  
Semester: 4  
Year: 2025

---

## Project Guide Certificate

**CERTIFICATE**

This is to certify that the project work entitled **"Plant Disease Detection Using Convolutional Neural Networks"** is a bonafide work carried out by **Mr. Subhash Halder**, Enrollment Number: **A9929724000690(el)**, in partial fulfillment of the requirements for the award of the degree of **Master of Computer Applications (Machine Learning Specialization)** from Amity University Online during the academic year 2025.

The project has been carried out under my supervision and guidance. The work is original and has been completed satisfactorily.

I recommend this project report for evaluation.

**Project Guide:**

**Mr. Ayan Pal**  
M.Tech (Computer Science)  
Senior Engineering Manager  
Walmart

Signature: ___________________________

Date: ___________________________

Place: ___________________________

---

## Student Declaration

**DECLARATION**

I, **Subhash Halder**, student of Master of Computer Applications (Machine Learning Specialization) at Amity University Online, Enrollment Number **A9929724000690(el)**, hereby declare that the project work entitled **"Plant Disease Detection Using Convolutional Neural Networks"** submitted by me is my original work.

I further declare that:

1. The work presented in this project report is authentic and original.

2. This project work has not been submitted earlier for any degree or diploma to any other university or institution.

3. All the information and data provided in this report is true to the best of my knowledge.

4. I have followed all ethical guidelines while conducting this research and preparing this report.

5. All sources of information and assistance received during the course of this investigation have been duly acknowledged.

I understand that any violation of the above statements may result in the cancellation of my degree and other appropriate disciplinary action.

**Student Name: Subhash Halder**  
Enrollment Number: A9929724000690(el)  
Program: MCA (Machine Learning)  
Semester: 4, Year: 2025

Signature: ___________________________

Date: ___________________________

Place: ___________________________

---

## Acknowledgments

**ACKNOWLEDGMENTS**

The successful completion of this project would not have been possible without the support, guidance, and encouragement of several individuals to whom I extend my deepest gratitude.

First and foremost, I express my sincere thanks to my project guide, **Mr. Ayan Pal**, Senior Engineering Manager at Walmart, for his invaluable guidance, continuous support, and expert advice throughout this project. His extensive experience in the field of computer science and his insightful feedback significantly enhanced the quality of this work. His encouragement and constructive criticism motivated me to strive for excellence at every stage of this project.

I am profoundly grateful to **Amity University Online** for providing me with the opportunity to pursue this Master's program in Computer Applications with specialization in Machine Learning. The knowledge and skills acquired during this program have been instrumental in successfully completing this project.

I would like to thank the faculty members of the MCA program for their excellent teaching and for laying a strong foundation in computer science and machine learning concepts that proved essential for this research.

My heartfelt appreciation goes to the creators and maintainers of the **PlantVillage dataset** and the **Kaggle community** for making high-quality agricultural image data freely available for research purposes. This project would not have been feasible without access to such comprehensive datasets.

I acknowledge the developers of **TensorFlow**, **Keras**, **Streamlit**, and other open-source libraries used in this project. The robust tools and frameworks they have created enable researchers and developers worldwide to work on cutting-edge artificial intelligence applications.

I am thankful to my family for their unwavering support, patience, and encouragement throughout my academic journey. Their belief in my abilities and their sacrifices have been my source of strength and motivation.

I also extend my thanks to my friends and colleagues who provided valuable suggestions, engaged in stimulating discussions, and offered moral support during the challenging phases of this project.

Finally, I am grateful to all those who, directly or indirectly, contributed to the successful completion of this project. Any omissions in these acknowledgments are unintentional and regretted.

**Subhash Halder**

---

## Abstract

**ABSTRACT**

Plant diseases represent a critical threat to global agricultural productivity, causing substantial crop losses and threatening food security worldwide. Traditional disease detection methods rely on manual inspection by experts, which is time-consuming, expensive, and often unavailable in resource-constrained agricultural regions. This project addresses these challenges by developing an automated Plant Disease Detection System leveraging deep learning techniques, specifically Convolutional Neural Networks (CNNs), to provide accurate, rapid, and accessible disease diagnosis from digital images of plant leaves.

The primary objective of this research was to design and implement a custom CNN architecture capable of classifying plant diseases across multiple species with accuracy exceeding 90%. The system encompasses 38 disease categories spanning 14 plant species including Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato. The research employed the Kaggle New Plant Diseases Dataset, comprising 87,867 images (70,295 training samples and 17,572 validation samples), providing comprehensive coverage of various fungal, bacterial, and viral diseases.

The methodology followed a systematic experimental approach, developing a custom five-layer CNN architecture with progressively increasing filter counts (32, 64, 128, 256, 512) to enable hierarchical feature learning. The model incorporates strategic dropout regularization (25% after convolutional blocks, 40% after dense layer) to prevent overfitting. Training utilized the Adam optimizer with a reduced learning rate of 0.0001, categorical cross-entropy loss function, and batch size of 32 over 10 epochs. The implementation leveraged TensorFlow and Keras frameworks with Apple Metal GPU acceleration for efficient training.

The trained model achieved exceptional performance, with training accuracy of 98.35% and validation accuracy of 95.57%, significantly exceeding the hypothesized 90% threshold. The minimal gap between training and validation accuracy (2.78%) demonstrates effective generalization without overfitting. Detailed per-class analysis revealed balanced performance across disease categories, with most classes achieving individual accuracy exceeding 90%. The confusion matrix analysis showed strong diagonal patterns indicating correct classifications, with minimal confusion primarily occurring between visually similar diseases within the same plant species.

To enhance practical applicability, the project developed an intuitive web-based application using the Streamlit framework. The application provides a user-friendly interface enabling agricultural professionals without technical expertise to upload plant leaf images and receive immediate disease predictions with confidence scores. The interface displays the top five probable diagnoses, provides interactive visualization of model training history, and presents comprehensive performance metrics. The responsive design adapts to various devices, ensuring accessibility across desktop computers, tablets, and smartphones.

The significance of this work extends across multiple dimensions. Academically, the project contributes to agricultural AI literature by demonstrating that custom CNN architectures trained from scratch can achieve competitive performance without requiring transfer learning, given adequate training data. The comprehensive documentation of methodology, architecture design, and performance evaluation supports reproducibility and provides guidance for future research. Practically, the accessible web application democratizes advanced AI diagnostics, making sophisticated disease detection capabilities available to farmers and extension workers in diverse agricultural settings. Economically, early disease detection enabled by the system can prevent substantial crop losses, reduce unnecessary pesticide applications, and support informed agricultural decision-making.

The project acknowledges several limitations, primarily the controlled nature of training images which may not fully represent field conditions with complex backgrounds and variable lighting. Additionally, the current implementation focuses solely on disease classification without assessing severity levels or providing treatment recommendations. These limitations provide clear directions for future work.

Future enhancements include expanding coverage to additional plant species and diseases, implementing disease severity assessment capabilities, developing native mobile applications for field-based diagnosis with offline operation, incorporating treatment recommendation systems, integrating with IoT sensors for continuous monitoring, and conducting extensive field validation across diverse geographic regions and agricultural practices.

In conclusion, this project successfully developed a high-accuracy, accessible plant disease detection system that bridges the gap between advanced deep learning research and practical agricultural application. The achieved validation accuracy of 95.57% across 38 disease categories validates the effectiveness of CNN-based approaches for automated plant disease diagnosis. The combination of robust model performance with user-friendly interface design positions the system for real-world deployment, potentially contributing to improved crop management, reduced agricultural losses, and enhanced global food security.

**Keywords:** Plant Disease Detection, Convolutional Neural Networks, Deep Learning, Image Classification, Agricultural AI, Machine Learning, TensorFlow, Keras, Precision Agriculture, Computer Vision

---

## Table of Contents

**TABLE OF CONTENTS**

| Chapter | Title | Page |
|---------|-------|------|
| | **FRONT MATTER** | |
| | Title Page | i |
| | Project Guide Certificate | ii |
| | Student Declaration | iii |
| | Acknowledgments | iv |
| | Abstract | v |
| | Table of Contents | vii |
| | List of Figures | ix |
| | List of Tables | xi |
| | | |
| | **MAIN CHAPTERS** | |
| 1 | **INTRODUCTION** | 1 |
| 1.1 | Background | 1 |
| 1.2 | Problem Statement | 4 |
| 1.3 | Research Objectives | 5 |
| 1.4 | Scope of the Project | 6 |
| 1.5 | Significance of the Study | 7 |
| 1.6 | Research Methodology Overview | 9 |
| 1.7 | Organization of the Report | 10 |
| | | |
| 2 | **LITERATURE REVIEW** | 12 |
| 2.1 | Introduction | 12 |
| 2.2 | Foundational Deep Learning and CNN Architectures | 13 |
| 2.3 | Plant Disease Detection Using Machine Learning | 18 |
| 2.4 | Recent Advances in Multi-Crop Disease Detection | 22 |
| 2.5 | Image Processing and Data Augmentation | 24 |
| 2.6 | Optimization and Training Methodologies | 25 |
| 2.7 | Agricultural AI and Precision Agriculture Context | 27 |
| 2.8 | Datasets and Benchmarks | 28 |
| 2.9 | Research Gaps and Opportunities | 29 |
| 2.10 | Summary | 30 |
| | | |
| 3 | **THEORETICAL FRAMEWORK** | 32 |
| 3.1 | Introduction | 32 |
| 3.2 | Artificial Neural Networks Fundamentals | 33 |
| 3.3 | Convolutional Neural Networks | 36 |
| 3.4 | Regularization Techniques | 42 |
| 3.5 | Loss Functions and Optimization | 44 |
| 3.6 | Model Evaluation Metrics | 46 |
| 3.7 | Progressive Feature Learning | 48 |
| 3.8 | Summary | 49 |
| | | |
| 4 | **RESEARCH METHODOLOGY** | 51 |
| 4.1 | Introduction | 51 |
| 4.2 | Research Design | 52 |
| 4.3 | Dataset Description and Analysis | 53 |
| 4.4 | Data Preprocessing Pipeline | 57 |
| 4.5 | Model Architecture Design | 59 |
| 4.6 | Training Methodology | 63 |
| 4.7 | Evaluation Strategy | 65 |
| 4.8 | Web Application Development | 67 |
| 4.9 | Research Limitations and Validity Considerations | 69 |
| 4.10 | Summary | 70 |
| | | |
| 5 | **SYSTEM DESIGN AND ARCHITECTURE** | 72 |
| 5.1 | Introduction | 72 |
| 5.2 | System Architecture Overview | 73 |
| 5.3 | CNN Model Architecture | 75 |
| 5.4 | Web Application Architecture | 79 |
| 5.5 | Data Flow Architecture | 82 |
| 5.6 | Model Storage and Serialization | 84 |
| 5.7 | Security and Error Handling | 85 |
| 5.8 | Scalability and Performance Considerations | 86 |
| 5.9 | Integration Points and APIs | 88 |
| 5.10 | Future Architecture Enhancements | 89 |
| 5.11 | Summary | 91 |
| | | |
| 6 | **IMPLEMENTATION** | 93 |
| 6.1 | Introduction | 93 |
| 6.2 | Development Environment Setup | 94 |
| 6.3 | Model Implementation | 95 |
| 6.4 | Model Evaluation Implementation | 98 |
| 6.5 | Web Application Implementation | 99 |
| 6.6 | Testing and Validation | 101 |
| 6.7 | Implementation Challenges and Solutions | 102 |
| 6.8 | Code Organization and Documentation | 103 |
| 6.9 | Summary | 104 |
| | | |
| 7 | **RESULTS AND ANALYSIS** | 106 |
| 7.1 | Introduction | 106 |
| 7.2 | Training Results | 107 |
| 7.3 | Final Model Performance | 110 |
| 7.4 | Confusion Matrix Analysis | 111 |
| 7.5 | Per-Class Performance Metrics | 113 |
| 7.6 | Test Set Predictions | 115 |
| 7.7 | Visualization of Results | 117 |
| 7.8 | Computational Performance | 118 |
| 7.9 | Web Application Demonstration | 119 |
| 7.10 | Comparative Analysis | 120 |
| 7.11 | Summary | 121 |
| | | |
| 8 | **DISCUSSION** | 123 |
| 8.1 | Introduction | 123 |
| 8.2 | Achievement of Research Objectives | 124 |
| 8.3 | Interpretation of Model Performance | 126 |
| 8.4 | Comparison with Existing Approaches | 128 |
| 8.5 | Strengths of the Proposed System | 130 |
| 8.6 | Limitations and Constraints | 132 |
| 8.7 | Real-World Applicability | 134 |
| 8.8 | Economic and Social Implications | 136 |
| 8.9 | Lessons Learned | 138 |
| 8.10 | Summary | 139 |
| | | |
| 9 | **CONCLUSION AND FUTURE WORK** | 141 |
| 9.1 | Summary of the Project | 141 |
| 9.2 | Key Achievements | 142 |
| 9.3 | Research Objectives Fulfillment | 144 |
| 9.4 | Hypothesis Validation | 145 |
| 9.5 | Significance and Impact | 146 |
| 9.6 | Limitations Acknowledged | 147 |
| 9.7 | Future Work and Enhancements | 148 |
| 9.8 | Recommendations | 152 |
| 9.9 | Final Remarks | 153 |
| 9.10 | Closing Statement | 154 |
| | | |
| | **BACK MATTER** | |
| | References | 156 |
| | Appendix A: Model Architecture Code | 162 |
| | Appendix B: Web Application Code | 165 |
| | Appendix C: Training Logs | 170 |
| | Appendix D: Classification Report | 172 |
| | Appendix E: Sample Predictions | 175 |

---

## List of Figures

**LIST OF FIGURES**

| Figure No. | Title | Page |
|------------|-------|------|
| 1.1 | Impact of Plant Diseases on Global Agricultural Production | 2 |
| 2.1 | Evolution of CNN Architectures from AlexNet to ResNet | 15 |
| 2.2 | Performance Comparison of Different CNN Architectures | 21 |
| 3.1 | Biological Neuron vs. Artificial Neuron | 33 |
| 3.2 | Multi-Layer Perceptron Architecture | 35 |
| 3.3 | Convolution Operation Visualization | 38 |
| 3.4 | Max Pooling Operation | 40 |
| 3.5 | Activation Functions (ReLU, Sigmoid, Softmax) | 41 |
| 3.6 | Dropout Regularization Mechanism | 43 |
| 4.1 | Dataset Distribution Across Plant Species | 55 |
| 4.2 | Sample Images from Each Disease Category | 56 |
| 4.3 | Data Preprocessing Pipeline Flowchart | 58 |
| 4.4 | CNN Model Architecture Diagram | 62 |
| 5.1 | System Architecture Overview | 74 |
| 5.2 | Detailed CNN Architecture with Dimensions | 77 |
| 5.3 | Feature Map Dimensions Through Network Layers | 78 |
| 5.4 | Web Application Architecture | 80 |
| 5.5 | User Interface Component Diagram | 81 |
| 5.6 | Training Phase Data Flow Diagram | 83 |
| 5.7 | Inference Phase Data Flow Diagram | 84 |
| 6.1 | Development Environment Setup Process | 94 |
| 6.2 | Model Training Execution Screenshot | 97 |
| 6.3 | Web Application Homepage Screenshot | 100 |
| 6.4 | Disease Detection Page Screenshot | 101 |
| 7.1 | Training and Validation Accuracy Curves | 108 |
| 7.2 | Training and Validation Loss Curves | 109 |
| 7.3 | Confusion Matrix Heatmap (38×38) | 112 |
| 7.4 | Per-Class Accuracy Bar Chart | 114 |
| 7.5 | Top-5 Predictions Visualization | 116 |
| 7.6 | Sample Predictions with Confidence Scores | 117 |
| 7.7 | Model Performance Metrics Dashboard | 119 |
| 8.1 | Comparison of Model Performance with Literature | 129 |
| 8.2 | Performance Distribution Across Disease Categories | 131 |
| 9.1 | Research Objectives Achievement Summary | 144 |
| 9.2 | Proposed System Architecture Enhancements | 150 |

---

## List of Tables

**LIST OF TABLES**

| Table No. | Title | Page |
|-----------|-------|------|
| 1.1 | Plant Species Covered in the Study | 7 |
| 2.1 | Summary of Literature Review | 31 |
| 3.1 | Comparison of Activation Functions | 42 |
| 3.2 | Model Evaluation Metrics Definitions | 47 |
| 4.1 | Dataset Composition by Plant Species | 54 |
| 4.2 | Dataset Statistics Summary | 55 |
| 4.3 | CNN Architecture Layer Specifications | 61 |
| 4.4 | Training Configuration Parameters | 64 |
| 4.5 | Hardware and Software Specifications | 65 |
| 5.1 | Parameter Distribution Across Layers | 76 |
| 5.2 | Feature Map Dimensions Table | 78 |
| 6.1 | Python Environment Dependencies | 95 |
| 6.2 | Implementation Challenges and Solutions | 103 |
| 7.1 | Epoch-by-Epoch Training Results | 107 |
| 7.2 | Final Model Performance Metrics | 110 |
| 7.3 | Per-Class Performance Summary | 113 |
| 7.4 | Test Set Prediction Results | 116 |
| 7.5 | Computational Performance Metrics | 118 |
| 7.6 | Comparison with Existing Studies | 120 |
| 8.1 | Research Objectives Status | 125 |
| 8.2 | Strengths and Limitations Summary | 133 |
| 9.1 | Key Achievements Summary | 143 |
| 9.2 | Proposed Future Enhancements | 151 |

---

# APPENDICES

## Appendix A: Complete Model Architecture Summary

```
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 128, 128, 32)      896       
conv2d_1 (Conv2D)           (None, 126, 126, 32)      9248      
max_pooling2d (MaxPooling2D)(None, 63, 63, 32)        0         
conv2d_2 (Conv2D)           (None, 63, 63, 64)        18496     
conv2d_3 (Conv2D)           (None, 61, 61, 64)        36928     
max_pooling2d_1 (MaxPooling2D)(None, 30, 30, 64)      0         
conv2d_4 (Conv2D)           (None, 30, 30, 128)       73856     
conv2d_5 (Conv2D)           (None, 28, 28, 128)       147584    
max_pooling2d_2 (MaxPooling2D)(None, 14, 14, 128)     0         
conv2d_6 (Conv2D)           (None, 14, 14, 256)       295168    
conv2d_7 (Conv2D)           (None, 12, 12, 256)       590080    
max_pooling2d_3 (MaxPooling2D)(None, 6, 6, 256)       0         
conv2d_8 (Conv2D)           (None, 6, 6, 512)         1180160   
conv2d_9 (Conv2D)           (None, 4, 4, 512)         2359808   
max_pooling2d_4 (MaxPooling2D)(None, 2, 2, 512)       0         
dropout (Dropout)           (None, 2, 2, 512)         0         
flatten (Flatten)           (None, 2048)              0         
dense (Dense)               (None, 1500)              3073500   
dropout_1 (Dropout)         (None, 1500)              0         
dense_1 (Dense)             (None, 38)                57038     
=================================================================
Total params: 6,192,762
Trainable params: 6,192,762
Non-trainable params: 0
_________________________________________________________________
```

## Appendix B: Disease Category List

Complete list of 38 disease categories:

1. Apple___Apple_scab
2. Apple___Black_rot
3. Apple___Cedar_apple_rust
4. Apple___healthy
5. Blueberry___healthy
6. Cherry_(including_sour)___Powdery_mildew
7. Cherry_(including_sour)___healthy
8. Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
9. Corn_(maize)___Common_rust_
10. Corn_(maize)___Northern_Leaf_Blight
11. Corn_(maize)___healthy
12. Grape___Black_rot
13. Grape___Esca_(Black_Measles)
14. Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
15. Grape___healthy
16. Orange___Haunglongbing_(Citrus_greening)
17. Peach___Bacterial_spot
18. Peach___healthy
19. Pepper,_bell___Bacterial_spot
20. Pepper,_bell___healthy
21. Potato___Early_blight
22. Potato___Late_blight
23. Potato___healthy
24. Raspberry___healthy
25. Soybean___healthy
26. Squash___Powdery_mildew
27. Strawberry___Leaf_scorch
28. Strawberry___healthy
29. Tomato___Bacterial_spot
30. Tomato___Early_blight
31. Tomato___Late_blight
32. Tomato___Leaf_Mold
33. Tomato___Septoria_leaf_spot
34. Tomato___Spider_mites Two-spotted_spider_mite
35. Tomato___Target_Spot
36. Tomato___Tomato_Yellow_Leaf_Curl_Virus
37. Tomato___Tomato_mosaic_virus
38. Tomato___healthy

## Appendix C: Web Application Usage Instructions

### Running the Application Locally

1. **Install Dependencies:**
```bash
python -m pip install -r requirement.txt
```

2. **Launch Streamlit App:**
```bash
streamlit run app.py
```

3. **Access Application:**
- Open web browser
- Navigate to `http://localhost:8501`

### Using the Disease Detection Feature

1. Navigate to "Disease Detection" page from sidebar
2. Click "Browse files" button
3. Select a plant leaf image (JPG, JPEG, or PNG format)
4. View prediction results including:
   - Identified disease name
   - Plant species
   - Confidence percentage
   - Top-5 alternative predictions

### Viewing Model Performance

1. Navigate to "Model Performance" page
2. Review training/validation accuracy curves
3. Examine loss curves
4. View model architecture summary
5. Check dataset statistics

## Appendix D: Installation Guide for Developers

### Prerequisites

- macOS, Linux, or Windows
- Python 3.12
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

### Installation Steps

1. **Clone Repository:**
```bash
git clone https://github.com/subhash-halder/plant-disease-detection-system
cd plant-disease-detection-system
```

2. **Create Virtual Environment:**
```bash
# Using conda/mamba
mamba create -n tf-gpu python=3.12
mamba activate tf-gpu

# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies:**
```bash
pip install -r requirement.txt
```

4. **Download Dataset:**
- Visit Kaggle: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- Download and extract to `dataset/` directory

5. **Train Model (Optional):**
```bash
jupyter lab
# Open training_model.ipynb and run all cells
```

6. **Run Web Application:**
```bash
streamlit run app.py
```

## Appendix E: Troubleshooting Guide

### Common Issues and Solutions

**Issue 1: TensorFlow Installation Fails**
- Solution: Ensure Python version 3.8-3.12
- Use pip install --upgrade pip before installing

**Issue 2: Model File Not Found**
- Solution: Ensure `trained_plant_disease_model.keras` is in root directory
- Re-train model or download pre-trained weights

**Issue 3: Out of Memory During Training**
- Solution: Reduce batch size from 32 to 16 or 8
- Close other applications
- Use GPU if available

**Issue 4: Slow Predictions**
- Solution: Enable GPU acceleration
- Ensure model is cached (@st.cache_resource)
- Check system resources

**Issue 5: Web Application Won't Start**
- Solution: Check port 8501 is not in use
- Verify Streamlit installation: pip install --upgrade streamlit
- Check firewall settings

---

**End of Front Matter and Appendices**

