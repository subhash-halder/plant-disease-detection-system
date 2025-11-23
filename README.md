# Plant Disease Detection System

**MCA Major Project - Amity University Online**  
Machine Learning Specialization | Semester 4, Year 2025

## Project Overview

This project implements an automated Plant Disease Detection System using Convolutional Neural Networks (CNN) to classify plant diseases from leaf images. The system achieves **95.57% validation accuracy** across 38 disease categories spanning 14 plant species, with an intuitive web-based interface built using Streamlit.

**🌐 Live Demo:** [https://subhash-ai.streamlit.app/](https://subhash-ai.streamlit.app/)  
**📂 GitHub Repository:** [https://github.com/subhash-halder/plant-disease-detection-system](https://github.com/subhash-halder/plant-disease-detection-system)

**Key Achievements:**
- ✅ Custom CNN architecture with 5 convolutional blocks
- ✅ 95.57% validation accuracy (exceeding 90% hypothesis threshold)
- ✅ 98.35% training accuracy with minimal overfitting (2.78% gap)
- ✅ Real-time web application with confidence scores
- ✅ Comprehensive coverage: 38 disease categories, 14 plant species
- ✅ Deployed on Streamlit Cloud for global access

## Dataset

**Source:** [Kaggle New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

**Statistics:**
- Training images: 70,295
- Validation images: 17,572
- Test images: 33 (custom)
- Total: 87,867 labeled images
- Image size: 128×128 pixels (RGB)

**Plant Species:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

## Project Setup

### Prerequisites

- Python 3.12
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space
- macOS, Linux, or Windows

### Python Environment Setup (macOS with Apple Silicon)

```bash
# Install Miniforge (for Apple Silicon GPU acceleration)
brew install --cask miniforge

# Create dedicated environment
mamba create -n tf-gpu python=3.12

# Activate environment
eval "$(mamba shell hook --shell zsh)"
mamba activate tf-gpu
```

### Install Dependencies

```bash
python -m pip install -r requirement.txt
```

### Verify TensorFlow Installation

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Running the Project

### 1. Web Application (Streamlit)

**🌐 Live Demo:** [https://subhash-ai.streamlit.app/](https://subhash-ai.streamlit.app/)

Launch locally:

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

**Features:**
- 🏠 Home page with system overview
- 🔬 Disease detection with image upload
- 📊 Model performance visualization
- ℹ️ About page with project details

### 2. Model Training (Jupyter Lab)

Re-train the model or experiment with architecture:

```bash
jupyter lab
```

Open `training_model.ipynb` and run all cells.

### 3. Testing

Test the model on custom images:

```bash
jupyter lab
```

Open `test_plant_disease.ipynb` and run cells.

## Model Architecture

**Custom 5-Block Progressive CNN:**

```
Input (128×128×3)
  ↓
Block 1: Conv2D(32) → Conv2D(32) → MaxPool
Block 2: Conv2D(64) → Conv2D(64) → MaxPool
Block 3: Conv2D(128) → Conv2D(128) → MaxPool
Block 4: Conv2D(256) → Conv2D(256) → MaxPool
Block 5: Conv2D(512) → Conv2D(512) → MaxPool
  ↓
Dropout(0.25) → Flatten
  ↓
Dense(1500) → Dropout(0.4)
  ↓
Output (38 classes, Softmax)
```

**Total Parameters:** 6.2 million  
**Training Time:** ~22 minutes (10 epochs on Apple M4 Pro)

## Project Report Documentation

Complete MCA project report following Amity University guidelines:

### Report Files

- **Front_Matter_and_Appendices.md** - Title page, certificates, acknowledgments, abstract, TOC
- **Chapter_1_Introduction.md** - Background, objectives, scope, significance
- **Chapter_2_Literature_Review.md** - Comprehensive literature survey (15-20 papers)
- **Chapter_3_Theoretical_Framework.md** - CNN theory, mathematics, principles
- **Chapter_4_Research_Methodology.md** - Dataset, preprocessing, model design, training
- **Chapter_5_System_Design.md** - Architecture, components, data flow
- **Chapters_6_7_8_9_Complete.md** - Implementation, results, discussion, conclusion
- **Extended_Abstract_Subhash_Halder.md** - 4,200-word extended abstract
- **References_APA.md** - 21 academic references in APA 6th edition
- **literature_references.md** - Detailed reference summaries

### Report Statistics

- **Total Word Count:** ~25,000 words
- **Chapters:** 9 comprehensive chapters
- **Originality:** >85% (written in original language)
- **Format:** APA 6th edition
- **References:** 21 academic sources

### Compiling the Report

The report files are in Markdown format. To create final submission documents:

**Option 1: Pandoc (Recommended)**
```bash
# Install pandoc
brew install pandoc  # macOS
sudo apt install pandoc  # Linux

# Combine all chapters into single document
cat Front_Matter_and_Appendices.md Chapter_1_Introduction.md \
    Chapter_2_Literature_Review.md Chapter_3_Theoretical_Framework.md \
    Chapter_4_Research_Methodology.md Chapter_5_System_Design.md \
    Chapters_6_7_8_9_Complete.md References_APA.md > Complete_Project_Report.md

# Convert to DOCX
pandoc Complete_Project_Report.md -o Project_Report_Subhash_Halder.docx

# Convert to PDF
pandoc Complete_Project_Report.md -o Project_Report_Subhash_Halder.pdf
```

**Option 2: Manual**
- Copy content from each chapter file into Word/Google Docs
- Format according to APA 6th edition guidelines
- Apply consistent styling (Times New Roman, 12pt, double-spaced)
- Add page numbers and running head
- Export as PDF

## Project Structure

```
plant-disease-detection-system/
├── dataset/                    # Dataset directory
│   ├── train/                 # 70,295 training images
│   ├── valid/                 # 17,572 validation images
│   └── test/                  # 33 test images
├── app.py                     # Streamlit web application
├── training_model.ipynb       # Model training notebook
├── test_plant_disease.ipynb   # Testing notebook
├── trained_plant_disease_model.keras  # Trained model (25MB)
├── training_hist.json         # Training history
├── requirement.txt            # Python dependencies
├── README.md                  # This file
│
├── Chapter_1_Introduction.md
├── Chapter_2_Literature_Review.md
├── Chapter_3_Theoretical_Framework.md
├── Chapter_4_Research_Methodology.md
├── Chapter_5_System_Design.md
├── Chapters_6_7_8_9_Complete.md
├── Extended_Abstract_Subhash_Halder.md
├── Front_Matter_and_Appendices.md
├── References_APA.md
└── literature_references.md
```

## Results Summary

| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | 98.35% | 95.57% |
| Loss | 0.1390 | 0.1747 |
| Per-Class Avg | 96%+ | 95%+ |

**Top Performing Classes:**
- Corn Common Rust: 99.6%
- Grape Black Rot: 99.2%
- Corn Healthy: 99.1%

## Technology Stack

- **Deep Learning:** TensorFlow 2.16.2, Keras 3.12.0
- **Web Framework:** Streamlit 1.51.0
- **Data Processing:** NumPy 1.26.4, Pandas 2.3.3
- **Visualization:** Matplotlib 3.10.7, Plotly 5.24.1, Seaborn 0.13.2
- **Image Processing:** OpenCV 4.5.5, Pillow 12.0.0
- **ML Utilities:** Scikit-learn 1.7.2
- **Development:** Jupyter Lab 4.5.0, Python 3.12

## Academic Information

- **Student:** Subhash Halder
- **Enrollment:** A9929724000690(el)
- **Program:** MCA (Machine Learning Specialization)
- **Institution:** Amity University Online
- **Semester:** 4, Year 2025
- **Project Guide:** Ayan Pal, M.Tech (Computer Science)
- **Guide Organization:** Walmart, Senior Engineering Manager (15 years exp)

## References

Key references include:
- Krizhevsky et al. (2012) - AlexNet
- Simonyan & Zisserman (2014) - VGGNet
- He et al. (2016) - ResNet
- Mohanty et al. (2016) - Plant disease detection with CNNs
- Ferentinos (2018) - Deep learning for plant diseases
- Recent advances (2024-2025) - Multi-crop detection systems

See `References_APA.md` for complete bibliography (21 sources).

## Future Enhancements

- 📱 Mobile application (iOS/Android)
- 🌍 Expand to more plant species and diseases
- 📏 Disease severity assessment
- 💊 Treatment recommendations
- 🌐 Multi-language support
- ☁️ Cloud deployment
- 🔌 IoT sensor integration

## License

This project is developed for academic purposes as part of MCA curriculum at Amity University Online.

## Contact

**Subhash Halder**  
MCA Student, Amity University Online  
Enrollment: A9929724000690(el)

**Live Application:** [https://subhash-ai.streamlit.app/](https://subhash-ai.streamlit.app/)  
**GitHub Repository:** [https://github.com/subhash-halder/plant-disease-detection-system](https://github.com/subhash-halder/plant-disease-detection-system)

---

**Convolutional Neural Networks Resources:**
- [Stanford CS230 Cheat Sheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)

---

*Last Updated: November 2025*


