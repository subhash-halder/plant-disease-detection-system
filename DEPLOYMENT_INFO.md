# DEPLOYMENT INFORMATION

## Live Application

**URL:** [https://subhash-ai.streamlit.app/](https://subhash-ai.streamlit.app/)

**Status:** ✅ Live and Operational

---

## Deployment Details

### Platform
- **Service:** Streamlit Cloud
- **Hosting:** Cloud-based (managed by Streamlit)
- **Region:** Auto-selected based on optimal performance
- **Availability:** 24/7 public access

### Application Features
- 🏠 **Home Page:** System overview and capabilities
- 🔬 **Disease Detection:** Upload plant leaf images for instant diagnosis
- 📊 **Model Performance:** Interactive visualization of training metrics
- ℹ️ **About:** Complete project information and specifications

### Technical Specifications
- **Framework:** Streamlit 1.51.0
- **ML Backend:** TensorFlow 2.16.2, Keras 3.12.0
- **Model Size:** ~25 MB
- **Response Time:** <2 seconds for prediction
- **Supported Formats:** JPG, JPEG, PNG
- **Image Size:** Automatically resized to 128×128

---

## Usage Instructions

### For End Users

1. **Access the Application:**
   - Navigate to: https://subhash-ai.streamlit.app/
   - No installation or registration required

2. **Upload an Image:**
   - Click on "Disease Detection" in the sidebar
   - Click "Browse files" button
   - Select a plant leaf image (JPG, JPEG, or PNG)

3. **View Results:**
   - Disease name and plant species
   - Confidence percentage
   - Top-5 alternative predictions
   - Recommendations based on confidence level

4. **Explore Model Performance:**
   - Navigate to "Model Performance" page
   - View training/validation accuracy curves
   - See detailed architecture information
   - Review dataset statistics

### For Developers

**Local Development:**
```bash
# Clone repository
git clone <repository-url>
cd plant-disease-detection-system

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

**Deploy to Streamlit Cloud:**
1. Push code to GitHub repository
2. Connect repository to Streamlit Cloud
3. Select `app.py` as main file
4. Deploy automatically

---

## Performance Metrics

### Model Performance
- **Validation Accuracy:** 95.57%
- **Training Accuracy:** 98.35%
- **Disease Categories:** 38
- **Plant Species:** 14

### Application Performance
- **Uptime:** 99.9% (managed by Streamlit Cloud)
- **Prediction Speed:** <2 seconds per image
- **Concurrent Users:** Supports multiple simultaneous users
- **Geographic Availability:** Global access

---

## Supported Plant Species and Diseases

### Plant Species (14 total)
1. Apple
2. Blueberry
3. Cherry (including sour)
4. Corn (Maize)
5. Grape
6. Orange
7. Peach
8. Pepper (Bell Pepper)
9. Potato
10. Raspberry
11. Soybean
12. Squash
13. Strawberry
14. Tomato

### Disease Categories (38 total)
Including:
- Fungal diseases (Apple Scab, Black Rot, Powdery Mildew, etc.)
- Bacterial diseases (Bacterial Spot)
- Viral diseases (Yellow Leaf Curl Virus, Mosaic Virus)
- Pest damage (Spider Mites)
- Healthy states for each plant species

*Full list available in the application's About page*

---

## Deployment Architecture

```
User Browser
    ↓
Streamlit Cloud (HTTPS)
    ↓
app.py (Web Application)
    ↓
Trained Model (.keras file)
    ↓
TensorFlow/Keras Inference
    ↓
Results Display
```

### Components
1. **Frontend:** Streamlit reactive UI
2. **Backend:** Python application logic
3. **Model:** Pre-trained CNN (6.2M parameters)
4. **Storage:** Model weights and training history
5. **Hosting:** Streamlit Cloud infrastructure

---

## Security and Privacy

### Data Handling
- **Image Upload:** Temporary processing only
- **No Storage:** Uploaded images are not permanently stored
- **No User Tracking:** No personal data collection
- **HTTPS:** Secure encrypted communication

### Model Security
- **Read-Only:** Model weights are read-only in deployment
- **Sandboxed:** Application runs in isolated environment
- **Version Control:** Model versioning through Git

---

## Limitations and Considerations

### Current Limitations
1. **Controlled Conditions:** Model trained on laboratory images with uniform backgrounds
2. **Internet Required:** Cloud deployment requires internet connectivity
3. **Image Quality:** Best results with clear, well-lit leaf images
4. **Disease Scope:** Limited to 38 categories in current version
5. **No Treatment Advice:** Provides diagnosis only, not treatment recommendations

### Best Practices for Users
- Use clear, focused images of plant leaves
- Ensure good lighting (avoid shadows)
- Center the affected area in the frame
- Use high-resolution images when possible
- Consider multiple images for confirmation

---

## Future Enhancements

### Planned Features
- 📱 **Mobile App:** Native iOS and Android applications
- 🌍 **Offline Mode:** Downloadable model for offline use
- 📏 **Severity Assessment:** Disease severity level prediction
- 💊 **Treatment Recommendations:** Integrated treatment guidance
- 🗣️ **Multi-Language:** Support for multiple languages
- 📍 **Location-Based:** Geographic disease prevalence data

### Technical Roadmap
- Expand to more plant species and diseases
- Implement real-time video analysis
- Add explainable AI visualizations
- Integrate with IoT sensors
- Develop API for third-party integration

---

## Support and Feedback

### Reporting Issues
- **Technical Issues:** Report via GitHub issues or contact developer
- **Feature Requests:** Submit suggestions for improvements
- **Bug Reports:** Include screenshots and steps to reproduce

### Contact Information
- **Developer:** Subhash Halder
- **Institution:** Amity University Online
- **Program:** MCA (Machine Learning Specialization)
- **Project Guide:** Ayan Pal (Walmart)

### Academic Use
This application is part of an MCA project submission to Amity University Online. The system demonstrates practical application of Convolutional Neural Networks for agricultural disease detection.

---

## Acknowledgments

### Technology Stack
- **Streamlit:** For the incredible web framework
- **TensorFlow/Keras:** For deep learning capabilities
- **Streamlit Cloud:** For free hosting and deployment
- **PlantVillage/Kaggle:** For the comprehensive dataset

### Academic Support
- **Project Guide:** Ayan Pal, M.Tech (Walmart)
- **Institution:** Amity University Online
- **Program:** MCA with Machine Learning Specialization

---

## Version Information

- **Application Version:** 1.0
- **Model Version:** 1.0 (10 epochs, 95.57% validation accuracy)
- **Last Updated:** November 2025
- **Deployment Date:** November 2025

---

## Quick Links

- 🌐 **Live Application:** https://subhash-ai.streamlit.app/
- 📚 **Documentation:** See README.md
- 📊 **Project Report:** See project report documents
- 💻 **Source Code:** GitHub repository

---

**Status:** ✅ Deployed and Operational  
**Accessibility:** Public (no authentication required)  
**Cost:** Free (educational project)  
**Maintenance:** Active

---

*For the complete project documentation, technical details, and academic report, refer to the project repository and associated documentation files.*

