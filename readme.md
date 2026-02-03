![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Neural%20Network-red?style=for-the-badge&logo=keras&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-ANN-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)


# Image Classification using Artificial Neural Network (ANN)

## Project Overview
This project implements an **Artificial Neural Network (ANN)** to perform image classification on the **Fashion-MNIST dataset**.  
The model classifies grayscale images of clothing items into 10 different categories.

The goal of this project is to understand the **end-to-end machine learning pipeline**, including data preprocessing, model building, training, and evaluation using multiple performance metrics.

---

##  Dataset
- **Dataset:** Fashion-MNIST  
- **Images:** 28×28 grayscale  
- **Classes:** 10 fashion categories  
- **Training samples:** 60,000  
- **Testing samples:** 10,000  

---

##  Class Labels
- T-shirt/top  
- Trouser  
- Pullover  
- Dress  
- Coat  
- Sandal  
- Shirt  
- Sneaker  
- Bag  
- Ankle boot  

---

##  Technologies & Tools Used
### Programming & Libraries
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

### Data Visualization
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-teal?style=for-the-badge)

### Machine Learning & Deep Learning
![TensorFlow](https://img.shields.io/badge/TensorFlow-Framework-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-API-red?style=for-the-badge&logo=keras&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?style=for-the-badge&logo=scikit-learn&logoColor=black)

### Tools
![Git](https://img.shields.io/badge/Git-Version%20Control-black?style=for-the-badge&logo=git)
![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![Colab](https://img.shields.io/badge/Google%20Colab-Cloud-yellow?style=for-the-badge&logo=googlecolab)


---

##  Machine Learning Pipeline
1. Data loading using Keras dataset API  
2. Data visualization and normalization  
3. ANN model creation using Keras Sequential API  
4. Model training with validation  
5. Performance evaluation using:
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  
   - Confusion Matrix  

---

##  Model Architecture
- Input layer: Flatten (28×28 → 784)  
- Hidden layer: Dense (128 neurons, ReLU activation)  
- Output layer: Dense (10 neurons, Softmax activation)  

The following diagram represents the architecture of the Artificial Neural Network (ANN) used in this project:

![ANN Model Architecture](model.png)
---

## Results
- **Test Accuracy:** ~88%  
- The model performs well on distinct classes such as **Trouser, Sandal, Bag, and Ankle Boot**  
- Lower performance observed on visually similar classes such as **Shirt, T-shirt, and Pullover**

---

##  Evaluation Metrics
A detailed classification report was generated using:
- Precision  
- Recall  
- F1-score  

This helps identify class-wise strengths and limitations of the ANN model.

---

##  Limitations
- ANN does not capture spatial features in images
- Similar apparel classes show confusion

---

##  Future Improvements
- Implement **Convolutional Neural Networks (CNN)**
- Apply data augmentation techniques
- Hyperparameter tuning

---

##  Conclusion
This project demonstrates a complete **image classification pipeline using ANN** and highlights the importance of evaluation metrics beyond accuracy.  
It serves as a strong **academic and learning project** in machine learning.

---

##  Author
**Aditya Dubey**

