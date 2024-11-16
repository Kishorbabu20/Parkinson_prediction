### **Parkinson's Disease Prediction Using SVM**

#### **Description:**
This project aims to predict whether a person has Parkinson's disease based on features extracted from their voice. The model is built using **Support Vector Machine (SVM)**, a popular machine learning algorithm, and is trained on a dataset that includes various voice features (e.g., jitter, shimmer, harmonic-to-noise ratio) that are indicative of Parkinson's disease.

The system is equipped with a graphical user interface (GUI) using **Tkinter**, allowing users to enter their data and receive real-time predictions regarding Parkinson’s disease status.

#### **Overview:**
1. **Data Source**: 
   The model is trained on a dataset containing audio features of patients, where the target variable is a binary indicator: 0 for healthy individuals and 1 for those with Parkinson’s disease.
   
2. **Model**: 
   - The **SVM (Support Vector Machine)** classifier with a linear kernel is used for classification tasks.
   - The model is trained using voice-related features, such as jitter (variations in pitch) and shimmer (variations in amplitude), which are used to detect Parkinson’s disease.
   
3. **GUI**:
   - A simple user interface is created using **Tkinter**, where users can input their data and get a prediction from the trained model.

4. **Features Used**:
   - The dataset contains various voice-related features, including:
     - **MDVP (Jitter Abs)**: Measures pitch variation.
     - **MDVP:PPQ**: Pitch perturbation quotient.
     - **MDVP:Shimmer**: Amplitude variation.
     - **HNR**: Harmonics-to-noise ratio, among others.
   
5. **Prediction Process**:
   - The trained model predicts if a person has Parkinson’s disease based on these features, with 0 indicating a healthy individual and 1 indicating a Parkinson’s patient.

#### **Key Features:**
1. **Data Preprocessing**:
   - **Standardization**: Features are standardized using **StandardScaler** to ensure all input features have similar ranges.
   - **Train-Test Split**: The dataset is split into training and testing sets (80% training, 20% testing).
   
2. **SVM Classification**:
   - SVM with a **linear kernel** is used to build a classification model.
   - The model is trained and evaluated on the data, with accuracy reported on both training and testing sets.
   
3. **Prediction**:
   - The system takes input features (age, gender, and voice-related features), processes them, and uses the trained model to predict the likelihood of Parkinson's disease.

4. **GUI Integration**:
   - A GUI is created using **Tkinter** to allow users to input their data manually and get the result.
   - The GUI includes input fields for age, gender, and other features, and displays the prediction (whether the person has Parkinson’s or not).

5. **Real-Time Prediction**:
   - After entering the data, users can click on a "Submit" button to receive a prediction about Parkinson’s disease.

#### **Usage:**
1. **Dataset**: 
   - The project uses a dataset of voice features (from a CSV file), which is assumed to be in the same directory as the script.
   
2. **Training**:
   - The dataset is loaded and preprocessed (handling missing values, standardizing features).
   - An SVM model is trained on the dataset to predict the disease status.

3. **Model Evaluation**:
   - The model is evaluated on a test set, and the accuracy is displayed for both training and testing data.
   
4. **GUI Application**:
   - The GUI allows a user to input their details such as age, gender, and audio-related features, which are then fed into the trained model for prediction.

#### **Configuration:**
1. **Libraries Required**:
   - **NumPy**: For handling numerical operations.
   - **Pandas**: For loading and manipulating the dataset.
   - **scikit-learn**: For machine learning tools (SVM, train-test split, standardization).
   - **Tkinter**: For creating the GUI.
   
   Install the required libraries:
   ```bash
   pip install numpy pandas scikit-learn
   ```

2. **Directory Structure**:
   ```
   Parkinsons_Disease_Prediction/
   ├── parkinsons (1).data      # The dataset CSV file
   ├── parkinsons_predictor.py  # The script containing the code
   ├── gui.py                   # The GUI for input
   ```

3. **Input Data**:
   - **CSV Data Format**: The dataset `parkinsons (1).data` contains various voice-related features and a target variable `status` (0 for healthy, 1 for Parkinson’s patient).

4. **Model Configuration**:
   - **SVM Model**: The model uses the `SVC` classifier from `scikit-learn` with a **linear kernel**.
   - **Standardization**: Input features are standardized using **StandardScaler** before model training and prediction.

5. **GUI Configuration**:
   - The GUI allows users to enter the following details:
     - **Age**: Numeric value.
     - **Gender**: Binary (0 for female, 1 for male).
     - **MDVP and HNR features**: These values should be entered for model prediction.
     
   - A **Submit** button triggers the prediction.

6. **Output**:
   - The model outputs whether the person is predicted to have Parkinson’s disease (`1`) or not (`0`).
   - This result is shown on the console or in the GUI after clicking the submit button.

#### **How It Works:**
1. **Train the Model**:
   - Load and preprocess the data.
   - Train an SVM model on the dataset.
   
2. **Predict Using the Model**:
   - Accept input features from the user through the GUI.
   - Standardize the input features using the same scaler fitted on the training data.
   - Predict whether the individual has Parkinson’s disease based on the input features.

3. **Display the Result**:
   - If the model predicts `0`, the person does not have Parkinson's.
   - If the model predicts `1`, the person has Parkinson’s disease.
