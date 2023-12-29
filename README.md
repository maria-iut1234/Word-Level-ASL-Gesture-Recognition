# CSE4554-Project: Word-Level American Sign Language (ASL) Recognition Project

## Overview

This project focuses on developing a model for American Sign Language (ASL) recognition using machine learning techniques. The dataset consists of ASL gestures captured in videos, and the goal is to accurately predict the corresponding sign language word or gesture.

## Folder Structure

The project is organized into the following main directories:

- **dataset:** Contains raw video data and the preprocessed dataset.
  - `videos/`: Raw video files.
  - `WLASL_v0.3.json`: JSON file containing dataset information.
  - `new_preprocessed-data/`: Preprocessed landmark data split into train, validation, and test sets.
  - `missing.txt`: Lists the video files that are missing from the dataset.

- **model_checkpoints:** Directory to store model checkpoints during training.

- **model:** Directory to save the trained model.

- **landmarks:** Directory for storing processed landmarks.

- `CSE 4554 Project.ipynb`: Notebook with the project implementation.

- `.gitignore`: File to specify files and directories excluded from version control.

/ML-Project
├── dataset/
│   ├── videos/
│   ├── WLASL_v0.3.json
│   ├── new_preprocessed-data/
│   ├── └── train/
│   ├── └── validation/
│   └── └── test/
│   ├── missing.txt
│
├── model_checkpoints/
│
├── model/
│
├── landmarks/
│
├── CSE_4554_Project.ipynb
└── .gitignore

## Project Structure

The project is organized into several key components:

1. **Data Preprocessing:**
   - Extracting landmarks from ASL videos.

2. **Folder Structure Setup:**
   - Set up the folder structure as outlined above and download the dataset with

2. **Data Augmentation:**
   - Applying various data augmentation techniques to enhance the model's robustness.

3. **Model Architecture:**
   - Utilizing a deep learning model for ASL recognition.
   - Incorporating masking and dropout layers for regularization.

4. **Training and Evaluation:**
   - Training the model with optimized learning rates and callbacks.
   - Evaluating the model's performance on a validation set.

5. **Label Encoding:**
   - Utilizing FastText to encode labels for better representation.

6. **Model Interpretation:**
   - Visualizing confusion matrices for model interpretation.
   - Generating ROC curves for multi-class classification.

7. **Model Deployment:**
   - Saving the trained model for future use.


# Usage Instructions

## 1. Environment Setup:
   - Ensure you have a Python 3.x environment set up on your machine.
   - Make sure to download the dataset and setup the folder structure provided above.

## 2. Running the Notebook:
   - Execute the provided notebook.
   - The notebook will automatically install the required dependencies and proceed with the following steps.

## 3. Data Processing:
   - The notebook includes code for processing the data. This step involves preparing and organizing the dataset for training.

## 4. Model Training:
   - The notebook will train the machine learning model using the processed data.
   - During this step, the model learns patterns and relationships in the data.

## 5. Model Evaluation:

- The trained model will be evaluated to assess its performance.

### Evaluation Metrics:

- Standard evaluation metrics, such as accuracy and loss, will be plotted to gauge the overall effectiveness of the model.

### Confusion Matrix:

- A confusion matrix will be generated to provide a detailed breakdown of the model's performance across different classes. This matrix is valuable for understanding the distribution of correct and incorrect predictions.

### ROC Curve and AUC:

- The Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) will be computed. These metrics are especially useful for binary and multiclass classification tasks, offering insights into the model's ability to discriminate between classes.

- The ROC curve illustrates the trade-off between true positive rate and false positive rate across various thresholds.

- The AUC represents the area under the ROC curve, with a higher AUC indicating better model discrimination.

- These metrics provide a comprehensive view of the model's discriminatory power and can be particularly insightful in scenarios where a balanced assessment of true positives and false positives is crucial.

## 6. Model Saving:
   - After successful training and evaluation, the notebook will save the trained model.
   - This saved model can be later used for making predictions on new data.




## Dependencies

- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Seaborn
- FastText

## Dataset

The WLASL dataset used in this project can be found [here](https://www.kaggle.com/datasets/risangbaskoro/WLASL-Processed).

## License

This project is licensed under the [MIT License](LICENSE).

## Contributors:

- **Shanta Maria**
  - *GitHub:* [NafisaMaliyat-iut](https://github.com/NafisaMaliyat-iut)

- **Nafisa Maliyat**
  - *GitHub:* [maria-iut1234](https://github.com/maria-iut1234)

