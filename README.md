# CSE4554-Project
# Word-Level American Sign Language (ASL) Recognition Project

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

- **notebooks:** 
  - `CSE 4554 Project.ipynb`: Notebook with the project implementation.

- `.gitignore`: File to specify files and directories excluded from version control.

## Project Structure

The project is organized into several key components:

1. **Data Preprocessing:**
   - Extracting landmarks from ASL videos.
   - Handling missing videos and creating a processed dataset.

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

## Usage

1. **Environment Setup:**
   - Ensure you have Python environment set up
   - Run the notebook 

2. **Data Processing:**
   - Run scripts to preprocess, augment, and save landmarks.

3. **Model Training:**
   - Train the ASL recognition model using the provided code.

4. **Model Evaluation:**
   - Evaluate the model performance using confusion matrices and ROC curves.

5. **Model Deployment:**
   - Save the trained model for future use.

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

## Acknowledgments

- FastText: [FastText Pre-trained Models](https://fasttext.cc/docs/en/crawl-vectors.html)

