Josh Nguyen Machine Learning Project
This repository contains the coursework for COMPSCI 361 - Machine Learning at the University of Auckland. The project focuses on classifying BBC news articles into their respective categories (labels) based on their text content using various machine learning models.

Project Overview
The core objective is to explore and evaluate different machine learning algorithms for text classification on a dataset of BBC articles. The project is structured into three main tasks:

Exploratory Data Analysis (EDA): Understanding the dataset's characteristics, preparing text data for modeling.

Classification Models Learning: Implementing and training various classification models.

Classification Quality Evaluation: Assessing model performance, tuning hyperparameters, and comparing results.

Setup and Installation
To run this project, you will need Python 3.10+ and the following libraries. You can install them using pip:

pip install pandas matplotlib scikit-learn numpy


Data
The project utilizes two CSV files:

train.csv: The training dataset containing BBC articles and their categories.

test.csv: The testing dataset used for evaluating model performance.

These files are expected to be in the root directory of the repository.

Project Structure and Methodology
Task 1: Exploratory Data Analysis (EDA)
This task involves a deep dive into the BBC article dataset.

Data Loading and Inspection: The train.csv and test.csv datasets are loaded using pandas. Initial checks confirm the dimensions and column names (ArticleId, Text, Category).

Text Pre-processing & Vectorization: The text data is notably clean (lowercase, no punctuation), requiring minimal pre-processing. TfidfVectorizer from sklearn.feature_extraction.text is used to transform text into numerical feature vectors, capturing term frequency-inverse document frequency.

Word Frequency Analysis:

Analysis of the 50 most common terms across the entire training dataset. The word "said" is identified as highly frequent.

Category-specific word frequency analysis for "tech" and "entertainment" categories, revealing distinct vocabulary patterns (e.g., "broadband" in tech, "film" in entertainment).

Class Distribution: Visualization of the distribution of "tech" and "entertainment" articles in the training set, showing a balanced representation (216 tech, 212 entertainment).

Task 2: Classification Models Learning
This section focuses on applying various machine learning algorithms for text classification. A custom plot_2d_decision_boundary function is used to visualize classifiers in a 2D PCA-reduced space.

Principal Component Analysis (PCA):

Text data (originally high-dimensional from TF-IDF) is reduced to 2 dimensions using PCA for visualization purposes. This helps in understanding decision boundaries in a simplified space.

Naive Bayes (MultinomialNB):

A Multinomial Naive Bayes classifier is trained.

Analysis of the top 20 most predictive words for each class (highest P(word∣class)) and the top 20 most discriminative words between classes (highest log-ratio). The latter is concluded to be more effective in describing unique vocabulary.

k-Nearest Neighbors (kNN):

A kNN classifier is trained with k=5 and Euclidean distance on the PCA-reduced 2D data.

A classification report is generated, showing high precision, recall, and F1-scores for both categories.

Discussion on the impact of hyperparameters k (sensitivity to noise vs. smoothness) and metric (angular vs. rounder boundaries).

Support Vector Machine (SVM):

Soft-Margin Linear SVM: Implemented with kernel='linear' and C=1.0. The classification report shows excellent performance.

Hard-Margin RBF Kernel SVM: Implemented with kernel='rbf', a large C (for hard-margin approximation), and gamma=0.1. Also yields strong performance, demonstrating the ability to capture non-linear relationships.

Neural Network (NN - MLPClassifier):

A single-hidden-layer MLPClassifier is tested with varying numbers of hidden units (5, 20, 40). ReLU activation and Adam optimizer are used.

Classification reports for each configuration show high accuracy.

Analysis of the training loss curve indicates diminishing returns in reducing loss as the number of hidden units increases beyond a certain point.

Task 3: Classification Quality Evaluation
This task assesses and compares the performance of the implemented models, including hyperparameter tuning.

F1 Score vs. Training Set Fraction (m):

Evaluates Naive Bayes, kNN, SVM, and NN models on varying fractions of the training data (m∈0.1,0.3,0.5,0.7,0.9).

Training Accuracy: No general pattern is observed across algorithms, with each showing unique responses to changes in m.

Testing Accuracy: A clearer pattern emerges, where testing accuracy generally increases with m initially, peaks between m=0.3 and m=0.5, and then slightly decreases.

5-fold Cross-Validation and Hyperparameter Tuning:

A comprehensive grid search with 5-fold stratified cross-validation is performed to find the optimal hyperparameters for each model.

The best hyperparameters for each model are identified and listed, along with a brief explanation of what each hyperparameter controls (e.g., alpha for Naive Bayes smoothing, n_neighbors for kNN, C and gamma for SVM, hidden_layer_sizes and learning_rate_init for NN).

Final Test F1 Scores with Best Hyperparameters:

Each algorithm is retrained on the full training set using its optimally tuned hyperparameters.

The final F1-scores on the test set are reported:

Naive Bayes: 0.971

kNN: 0.981

SVM: 0.990

NN: 0.990

Conclusion: SVM and NN achieve the highest F1-scores, likely due to their ability to model complex, non-linear relationships in text data. kNN also performs very well, while Naive Bayes, despite a high score, is the lowest-performing, possibly due to its feature independence assumption.

Usage
This project is best viewed and executed in a Jupyter Notebook environment.

Clone the repository.

Install the required libraries (see Setup and Installation).

Open the .ipynb file in Jupyter Notebook or JupyterLab.

Run the cells sequentially to reproduce the analysis and results.

Acknowledgments
This project was developed by Josh Nguyen as part of the COMPSCI 361 - Machine Learning course at the University of Auckland.
