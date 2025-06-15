# Josh Nguyen Machine Learning Project

This repository contains the coursework for **COMPSCI 361 - Machine Learning** at the **University of Auckland**. The project focuses on classifying BBC news articles into their respective categories (labels) based on their text content using various machine learning models.

---

## Project Overview

The core objective is to explore and evaluate different machine learning algorithms for text classification on a dataset of BBC articles. The project is structured into three main tasks:

1. **Exploratory Data Analysis (EDA)**: Understanding the dataset's characteristics, preparing text data for modeling.  
2. **Classification Models Learning**: Implementing and training various classification models.  
3. **Classification Quality Evaluation**: Assessing model performance, tuning hyperparameters, and comparing results.

---

## Setup and Installation

To run this project, you'll need Python 3.10+ and the following libraries. You can install them using pip:

pip install pandas matplotlib scikit-learn numpy

## Data

The project utilizes two CSV files:

- **`train.csv`**: The training dataset containing BBC articles and their categories.
- **`test.csv`**: The testing dataset used for evaluating model performance.

These files are expected to be in the root directory of the repository.

---

## Project Structure and Methodology

### Task 1: Exploratory Data Analysis (EDA)

- **Data Loading and Inspection**  
  Load `train.csv` and `test.csv` using pandas. Check dimensions and columns (`ArticleId`, `Text`, `Category`).

- **Text Pre-processing & Vectorization**  
  Use `TfidfVectorizer` from `sklearn.feature_extraction.text` to convert clean text into TF-IDF vectors.

- **Word Frequency Analysis**
  - Top 50 common terms in training set (e.g., `"said"`).
  - Category-specific terms:
    - `"broadband"` in **tech**
    - `"film"` in **entertainment**

- **Class Distribution**  
  Visualize class balance:
  - 216 articles in **tech**
  - 212 articles in **entertainment**

---

### Task 2: Classification Models Learning

- **Principal Component Analysis (PCA)**
  - Reduce high-dimensional TF-IDF vectors to 2D for visualization.

- **Naive Bayes (`MultinomialNB`)**
  - Train classifier and analyze:
    - Top 20 most predictive words per class.
    - Top 20 most discriminative words (via log-ratio).

- **k-Nearest Neighbors (`kNN`)**
  - Train with `k=5`, Euclidean distance on 2D PCA-reduced data.
  - Analyze precision, recall, and F1-scores.
  - Discuss the effects of `k` and distance metric.

- **Support Vector Machine (`SVM`)**
  - **Soft-Margin Linear SVM**: `kernel='linear'`, `C=1.0`
  - **Hard-Margin RBF Kernel SVM**: `kernel='rbf'`, large `C`, `gamma=0.1`

- **Neural Network (`MLPClassifier`)**
  - Single hidden layer MLP with ReLU activation and Adam optimizer.
  - Vary hidden units: 5, 20, 40
  - Analyze classification reports and training loss.

---

### Task 3: Classification Quality Evaluation

#### F1 Score vs. Training Set Fraction (`m`)

- Evaluate performance on training set fractions `m ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`
- **Training Accuracy**: No consistent trend.
- **Testing Accuracy**: Generally improves up to `m ≈ 0.5`, then plateaus or slightly decreases.

#### 5-fold Cross-Validation and Hyperparameter Tuning

Grid search with stratified 5-fold cross-validation is used to optimize:

- `alpha` for **Naive Bayes**
- `n_neighbors` for **kNN**
- `C`, `gamma` for **SVM**
- `hidden_layer_sizes`, `learning_rate_init` for **Neural Network**

#### Final Test F1 Scores (Best Hyperparameters)

| Model           | F1 Score |
|------------------|-----------|
| Naive Bayes      | 0.971     |
| kNN              | 0.981     |
| SVM              | 0.990     |
| Neural Network   | 0.990     |

---

### Conclusion

**SVM** and **Neural Network** deliver the highest F1-scores due to their capability to model complex, non-linear patterns.  
**kNN** also performs strongly with minimal tuning.  
**Naive Bayes** trails slightly behind, likely due to its feature independence assumption.

---

## Usage

This project is best viewed and executed in a **Jupyter Notebook** environment:

1. Clone the repository.
2. Install required libraries (see Setup and Installation).
3. Open the `.ipynb` file in **Jupyter Notebook** or **JupyterLab**.
4. Run the cells sequentially to reproduce the analysis and results.
