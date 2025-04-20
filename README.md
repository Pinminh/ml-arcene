# Machine Learning Project: Arcene Dataset Analysis

## 1. Introduction
The Arcene dataset, sourced from the UCI Machine Learning Repository ([Arcene Dataset](https://archive.ics.uci.edu/ml/datasets/Arcene)), is a high-dimensional dataset designed to distinguish cancer versus normal patterns from mass-spectrometric data. It was part of the NIPS 2003 feature selection challenge, featuring 900 instances and 10,000 features, including 7,000 real features and 3,000 probe features with no predictive power. This project aims to:
- Compare the performance of decision trees and Support Vector Machines (SVMs) on the original dataset.
- Apply feature selection methods to reduce dimensionality and evaluate their impact.
- Implement ensemble methods (bagging and boosting) to enhance model performance.
- Analyze which combination of models and methods performs best for this two-class classification task.

The high dimensionality and presence of irrelevant features make feature selection critical for improving model efficiency and accuracy. Ensemble methods are expected to further enhance performance by leveraging multiple models.

## 2. Data Exploration and Preprocessing
### 2.1 Dataset Overview
The Arcene dataset consists of:
- **Instances**: 900 (100 training, 100 validation, 700 test).
- **Features**: 10,000 (7,000 real, 3,000 probes).
- **Classes**: Two (cancer vs. normal).
- **Class Distribution**: Slightly imbalanced (e.g., 44 positive vs. 56 negative in training).
- **Missing Values**: None.

### 2.2 Preprocessing Steps
- **Load Data**: Import training, validation, and test sets from the UCI repository.
- **Check for Missing Values**: Confirm no missing values exist.
- **Analyze Target Distribution**: Visualize the class distribution to assess imbalance.
- **Feature Analysis**: Summarize feature statistics (e.g., mean, variance) to understand the data.
- **Data Splitting**: Use the provided training set for model training, validation set for hyperparameter tuning, and test set for final evaluation.

### 2.3 Handling Class Imbalance
Given the slight class imbalance, consider:
- **SMOTE**: Generate synthetic samples for the minority class using Synthetic Minority Over-sampling Technique.
- **Class Weighting**: Assign higher weights to the minority class during model training.

## 3. Feature Selection
The Arcene dataset’s high dimensionality necessitates feature selection to reduce overfitting, improve computational efficiency, and eliminate irrelevant probe features. The following methods are implemented:

### 3.1 Filter Methods
- **Variance Threshold**: Remove features with low variance, as they are unlikely to be informative.
- **Correlation-based Feature Selection**: Identify and remove highly correlated features to reduce redundancy.

### 3.2 Wrapper Methods
- **Recursive Feature Elimination (RFE)**: Use a decision tree or SVM to recursively eliminate the least important features, selecting the top-performing subset.
- **Forward Selection**: Start with no features and iteratively add the most informative ones based on model performance.

### 3.3 Embedded Methods
- **L1 Regularization (Lasso)**: Train a linear SVM with an L1 penalty to shrink irrelevant feature coefficients to zero.
- **Decision Tree Feature Importance**: Use Gini importance or information gain from decision trees to rank and select features.

### 3.4 Evaluation of Feature Selection
- Compare the number of features selected by each method.
- Assess whether probe features are effectively removed by analyzing feature importance scores.
- Evaluate the impact of feature selection on model performance in subsequent sections.

## 4. Model Training and Evaluation (Without Feature Selection)
### 4.1 Decision Trees
- **Implementation**: Use a standard decision tree classifier (e.g., scikit-learn’s `DecisionTreeClassifier`).
- **Hyperparameter Tuning**: Tune parameters like maximum depth and minimum samples per leaf using 5-fold cross-validation on the training set.
- **Evaluation**: Compute accuracy, precision, recall, F1-score, and AUC-ROC on the validation set.

### 4.2 Support Vector Machines (SVMs)
- **Implementation**: Use a linear SVM for computational efficiency, with an option to test an RBF kernel if resources allow.
- **Hyperparameter Tuning**: Tune the regularization parameter `C` and kernel parameters (if applicable) using cross-validation.
- **Evaluation**: Compute the same metrics as for decision trees.

### 4.3 Comparison
- Compare decision tree and SVM performance on the original dataset.
- Analyze which model handles the high-dimensional data better without feature selection.

## 5. Model Training and Evaluation (With Feature Selection)
- **Procedure**: For each feature selection method (variance threshold, correlation, RFE, L1 regularization, tree-based importance):
  - Select the reduced feature set.
  - Retrain decision trees and SVMs on the reduced dataset.
  - Evaluate performance on the validation set using the same metrics.
- **Comparison**: Assess the impact of each feature selection method on model performance and computational efficiency.
- **Probe Feature Analysis**: Check if the selected features exclude the 3,000 probe features, indicating effective feature selection.

## 6. Ensemble Methods
Ensemble methods combine multiple models to improve performance. The following methods are applied to both decision trees and SVMs:

### 6.1 Bagging
- **Random Forests** (for Decision Trees):
  - Train multiple decision trees on random subsets of data and features.
  - Tune parameters like the number of trees and maximum features per split.
- **Bagged SVMs**:
  - Train multiple SVMs on different data subsets and average their predictions.
  - Note: This may be computationally intensive, so limit the number of SVMs if needed.

### 6.2 Boosting
- **AdaBoost** (for Decision Trees):
  - Use decision trees as base learners, iteratively adjusting weights to focus on misclassified samples.
  - Tune the number of estimators and learning rate.
- **Gradient Boosting** (for Decision Trees):
  - Use Gradient Boosting Machines (e.g., scikit-learn’s `GradientBoostingClassifier` or XGBoost).
  - Tune parameters like learning rate, number of estimators, and tree depth.
- **Note**: Boosting SVMs is less common due to computational complexity, so focus on decision tree-based boosting.

### 6.3 Evaluation
- Evaluate all ensemble models on the validation set using the same metrics.
- Compare ensemble performance to individual models (with and without feature selection).

## 7. Final Evaluation
- **Model Selection**: Choose the best-performing models based on validation set performance (e.g., highest F1-score or AUC-ROC).
- **Test Set Evaluation**: Evaluate the selected models on the test set (700 instances) to obtain unbiased performance estimates.
- **Comparison**: Summarize the performance of all models in a table, including:
  - Original decision tree and SVM.
  - Decision tree and SVM with each feature selection method.
  - Ensemble models (Random Forests, Bagged SVMs, AdaBoost, Gradient Boosting).

### Performance Comparison Table
| Model | Feature Selection | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|-------------------|----------|-----------|--------|----------|---------|
| Decision Tree | None | - | - | - | - | - |
| SVM | None | - | - | - | - | - |
| Decision Tree | Variance Threshold | - | - | - | - | - |
| SVM | Variance Threshold | - | - | - | - | - |
| Decision Tree | RFE | - | - | - | - | - |
| SVM | RFE | - | - | - | - | - |
| Random Forest | None | - | - | - | - | - |
| Bagged SVM | None | - | - | - | - | - |
| AdaBoost | None | - | - | - | - | - |
| Gradient Boosting | None | - | - | - | - | - |

*Note*: Values will be filled in after experiments.

## 8. Conclusion and Future Work
### 8.1 Summary of Findings
- **Feature Selection Impact**: Discuss which feature selection methods improved performance and whether they effectively removed probe features.
- **Model Performance**: Identify the best-performing model (e.g., Random Forest with RFE) and explain why it excelled.
- **Ensemble Benefits**: Highlight how bagging and boosting enhanced performance compared to individual models.

### 8.2 Limitations
- Small training set (100 instances) may limit model generalization.
- Slight class imbalance may affect performance, despite mitigation strategies.
- Computational constraints may limit the use of bagged SVMs or complex feature selection methods.

### 8.3 Future Work
- Explore additional feature selection methods, such as mutual information or PCA.
- Test advanced ensemble methods, like stacking, to combine decision trees and SVMs.
- Address class imbalance further using techniques like cost-sensitive learning.
- Compare results with those from the NIPS 2003 challenge, where Bayesian neural networks and Random Forests performed well ([NIPS 2003 Results](https://papers.nips.cc/paper/2728-result-analysis-of-the-nips-2003-feature-selection-challenge)).
