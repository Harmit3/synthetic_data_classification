# COMP 4740 Assignment 1

This project compares different machine learning classifiers on multiple synthetic datasets. The goal is to evaluate how different classifiers perform using standard classification metrics.

## Project Overview

The program trains and tests several classifiers, then generates performance results and plots.

## Classifiers Used

- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Gaussian Naive Bayes
- k-Nearest Neighbors (k-NN)
- Support Vector Machine (SVM) with RBF kernel

## Evaluation Metrics

The project evaluates each classifier using:

- PPV
- NPV
- Sensitivity
- Specificity
- Accuracy

## Project Structure

```text
Project/
├── main.py
├── datasets/
├── results/
├── README.md
└── .gitignore
```

## How to Run

First, install the required Python libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Then run the project:

```bash
python main.py
```

## Output

The program generates output files inside the `results/` folder, including:

- Dataset visualizations
- Classifier result CSV files
- k-NN optimization plots
- Summary text file

## Notes

The assignment report PDF is not included in GitHub by default because it may contain personal or submission-related information.
