"""
COMP-4740 Machine Learning Assignment 1
Complete implementation of all required classifiers and analyses
By Harmit Patel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import confusion_matrix, make_scorer
import warnings
warnings.filterwarnings('ignore')

# Custom scoring functions for all required metrics
def calculate_metrics(y_true, y_pred):
    """Calculate PPV, NPV, Sensitivity, Specificity, and Accuracy"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # PPV (Positive Predictive Value) = Precision = TP / (TP + FP)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # NPV (Negative Predictive Value) = TN / (TN + FN)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Sensitivity (Recall/True Positive Rate) = TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Specificity (True Negative Rate) = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'PPV': ppv,
        'NPV': npv,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Accuracy': accuracy
    }

def ppv_score(y_true, y_pred):
    metrics = calculate_metrics(y_true, y_pred)
    return metrics['PPV']

def npv_score(y_true, y_pred):
    metrics = calculate_metrics(y_true, y_pred)
    return metrics['NPV']

def sensitivity_score(y_true, y_pred):
    metrics = calculate_metrics(y_true, y_pred)
    return metrics['Sensitivity']

def specificity_score(y_true, y_pred):
    metrics = calculate_metrics(y_true, y_pred)
    return metrics['Specificity']

def run_classifier_cv(classifier, X, y, clf_name):
    """Run 10-fold cross-validation and return all metrics"""
    scoring = {
        'accuracy': 'accuracy',
        'ppv': make_scorer(ppv_score),
        'npv': make_scorer(npv_score),
        'sensitivity': make_scorer(sensitivity_score),
        'specificity': make_scorer(specificity_score)
    }
    
    scores = cross_validate(classifier, X, y, cv=10, scoring=scoring)
    
    results = {
        'Classifier': clf_name,
        'PPV': np.mean(scores['test_ppv']),
        'NPV': np.mean(scores['test_npv']),
        'Sensitivity': np.mean(scores['test_sensitivity']),
        'Specificity': np.mean(scores['test_specificity']),
        'Accuracy': np.mean(scores['test_accuracy'])
    }
    
    return results

def find_best_k(X, y, k_range=range(1, 21)):
    """Find the best k for k-NN based on accuracy"""
    accuracies = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        accuracies.append(np.mean(scores))
    
    best_k = k_range[np.argmax(accuracies)]
    best_accuracy = max(accuracies)
    
    return best_k, accuracies, best_accuracy

def plot_dataset(X, y, dataset_name, save_path):
    """Plot the 2D dataset with different colors and shapes for each class"""
    plt.figure(figsize=(8, 6))
    
    # Class 0: blue circles
    mask_0 = y == 0
    plt.scatter(X[mask_0, 0], X[mask_0, 1], c='blue', marker='o', 
                label='Class 0', s=50, alpha=0.6, edgecolors='black')
    
    # Class 1: red triangles
    mask_1 = y == 1
    plt.scatter(X[mask_1, 0], X[mask_1, 1], c='red', marker='^', 
                label='Class 1', s=50, alpha=0.6, edgecolors='black')
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(f'Dataset: {dataset_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_knn_optimization(k_range, accuracies, dataset_name, save_path):
    """Plot k-NN accuracy vs k value"""
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('k Value', fontsize=12)
    plt.ylabel('Cross-Validation Accuracy', fontsize=12)
    plt.title(f'k-NN Optimization for {dataset_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Mark the best k
    best_k = k_range[np.argmax(accuracies)]
    best_acc = max(accuracies)
    plt.scatter([best_k], [best_acc], color='red', s=200, zorder=5, 
                label=f'Best k={best_k} (Acc={best_acc:.4f})')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_dataset(dataset_path, dataset_name, output_dir):
    """Process a single dataset with all classifiers"""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Load dataset (with header)
    data = pd.read_csv(dataset_path)
    X = data.iloc[:, :2].values
    y = data.iloc[:, 2].values.astype(int)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")
    
    # Plot dataset
    plot_dataset(X, y, dataset_name, f"{output_dir}/{dataset_name}_plot.png")
    
    
    # Initialize classifiers
    classifiers = {
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'Naive Bayes': GaussianNB(),
        'k-NN (k=1)': KNeighborsClassifier(n_neighbors=1, metric='euclidean'),
        'SVM-RBF': SVC(kernel='rbf', gamma='scale')
    }
    
    # Run classifiers and collect results
    results = []
    for clf_name, classifier in classifiers.items():
        print(f"Running {clf_name}...", end=' ')
        result = run_classifier_cv(classifier, X, y, clf_name)
        results.append(result)
        print(f"Accuracy: {result['Accuracy']:.4f}")
    
    # Find best k for k-NN
    print("\nOptimizing k-NN...")
    k_range = range(1, 21)
    best_k, accuracies, best_accuracy = find_best_k(X, y, k_range)
    print(f"Best k: {best_k} with accuracy: {best_accuracy:.4f}")
    
    # Plot k-NN optimization
    plot_knn_optimization(k_range, accuracies, dataset_name, 
                         f"{output_dir}/{dataset_name}_knn_optimization.png")
    
    
    # Add best k-NN to results
    best_knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    best_knn_result = run_classifier_cv(best_knn, X, y, f'k-NN (k={best_k})')
    results.append(best_knn_result)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    return results_df, best_k, accuracies

def main():
    """Main function to process all datasets"""
    import os
    
    # Create output directory
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset files (you need to place these in the same directory)
    datasets = {
        'circles0.3': 'circles0.3.csv',
        'moons1': 'moons1.csv',
        'spiral1': 'spiral1.csv',
        'twogaussians33': 'twogaussians33.csv',
        'twogaussians42': 'twogaussians42.csv',
        'halfkernel': 'halfkernel.csv'
    }
    
    all_results = {}
    knn_results = {}
    
    for dataset_name, dataset_file in datasets.items():
        try:
            dataset_path = f'./datasets/{dataset_file}'
            results_df, best_k, accuracies = process_dataset(
                dataset_path, dataset_name, output_dir
            )
            all_results[dataset_name] = results_df
            knn_results[dataset_name] = {
                'best_k': best_k,
                'accuracies': accuracies
            }
            
            # Save individual results
            results_df.to_csv(f'{output_dir}/{dataset_name}_results.csv', index=False)
            print(f"Results saved to {dataset_name}_results.csv\n")
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
    
    # Create summary report
    print("\n" + "="*60)
    print("SUMMARY OF ALL RESULTS")
    print("="*60)
    
    for dataset_name, results_df in all_results.items():
        print(f"\n{dataset_name}:")
        print(results_df.to_string(index=False))
        print(f"Best k for k-NN: {knn_results[dataset_name]['best_k']}")
    
    # Save summary
    with open(f'{output_dir}/summary.txt', 'w') as f:
        for dataset_name, results_df in all_results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"{dataset_name}\n")
            f.write(f"{'='*60}\n")
            f.write(results_df.to_string(index=False))
            f.write(f"\nBest k for k-NN: {knn_results[dataset_name]['best_k']}\n")
    
    print(f"\nAll results saved to {output_dir}")
    print("Processing complete!")

if __name__ == "__main__":
    main()