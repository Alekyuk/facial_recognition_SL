# Facial Recognition with Supervised Learning

A comprehensive machine learning project that demonstrates building a binary classifier to identify Arnold Schwarzenegger in facial images using three different supervised learning algorithms.

##  Project Overview

This project implements a complete facial recognition pipeline that compares three supervised learning algorithms to accurately identify Arnold Schwarzenegger in facial images. The implementation leverages Principal Component Analysis (PCA) for dimensionality reduction and evaluates multiple classification approaches including Logistic Regression, K-Nearest Neighbors, and Support Vector Machines.

###  Key Features

- **Multi-algorithm comparison**: Logistic Regression, KNN, and SVM implementations
- **PCA-based feature extraction**: 150 principal components from facial images
- **Systematic evaluation**: 5-fold cross-validation with hyperparameter tuning
- **Security-focused metrics**: Prioritizes recall for VIP recognition applications
- **Clean pipeline architecture**: Scikit-learn pipelines with proper preprocessing
- **Comprehensive analysis**: Detailed performance evaluation and visualization

##  Dataset

The project uses a preprocessed dataset derived from the *Labeled Faces in the Wild* (LFW) dataset:

- **Total samples**: 190 images
- **Arnold Schwarzenegger**: 40 images (21.1%)
- **Other individuals**: 150 images (78.9%)
- **Features**: 150 principal components (PCA-transformed)
- **Target**: Binary classification (1 = Arnold, 0 = Other)

### Data Structure

```csv
PC1,PC2,...,PC150,Label
-2.06,0.58,...,0.12,1
-0.79,-0.67,...,0.1,1
...
```

##  Installation

### Prerequisites

- Python >= 3.10
- uv package manager (recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd facial-recognition-with-supervised-learning
   ```

2. **Install dependencies using uv**:
   ```bash
   uv sync
   ```

3. **Alternative installation with pip**:
   ```bash
   pip install -r requirements.txt
   ```

### Required Dependencies

```toml
[project]
name = "facial-recognition-with-supervised-learning"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "ipykernel>=7.0.1",
    "matplotlib>=3.10.7", 
    "pandas>=2.3.3",
    "scikit-learn>=1.7.2",
    "seaborn>=0.13.2",
]
```

##  Usage

### Running the Analysis

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook main.ipynb
   ```

2. **Execute the notebook** to run the complete analysis including:
   - Data loading and exploration
   - Exploratory data analysis
   - Model development and training
   - Hyperparameter optimization
   - Performance evaluation

### Key Code Examples

#### Data Loading
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("data/lfw_arnie_nonarnie.csv")
X = df.drop('Label', axis=1)
y = df['Label']

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=21, stratify=y
)
```

#### Model Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Create model pipeline
pipeline_logreg = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression())
])

# Train the model
pipeline_logreg.fit(X_train, y_train)
```

##  API Reference

### Model Configuration

#### Supported Algorithms

1. **Logistic Regression**
   - Parameters: `C`, `penalty`, `solver`, `max_iter`
   - Best for: Linear separability, interpretable results

2. **K-Nearest Neighbors (KNN)**
   - Parameters: `n_neighbors`, `weights`, `p`
   - Best for: Local pattern recognition, non-parametric approach

3. **Support Vector Machine (SVM)**
   - Parameters: `C`, `kernel`, `gamma`, `degree`
   - Best for: Complex decision boundaries, high-dimensional data

### Evaluation Metrics

- **Accuracy**: Overall classification performance
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

### Security Context Configuration

For VIP recognition systems, the project prioritizes **recall over precision**:

```python
# Evaluate with security-focused metrics
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)  # Critical for security

# Minimize false negatives (missed detections)
if fn > 0:
    print(f"CRITICAL: {fn} Arnold images were missed")
```

##  Configuration Options

### Data Configuration

- **Test size**: Default 20% (0.2)
- **Random state**: 21 for reproducibility
- **Stratification**: Maintains class balance across splits

### Model Configuration

#### Cross-Validation
- **Folds**: 5-fold cross-validation
- **Shuffle**: True for random ordering
- **Scoring**: Accuracy-based selection

#### Hyperparameter Ranges

**Logistic Regression**:
- `C`: [0.001, 0.01, 0.1, 1, 10, 100]
- `penalty`: ['l1', 'l2']
- `solver`: ['liblinear', 'saga']
- `max_iter`: [100, 200, 300, 500]

**KNN**:
- `n_neighbors`: [3, 5, 7, 9, 11]
- `weights`: ['uniform', 'distance']
- `p`: [1, 2] (Manhattan, Euclidean)

**SVM**:
- `C`: [0.1, 1, 10, 100]
- `kernel`: ['linear', 'rbf', 'poly']
- `gamma`: ['scale', 'auto', 0.1, 1]
- `degree`: [2, 3, 4] (for polynomial kernel)

### Visualization Options

- **Class distribution plots**: Bar charts and pie charts
- **Confusion matrices**: Heatmap visualization
- **Model comparison**: Performance bar charts

##  Performance Results

The project demonstrates strong classification performance across all three algorithms:

- **Baseline performance**: >78% accuracy without tuning
- **Optimized performance**: Significant improvement through hyperparameter tuning
- **Best model selection**: Based on cross-validation scores
- **Security metrics**: High recall rates for Arnold detection

### Model Selection Rationale

The analysis reveals that PCA features provide excellent class separability, with Logistic Regression often performing best due to the linear nature of the transformed feature space.

##  Security Applications

This project is designed with **VIP recognition systems** in mind:

- **High recall priority**: Minimizing missed detections (false negatives)
- **False positive tolerance**: Additional alerts can be manually filtered
- **Threshold adjustment**: Easily configurable for security vs. convenience trade-offs

### Deployment Considerations

- **Real-time performance**: Optimized for production inference
- **Confidence scoring**: Uncertainty quantification
- **Human verification**: High-stakes decision validation
- **Continuous monitoring**: Performance tracking and retraining

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run the analysis** to ensure no regressions

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Project Structure

```
facial-recognition-with-supervised-learning/
├── README.md                 # This file
├── main.ipynb              # Complete analysis notebook
├── pyproject.toml         # Project configuration
├── uv.lock              # Dependency lock file
└── data/
    └── lfw_arnie_nonarnie.csv  # Dataset with PCA features
```

##  Further Reading

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [Facial Recognition Systems](https://en.wikipedia.org/wiki/Facial_recognition_system)
- [Cross-Validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))

##  Known Limitations

- **Dataset size**: Limited to 190 samples restricts generalization
- **Single person focus**: Not directly transferable to other individuals
- **Static features**: PCA features lack adaptability to new data
- **Privacy considerations**: Requires careful handling of facial data

##  Future Enhancements

- **Multi-person recognition**: Extend to multiple celebrities
- **Dynamic feature learning**: Online PCA updates
- **Deep learning integration**: CNN-based feature extraction
- **Real-time processing**: Video stream analysis
- **Mobile deployment**: Optimized for edge devices

---

**Note**: This project demonstrates machine learning best practices and should be used as a reference for developing production facial recognition systems. Consider privacy regulations and ethical implications when deploying in real-world scenarios.