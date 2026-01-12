# Customer Churn Prediction 📊

A machine learning project to predict customer churn in the telecommunications industry using the Telco Customer Churn dataset.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Customer churn (also known as customer attrition) refers to when customers stop doing business with a company. This project aims to predict which customers are likely to churn based on their demographic information, account details, and service usage patterns. By identifying at-risk customers, businesses can take proactive measures to retain them and reduce revenue loss.

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), which contains information about:
- **Customer demographics**: gender, age, partners, dependents
- **Account information**: tenure, contract type, payment method, billing
- **Services**: phone service, internet service, online security, tech support, streaming services
- **Target variable**: Churn (Yes/No)

The dataset includes approximately 7,000+ customer records with 21 features.

## 📁 Project Structure

```
Customer-Churn_prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Dataset
├── customer_churn_prediction2.ipynb         # Main analysis notebook
├── main.ipynb                               # Alternative/experimental notebook
└── README.md                                # Project documentation
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/Anburaja07/Customer-Churn_prediction.git
cd Customer-Churn_prediction
```

2. **Install required dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Or create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Usage

1. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

2. **Open the main notebook**
   - Navigate to `customer_churn_prediction2.ipynb` or `main.ipynb`
   - Run the cells sequentially to reproduce the analysis

3. **Explore the data and models**
   - Follow the step-by-step analysis in the notebooks
   - Modify hyperparameters or try different models as needed

## 🔬 Methodology

### 1. **Data Preprocessing**
- Handling missing values
- Encoding categorical variables
- Feature scaling/normalization
- Train-test split

### 2. **Exploratory Data Analysis (EDA)**
- Univariate and bivariate analysis
- Correlation analysis
- Visualization of churn patterns
- Identifying key features

### 3. **Feature Engineering**
- Creating new relevant features
- Feature selection
- Dimensionality reduction (if applicable)

### 4. **Model Building**
Experimenting with various classification algorithms:
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
- Support Vector Machines

### 5. **Model Evaluation**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curves
- Confusion Matrix
- Cross-validation
- Hyperparameter tuning

## 📈 Results

*Add your specific results here after completing the analysis, such as:*
- Best performing model: [Model Name]
- Accuracy: XX%
- Precision: XX%
- Recall: XX%
- Key features influencing churn: [Feature 1, Feature 2, etc.]

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning algorithms
- **Jupyter Notebook** - Interactive development environment

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request



---

⭐ If you find this project useful, please consider giving it a star!
