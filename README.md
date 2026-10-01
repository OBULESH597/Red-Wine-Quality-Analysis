# 🍷 Red Wine Quality Analysis

## 📊 Data Analytics & Machine Learning Project

An end-to-end **Data Analytics and Machine Learning project** focused on analyzing red wine chemical properties and understanding the factors associated with wine quality.

This project covers the complete data analysis workflow, including **data cleaning, exploratory data analysis (EDA), visualization, correlation analysis, outlier detection, target analysis, class balancing using SMOTE, machine learning, model evaluation, and feature importance analysis**.

---

## 📌 Project Overview

Wine quality can be influenced by several chemical properties such as:

- Alcohol
- Volatile Acidity
- Sulphates
- Citric Acid
- Chlorides
- Density
- pH
- Total Sulfur Dioxide
- Free Sulfur Dioxide
- Residual Sugar
- Fixed Acidity

The objective of this project is to explore these variables, identify important patterns, and build machine learning models to analyze wine quality.

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Understand the Red Wine Quality dataset.
2. Perform data cleaning and preprocessing.
3. Identify missing values and duplicate records.
4. Perform statistical analysis.
5. Conduct univariate analysis.
6. Perform bivariate analysis.
7. Analyze relationships between numerical variables.
8. Perform correlation analysis.
9. Detect potential outliers.
10. Analyze the target variable `quality`.
11. Handle class imbalance using SMOTE.
12. Build machine learning models.
13. Evaluate model performance.
14. Identify important features using Random Forest.
15. Create an interactive analytics dashboard.
16. Summarize the major findings from the analysis.

---

# 📂 Dataset

The project uses the **Red Wine Quality dataset**.

### Dataset Information

| Information | Value |
|---|---:|
| Original Records | 1,599 |
| Duplicate Records | 240 |
| Final Clean Records | 1,359 |
| Missing Values | 0 |
| Features | 11 |
| Target Variable | `quality` |
| Quality Range | 3–8 |

After removing duplicate records, the final dataset contains **1,359 unique records**.

---

# 📋 Dataset Features

| Feature | Description |
|---|---|
| `fixed.acidity` | Amount of fixed acids in wine |
| `volatile.acidity` | Amount of volatile acids |
| `citric.acid` | Amount of citric acid |
| `residual.sugar` | Remaining sugar after fermentation |
| `chlorides` | Amount of salt in the wine |
| `free.sulfur.dioxide` | Free sulfur dioxide concentration |
| `total.sulfur.dioxide` | Total sulfur dioxide concentration |
| `density` | Density of the wine |
| `pH` | Acidity/basicity level |
| `sulphates` | Sulphate concentration |
| `alcohol` | Alcohol percentage |
| `quality` | Wine quality score |

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Imbalanced-learn

### Tools

- Google Colab
- Jupyter Notebook
- GitHub

### Machine Learning

- Logistic Regression
- Random Forest Classifier
- SMOTE
- Train-Test Split
- Classification Report
- Confusion Matrix
- Feature Importance

---

# 🔄 Project Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Duplicate Removal
   ↓
Missing Value Analysis
   ↓
Statistical Analysis
   ↓
Univariate Analysis
   ↓
Bivariate Analysis
   ↓
Correlation Analysis
   ↓
Outlier Detection
   ↓
Target Variable Analysis
   ↓
Train-Test Split
   ↓
SMOTE
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
Interactive Dashboard
   ↓
Final Conclusions

1️⃣ Data Understanding
The first step was to understand the dataset structure.
The following operations were performed:
df.head()df.tail()df.sample()df.columnsdf.shapedf.info()


The dataset initially contained:
Rows: 1599
Columns: 13

One index-like column was removed:
df.drop('Unnamed: 0', axis=1, inplace=True)


After removing the unnecessary column:
Rows: 1599
Columns: 12

2️⃣ Data Cleaning
Missing Values
Missing values were checked using:
df.isnull().sum()


Result
No missing values were found.

Duplicate Records
Duplicate records were checked using:
df.duplicated().sum()


Result
Duplicate records: 240

The duplicate records were removed:
df.drop_duplicates(inplace=True)


Final Dataset
Original Records: 1599
Duplicates Removed: 240
Final Records: 1359

3️⃣ Statistical Analysis
The statistical summary was generated using:
df.describe()


This helped analyze:
- Mean
- Standard deviation
- Minimum value
- Maximum value
- 25th percentile
- Median
- 75th percentile
Example:
df.describe()


4️⃣ Univariate Analysis
Univariate analysis was performed to understand individual variables.
Visualizations included:
- Histograms
- Distribution plots
- Boxplots
- Quality distribution
Example:
plt.figure(figsize=(8,5))sns.histplot(df['alcohol'], kde=True)plt.title('Alcohol Distribution')plt.show()


5️⃣ Bivariate Analysis
Bivariate analysis was performed to understand relationships between two variables.
Visualizations
Alcohol vs Quality
sns.boxplot(x='quality', y='alcohol', data=df)plt.title('Alcohol vs Wine Quality')plt.show()


Volatile Acidity vs Quality
sns.boxplot(x='quality', y='volatile.acidity', data=df)plt.title('Volatile Acidity vs Wine Quality')plt.show()


Sulphates vs Quality
sns.boxplot(x='quality', y='sulphates', data=df)plt.title('Sulphates vs Wine Quality')plt.show()


Alcohol vs Density
sns.scatterplot(    x='alcohol',    y='density',    data=df)plt.title('Alcohol vs Density')plt.show()


6️⃣ Correlation Analysis
Correlation analysis was performed to understand relationships between numerical variables.
plt.figure(figsize=(12,8))sns.heatmap(    df.corr(),    annot=True,    cmap='coolwarm')plt.title('Correlation Heatmap')plt.show()


The correlation matrix helped identify:
- Positive relationships
- Negative relationships
- Weak relationships
- Stronger relationships between variables
7️⃣ Outlier Detection
Boxplots and the IQR method were used to identify potential outliers.
Q1 = df.quantile(0.25)Q3 = df.quantile(0.75)IQR = Q3 - Q1lower = Q1 - 1.5 * IQRupper = Q3 + 1.5 * IQR


Outliers were examined across numerical features.
8️⃣ Target Variable Analysis
The target variable is:
quality

The quality scores range from:
3 to 8

The distribution was analyzed using:
df['quality'].value_counts().sort_index()


A count plot was also created:
sns.countplot(    x='quality',    data=df)plt.title('Wine Quality Distribution')plt.show()


The dataset has an imbalanced quality distribution, with quality scores 5 and 6 occurring more frequently than several other quality levels.
9️⃣ Machine Learning
Feature and Target Separation
X = df.drop('quality', axis=1)y = df['quality']


Where:
X = Input Features
y = Target Variable

🔟 Train-Test Split
The dataset was divided into training and testing datasets.
from sklearn.model_selection import train_test_splitX_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.20,    random_state=42,    stratify=y)


Split
Training Data: 80%
Testing Data: 20%

1️⃣1️⃣ Handling Class Imbalance with SMOTE
The target variable has an imbalanced distribution.
To address class imbalance, SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data.
from imblearn.over_sampling import SMOTEsmote = SMOTE(random_state=42)X_train_smote, y_train_smote = smote.fit_resample(    X_train,    y_train)


SMOTE was applied only to the training data to avoid data leakage into the test set.
1️⃣2️⃣ Logistic Regression
A Logistic Regression model was trained as one of the machine learning approaches.
from sklearn.linear_model import LogisticRegressionlr = LogisticRegression(    max_iter=1000)lr.fit(    X_train_smote,    y_train_smote)y_pred_lr = lr.predict(X_test)


1️⃣3️⃣ Random Forest Classifier
A Random Forest model was also trained.
from sklearn.ensemble import RandomForestClassifierrf = RandomForestClassifier(    n_estimators=200,    random_state=42)rf.fit(    X_train_smote,    y_train_smote)y_pred_rf = rf.predict(X_test)


1️⃣4️⃣ Model Evaluation
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report
Example:
from sklearn.metrics import (    accuracy_score,    classification_report,    confusion_matrix)print("Accuracy:", accuracy_score(y_test, y_pred_rf))print(    classification_report(        y_test,        y_pred_rf    ))


1️⃣5️⃣ Confusion Matrix
A confusion matrix was used to understand the classification performance across different wine quality classes.
cm = confusion_matrix(    y_test,    y_pred_rf)sns.heatmap(    cm,    annot=True,    fmt='d',    cmap='Blues')plt.xlabel('Predicted')plt.ylabel('Actual')plt.title('Random Forest Confusion Matrix')plt.show()


1️⃣6️⃣ Feature Importance
Random Forest feature importance was used to identify the variables that contributed most to the model.
feature_importance = pd.DataFrame({    'Feature': X.columns,    'Importance': rf.feature_importances_})feature_importance = feature_importance.sort_values(    by='Importance',    ascending=False)print(feature_importance)


Important Features Observed
Some of the important variables included:
- Alcohol
- Sulphates
- Total Sulfur Dioxide
- Volatile Acidity
- Density
- Chlorides
- pH
Feature importance indicates how useful a variable was to the trained Random Forest model; it does not by itself establish causation.
📊 Dashboard
An interactive dashboard was created using Plotly.
The dashboard contains:
KPI Cards
- Total Samples
- Average Quality
- Quality Range
Visualizations
- Wine Quality Distribution
- Alcohol by Quality
- Alcohol vs Quality
- Correlation Heatmap
- Random Forest Feature Importance
📈 Key EDA Findings
The analysis produced several observations:
🍷 Wine Quality
Wine quality scores range from 3 to 8.
Quality scores 5 and 6 represent a large portion of the dataset.
🧪 Data Quality
- No missing values were identified.
- 240 duplicate records were identified.
- After duplicate removal, 1,359 unique records remained.
🍺 Alcohol
Alcohol showed an important relationship with wine quality in the exploratory analysis and was also among the more important Random Forest features.
🧪 Volatile Acidity
Volatile acidity showed a generally negative relationship with wine quality in the exploratory analysis.
📊 Feature Importance
Random Forest identified several variables as important for classification, including:
Alcohol
Sulphates
Total Sulfur Dioxide
Volatile Acidity
Density

📁 Project Structure
Recommended GitHub repository structure:
Red-Wine-Quality-Analysis/
│
├── 📁 data/
│   └── wineQualityReds.csv
│
├── 📁 notebooks/
│   └── Red_Wine_Quality_Analysis.ipynb
│
├── 📁 dashboard/
│   └── red_wine_final_dashboard.html
│
├── 📁 presentation/
│   └── Red_Wine_Quality_Analysis_Presentation.pptx
│
├── 📁 images/
│   ├── wine_quality_distribution.png
│   ├── correlation_heatmap.png
│   ├── alcohol_vs_quality.png
│   └── feature_importance.png
│
├── requirements.txt
└── README.md

⚙️ Installation
Clone the repository:
git clone https://github.com/OBULESH597/Red-Wine-Quality-Analysis.git

Move into the project directory:
cd Red-Wine-Quality-Analysis

Install the required libraries:
pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn

▶️ How to Run
Option 1 — Google Colab
Upload the notebook to Google Colab and run the cells sequentially.
Option 2 — Jupyter Notebook
Run:
jupyter notebook

Open:
Red_Wine_Quality_Analysis.ipynb

and execute the cells.
📦 requirements.txt
Create a file named:
requirements.txt

Add:
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
imbalanced-learn
jupyter

📊 Skills Demonstrated
This project demonstrates practical knowledge of:
Data Analytics
- Data Cleaning
- Data Preprocessing
- Exploratory Data Analysis
- Statistical Analysis
- Univariate Analysis
- Bivariate Analysis
- Correlation Analysis
- Outlier Detection
- Data Visualization
Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
Machine Learning
- Train-Test Split
- SMOTE
- Logistic Regression
- Random Forest
- Classification
- Model Evaluation
- Confusion Matrix
- Feature Importance
💡 Business/Analytical Perspective
The project demonstrates how structured data can be used to:
- Explore wine quality patterns
- Identify relationships between chemical properties
- Understand important variables
- Prepare data for predictive modeling
- Compare machine learning approaches
- Communicate analytical findings through visualizations and dashboards
🚀 Future Improvements
Potential future improvements include:
- Hyperparameter tuning
- Cross-validation
- Testing additional classification algorithms
- Feature engineering
- Model optimization
- Interactive dashboard filters
- Deployment as a web application
- Comparing additional evaluation metrics
- Experimenting with alternative approaches to the quality target
📚 Project Deliverables
This repository contains:
- 📓 Data Analysis Notebook
- 📊 Exploratory Data Analysis
- 📈 Data Visualizations
- 🤖 Machine Learning Models
- 📊 Feature Importance Analysis
- 📉 Confusion Matrix
- 📊 Interactive Dashboard
- 📑 Project Presentation
- 📄 Dataset
- 📝 Project Documentation
👨‍💻 Author
Obulesu Polisetti
B.Tech Computer Science & Engineering Graduate | Data Analyst Aspirant
Skills
SQL
Python
Pandas
NumPy
Microsoft Excel
Power BI
Matplotlib
Seaborn
Data Cleaning
Data Analysis
Data Visualization
Machine Learning

🔗 Connect With Me
LinkedIn:
https://linkedin.com/in/obulesu-polisetti-984b0236b
GitHub:
https://github.com/OBULESH597
Portfolio:
https://obulesh597.github.io/PORTFOLIO-DATA-ANALYST/portfolio/
⭐ If You Find This Project Useful
If you find this project useful or interesting, consider giving the repository a ⭐.
Thank you for visiting this project! 🍷📊
📌 Tags
#DataAnalytics #DataAnalyst #Python #Pandas #NumPy #EDA #DataVisualization #MachineLearning #ScikitLearn #RandomForest #SMOTE #Plotly #DataScience #GitHub #Analytics #PythonProject

### Recommended GitHub repository name

Use:

**`Red-Wine-Quality-Analysis`**

And your GitHub repository description can be:

> **End-to-end Red Wine Quality Analysis using Python, EDA, data visualization, SMOTE, machine learning, Random Forest, and an interactive Plotly dashboard.**

For your recruiter-facing GitHub, I would keep the repository structure exactly as shown above: **`data → notebooks → dashboard → images → presentation → README.md → requirements.txt`**. This makes the project much easier for recruiters to understand quickly.






    








send me entire in one code





 






Yes. Copy everything below as one single README.md file:
# 🍷 Red Wine Quality Analysis

## 📊 Data Analytics & Machine Learning Project

An end-to-end **Data Analytics and Machine Learning project** focused on analyzing red wine chemical properties and understanding the factors associated with wine quality.

This project covers **data cleaning, exploratory data analysis (EDA), data visualization, correlation analysis, outlier detection, target analysis, SMOTE, machine learning, model evaluation, feature importance, and an interactive dashboard**.

---

## 🎯 Project Objectives

- Understand the Red Wine Quality dataset
- Clean and preprocess the data
- Identify missing values and duplicate records
- Perform statistical analysis
- Perform univariate analysis
- Perform bivariate analysis
- Analyze correlations between variables
- Detect potential outliers
- Analyze the target variable
- Handle class imbalance using SMOTE
- Build machine learning models
- Evaluate model performance
- Identify important features
- Create an interactive analytics dashboard
- Present the final analytical findings

---

## 📂 Dataset Overview

The project uses the **Red Wine Quality dataset**.

| Metric | Value |
|---|---:|
| Original Records | 1,599 |
| Duplicate Records | 240 |
| Final Clean Records | 1,359 |
| Missing Values | 0 |
| Features | 11 |
| Target Variable | `quality` |
| Quality Range | 3–8 |

After removing duplicate records, the final dataset contains **1,359 unique records**.

---

## 📋 Dataset Features

| Feature | Description |
|---|---|
| `fixed.acidity` | Amount of fixed acids in wine |
| `volatile.acidity` | Amount of volatile acids |
| `citric.acid` | Amount of citric acid |
| `residual.sugar` | Remaining sugar after fermentation |
| `chlorides` | Amount of salt in the wine |
| `free.sulfur.dioxide` | Free sulfur dioxide concentration |
| `total.sulfur.dioxide` | Total sulfur dioxide concentration |
| `density` | Density of the wine |
| `pH` | Acidity/basicity level |
| `sulphates` | Sulphate concentration |
| `alcohol` | Alcohol percentage |
| `quality` | Wine quality score |

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Python Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Imbalanced-learn

### Tools

- Google Colab
- Jupyter Notebook
- GitHub

### Machine Learning

- Logistic Regression
- Random Forest Classifier
- SMOTE
- Train-Test Split
- Classification Report
- Confusion Matrix
- Feature Importance

---

# 🔄 Project Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Duplicate Removal
   ↓
Missing Value Analysis
   ↓
Statistical Analysis
   ↓
Univariate Analysis
   ↓
Bivariate Analysis
   ↓
Correlation Analysis
   ↓
Outlier Detection
   ↓
Target Variable Analysis
   ↓
Train-Test Split
   ↓
SMOTE
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
Interactive Dashboard
   ↓
Final Conclusions

1️⃣ Data Understanding
The dataset was first explored to understand its structure, columns, data types, and sample records.
df.head()df.tail()df.sample()df.columnsdf.shapedf.info()


The original dataset contained:
Rows: 1599
Columns: 13

An unnecessary index column was removed:
df.drop('Unnamed: 0', axis=1, inplace=True)


After removing the unnecessary column:
Rows: 1599
Columns: 12

2️⃣ Data Cleaning
Missing Value Analysis
Missing values were checked using:
df.isnull().sum()


Result
No missing values were found.

Duplicate Analysis
Duplicate records were checked using:
df.duplicated().sum()


Result
Duplicate Records: 240

Duplicate records were removed:
df.drop_duplicates(inplace=True)


Final Dataset
Original Records : 1599
Duplicates Removed: 240
Final Records    : 1359

3️⃣ Statistical Analysis
Statistical information was obtained using:
df.describe()


This provided information about:
- Count
- Mean
- Standard deviation
- Minimum
- Maximum
- 25th percentile
- Median
- 75th percentile
4️⃣ Univariate Analysis
Univariate analysis was performed to understand individual variables.
Visualizations
- Histograms
- Distribution plots
- Boxplots
- Target variable distribution
Example:
plt.figure(figsize=(8,5))sns.histplot(    df['alcohol'],    kde=True)plt.title('Alcohol Distribution')plt.xlabel('Alcohol')plt.ylabel('Frequency')plt.show()


5️⃣ Bivariate Analysis
Bivariate analysis was performed to understand relationships between two variables.
Alcohol vs Quality
plt.figure(figsize=(8,5))sns.boxplot(    x='quality',    y='alcohol',    data=df)plt.title('Alcohol vs Wine Quality')plt.show()


Volatile Acidity vs Quality
plt.figure(figsize=(8,5))sns.boxplot(    x='quality',    y='volatile.acidity',    data=df)plt.title('Volatile Acidity vs Wine Quality')plt.show()


Sulphates vs Quality
plt.figure(figsize=(8,5))sns.boxplot(    x='quality',    y='sulphates',    data=df)plt.title('Sulphates vs Wine Quality')plt.show()


Alcohol vs Density
plt.figure(figsize=(8,5))sns.scatterplot(    x='alcohol',    y='density',    data=df)plt.title('Alcohol vs Density')plt.show()


6️⃣ Correlation Analysis
Correlation analysis was performed to understand relationships between numerical variables.
plt.figure(figsize=(12,8))sns.heatmap(    df.corr(),    annot=True,    cmap='coolwarm')plt.title('Correlation Heatmap')plt.show()


The correlation analysis helped identify:
- Positive relationships
- Negative relationships
- Weak relationships
- Stronger relationships between variables
7️⃣ Outlier Detection
Potential outliers were examined using boxplots and the IQR method.
Q1 = df.quantile(0.25)Q3 = df.quantile(0.75)IQR = Q3 - Q1lower = Q1 - 1.5 * IQRupper = Q3 + 1.5 * IQR


Boxplots were used to visually identify potential outliers across numerical variables.
8️⃣ Target Variable Analysis
The target variable in this project is:
quality

The quality scores range from:
3 to 8

The distribution was analyzed using:
df['quality'].value_counts().sort_index()


A visualization was created using:
plt.figure(figsize=(8,5))sns.countplot(    x='quality',    data=df)plt.title('Wine Quality Distribution')plt.xlabel('Quality')plt.ylabel('Count')plt.show()


The dataset has an imbalanced quality distribution, with quality scores 5 and 6 occurring more frequently than several other quality levels.
9️⃣ Machine Learning
Feature and Target Separation
The input variables and target variable were separated:
X = df.drop('quality', axis=1)y = df['quality']


Where:
X = Input Features
y = Target Variable

🔟 Train-Test Split
The dataset was divided into training and testing data.
from sklearn.model_selection import train_test_splitX_train, X_test, y_train, y_test = train_test_split(    X,    y,    test_size=0.20,    random_state=42,    stratify=y)


Dataset Split
Training Data: 80%
Testing Data : 20%

1️⃣1️⃣ Handling Class Imbalance Using SMOTE
The quality target has an imbalanced class distribution.
SMOTE was applied to the training data:
from imblearn.over_sampling import SMOTEsmote = SMOTE(random_state=42)X_train_smote, y_train_smote = smote.fit_resample(    X_train,    y_train)


SMOTE was applied only to the training data to avoid data leakage into the test dataset.
1️⃣2️⃣ Logistic Regression
Logistic Regression was used as one of the classification models.
from sklearn.linear_model import LogisticRegressionlr = LogisticRegression(    max_iter=1000)lr.fit(    X_train_smote,    y_train_smote)y_pred_lr = lr.predict(X_test)


1️⃣3️⃣ Random Forest Classifier
Random Forest was used as another classification model.
from sklearn.ensemble import RandomForestClassifierrf = RandomForestClassifier(    n_estimators=200,    random_state=42)rf.fit(    X_train_smote,    y_train_smote)y_pred_rf = rf.predict(X_test)


1️⃣4️⃣ Model Evaluation
The machine learning models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix
Example:
from sklearn.metrics import (    accuracy_score,    classification_report,    confusion_matrix)print(    "Accuracy:",    accuracy_score(y_test, y_pred_rf))print(    classification_report(        y_test,        y_pred_rf    ))


1️⃣5️⃣ Confusion Matrix
A confusion matrix was created to understand classification performance across the wine quality classes.
cm = confusion_matrix(    y_test,    y_pred_rf)plt.figure(figsize=(8,6))sns.heatmap(    cm,    annot=True,    fmt='d',    cmap='Blues')plt.xlabel('Predicted')plt.ylabel('Actual')plt.title('Random Forest Confusion Matrix')plt.show()


1️⃣6️⃣ Feature Importance
Random Forest feature importance was used to identify variables that contributed to the trained model.
feature_importance = pd.DataFrame({    'Feature': X.columns,    'Importance': rf.feature_importances_})feature_importance = feature_importance.sort_values(    by='Importance',    ascending=False)print(feature_importance)


Some of the important features observed included:
Alcohol
Sulphates
Total Sulfur Dioxide
Volatile Acidity
Density
Chlorides
pH

Feature importance indicates the contribution of variables to the trained Random Forest model. It does not by itself establish a causal relationship.
📊 Interactive Dashboard
An interactive dashboard was created using Plotly.
The dashboard contains:
KPI Metrics
- Total Samples
- Average Quality
- Quality Range
Visualizations
- Wine Quality Distribution
- Alcohol by Quality
- Alcohol vs Quality
- Correlation Heatmap
- Random Forest Feature Importance
📈 Key Findings
🍷 Wine Quality
Wine quality scores range from 3 to 8.
Quality scores 5 and 6 represent a large portion of the dataset.
🧹 Data Quality
- No missing values were identified.
- 240 duplicate records were identified.
- 240 duplicate records were removed.
- Final dataset contains 1,359 unique records.
🍺 Alcohol
Alcohol showed an important relationship with wine quality during exploratory analysis and was also among the more important Random Forest features.
🧪 Volatile Acidity
Volatile acidity showed a generally negative relationship with wine quality during exploratory analysis.
📊 Feature Importance
Random Forest identified several variables as important for classification, including:
Alcohol
Sulphates
Total Sulfur Dioxide
Volatile Acidity
Density

📁 Project Structure
Red-Wine-Quality-Analysis/
│
├── data/
│   └── wineQualityReds.csv
│
├── notebooks/
│   └── Red_Wine_Quality_Analysis.ipynb
│
├── dashboard/
│   ├── red_wine_final_dashboard.html
│   └── red_wine_feature_importance.html
│
├── presentation/
│   └── Red_Wine_Quality_Analysis_Presentation.pptx
│
├── images/
│   ├── wine_quality_distribution.png
│   ├── correlation_heatmap.png
│   ├── alcohol_vs_quality.png
│   └── feature_importance.png
│
├── requirements.txt
└── README.md

⚙️ Installation
Clone the repository:
git clone https://github.com/OBULESH597/Red-Wine-Quality-Analysis.git

Move into the project folder:
cd Red-Wine-Quality-Analysis

Install the required libraries:
pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn jupyter

▶️ How to Run
Google Colab
1. Open Google Colab.
2. Upload the project notebook.
3. Upload wineQualityReds.csv.
4. Run the notebook cells sequentially.
Jupyter Notebook
Run:
jupyter notebook

Then open:
Red_Wine_Quality_Analysis.ipynb

📦 requirements.txt
Create a requirements.txt file containing:
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
imbalanced-learn
jupyter

📊 Skills Demonstrated
Data Analytics
- Data Cleaning
- Data Preprocessing
- Exploratory Data Analysis
- Statistical Analysis
- Univariate Analysis
- Bivariate Analysis
- Correlation Analysis
- Outlier Detection
- Data Visualization
Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
Machine Learning
- Train-Test Split
- SMOTE
- Logistic Regression
- Random Forest
- Classification
- Model Evaluation
- Confusion Matrix
- Feature Importance
💡 Analytical Perspective
This project demonstrates how structured data can be used to:
- Explore wine quality patterns
- Identify relationships between chemical properties
- Analyze important variables
- Prepare data for machine learning
- Handle class imbalance
- Compare machine learning models
- Communicate analytical findings through visualizations
- Build an interactive dashboard
🚀 Future Improvements
Possible future improvements include:
- Hyperparameter tuning
- Cross-validation
- Additional machine learning algorithms
- Feature engineering
- Model optimization
- More interactive dashboard filters
- Web application deployment
- Additional model evaluation metrics
- Alternative approaches to wine quality classification
📚 Project Deliverables
This project includes:
- 📓 Data Analysis Notebook
- 🧹 Data Cleaning
- 📊 Exploratory Data Analysis
- 📈 Data Visualizations
- 🔥 Correlation Analysis
- 📦 Outlier Analysis
- 🤖 Machine Learning Models
- ⚖️ SMOTE Class Balancing
- 📊 Feature Importance
- 📉 Confusion Matrix
- 📊 Interactive Plotly Dashboard
- 📑 PowerPoint Presentation
- 📝 Project Documentation
👨‍💻 Author
Obulesu Polisetti
B.Tech Computer Science & Engineering Graduate | Data Analyst Aspirant
Technical Skills
SQL
Python
Pandas
NumPy
Microsoft Excel
Power BI
Matplotlib
Seaborn
Plotly
Data Cleaning
Data Analysis
Data Visualization
Machine Learning

🔗 Connect With Me
LinkedIn
https://linkedin.com/in/obulesu-polisetti-984b0236b
GitHub
https://github.com/OBULESH597
Portfolio
https://obulesh597.github.io/PORTFOLIO-DATA-ANALYST/portfolio/
⭐ Support
If you find this project useful or interesting, please consider giving this repository a ⭐.
Thank you for visiting my project! 🍷📊
🏷️ Tags
#DataAnalytics #DataAnalyst #Python #Pandas #NumPy #EDA #DataVisualization #MachineLearning #ScikitLearn #RandomForest #SMOTE #Plotly #DataScience #GitHub #Analytics #PythonProject
