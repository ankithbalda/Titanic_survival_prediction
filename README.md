# 🚢 Titanic Survival Prediction – Machine Learning Project

A machine learning project that predicts whether a passenger survived the Titanic disaster based on features such as age, gender, passenger class, and more.  
This is based on the **famous Kaggle Titanic challenge**.

---

## 📌 Project Overview

- **Goal:** Predict passenger survival (0 = No, 1 = Yes) using classification models.
- **Dataset:** Provided by [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- **Approach:**
  - Data analysis & visualization
  - Feature engineering
  - Model training and evaluation (Logistic Regression)
  - Generating submission file for Kaggle

---

## 📂 Project Structure

📁 titanic-survival-prediction/

├── train.csv # Training dataset

├── test.csv # Test dataset

├── gender_submission.csv # Sample submission (Kaggle format)

├── Survival_prediction.ipynb # Jupyter notebook with full analysis & modeling

├── submission.csv # Your model's predictions (generated)

└── README.md # Documentation


---

## 📊 Dataset Details

- **train.csv** — Passenger details + target (`Survived`) for training.
- **test.csv** — Passenger details (without `Survived`) for which predictions are required.
- **gender_submission.csv** — Example format for submitting predictions to Kaggle.

**Key Features:**
- `Pclass` – Passenger ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Sex` – Gender
- `Age` – Age in years (some missing values)
- `SibSp` – # of siblings/spouses aboard
- `Parch` – # of parents/children aboard
- `Fare` – Ticket fare
- `Embarked` – Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---

## 🛠 Tools & Libraries

- **Python 3**
- **Pandas** – Data manipulation
- **NumPy** – Numerical operations
- **Matplotlib / Seaborn** – Visualization
- **Scikit‑Learn** – Machine learning models

---

## 🚀 Steps Performed

1. **Data Loading**
   - Read train & test datasets with Pandas.

2. **Exploratory Data Analysis (EDA)**
   - Missing values inspection.
   - Survival rate visualizations by gender, passenger class, age groups, etc.

3. **Data Cleaning & Preprocessing**
   - Fill missing ages with median.
   - Fill missing embarked values with most frequent.
   - Convert categorical variables (`Sex`, `Embarked`) into numeric format.
   - Feature scaling if required.

4. **Model Training**
   - Logistic Regression classifier from Scikit‑Learn.
   - Train on processed training data.

5. **Model Evaluation**
   - Display training accuracy.
   - Validate on a hold‑out set (if split in notebook).

6. **Prediction & Submission**
   - Predict on `test.csv`.
   - Save results in `submission.csv` in Kaggle format.

---

## 📈 Example Accuracy

*Training Accuracy:* ~80% (may vary depending on preprocessing)

---

## 📦 Installation & Usage

1. **Clone the Repository**
git clone https://github.com/ankithbalda/Titanic_survival_prediction.git

cd Titanic_survival_prediction

2. **Install Dependencies**
pip install pandas numpy matplotlib seaborn scikit-learn

3. **Run the Notebook**
- Open `Survival_prediction.ipynb` in Jupyter Notebook or Jupyter Lab.
- Execute cells step‑by‑step to train the model and generate predictions.

4. **Generate Submission**
- The notebook saves `submission.csv`.
- Upload `submission.csv` to Kaggle for scoring.

---


## 👨‍💻 Author
**Name:** Ankith Balda

**Email:** ankithbalda.wk@gmail.com

**LinkedIn:** https://www.linkedin.com/in/ankith-balda-812177278/

**GitHub:** https://github.com/ankithbalda

---

## ⭐ Acknowledgements

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- Scikit‑Learn documentation

