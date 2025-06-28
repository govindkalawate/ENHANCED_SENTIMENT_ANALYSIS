
```markdown
# Enhanced Sentiment Analysis of Twitter (X) Data Using an Ensemble Stacking Model

This project proposes an enhanced approach to sentiment analysis using an ensemble stacking model. The model leverages the combined strengths of multiple machine learning classifiers for improved sentiment classification on Twitter (X) data.

---

## 🔍 Abstract

Sentiment analysis has become a critical task in understanding public opinion on platforms like Twitter (now X). This study develops an ensemble stacking model incorporating Random Forest, Logistic Regression, SVM, and XGBoost as base learners, with Logistic Regression as the meta-learner. The model is trained on a real-world tweet dataset, preprocessed using TF-IDF, and evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.

---

## 🧪 Models Used

- **Base Learners**:
  - Random Forest
  - Logistic Regression
  - Support Vector Machine (SVM)
  - XGBoost

- **Meta Learner**:
  - Logistic Regression

---

## 📝 Dataset

- **Total Tweets**: 1,584  
- **Columns**:  
  - `id`: Unique tweet ID  
  - `label`: Sentiment label (0 = Negative, 1 = Positive)  
  - `tweet`: The text content of the tweet  

Data was collected using a keyword-based search strategy and includes diverse opinions on technology, services, and everyday experiences.

---

## ⚙️ Preprocessing Steps

- Removal of URLs, mentions (@user), hashtags (#tag), and emojis  
- Lowercasing text and removing punctuation  
- Tokenization and stopword filtering  
- TF-IDF vectorization to convert text to numerical format  

---

## 📊 Evaluation Metrics

The following metrics were used to evaluate model performance:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC Score  

---

## 📈 Results

| Model               | Accuracy  | Precision | Recall  | F1-Score | ROC-AUC |
|--------------------|-----------|-----------|---------|----------|---------|
| SVM                | 87.71%    | 80.66     | 85.73   | 73.02    | —       |
| Random Forest      | 86.55%    | 81.67     | 85.67   | 78.10    | —       |
| Decision Tree      | 88.51%    | 79.62     | 84.88   | 55.57    | —       |
| **Proposed Model** | **89.71%**| **82.40** | **79.16** | **80.75** | **86.41** |

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Enhanced-Sentiment-Analysis.git
cd Enhanced-Sentiment-Analysis
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Main Script

```bash
python main.py
```

### 4. (Optional) Open Jupyter Notebook

```bash
jupyter notebook notebooks/SentimentAnalysis_Stacking.ipynb
```

---

## 📦 Requirements

Ensure the following packages are installed:

* Python 3.8 or higher
* pandas
* numpy
* scikit-learn
* xgboost
* matplotlib
* seaborn

Install all at once using:

```bash
pip install -r requirements.txt
```

---

## 🔗 IEEE Xplore Info

* **Project Title**: Enhanced Sentiment Analysis of Twitter (X) Data Using an Ensemble Stacking Model
* **Authors**: Govind Kalawate Dr. M.Salomi, Yash Talreja.
* **IEEE Xplore Link**: *\[https://ieeexplore.ieee.org/document/10987415]*

---

## 👨‍💻 Author Info

**Govind Kalawate**
B.Tech in Computer Science & Engineering (Minor in Software Engineering)
SRM Institute of Science and Technology, Chennai
📧 [ga3211@srmist.edu.in](mailto:ga3211@srmist.edu.in)
📍 Shegaon, Maharashtra, India

---

## 📃 License

This project is licensed under the MIT License. You are free to use, share, and modify it with attribution.

---

## 💡 Statement of Purpose

We designed this model to **combine the strengths and complement the weaknesses** of individual classifiers. By leveraging ensemble learning and a robust text preprocessing pipeline, we aim to offer better real-world sentiment classification performance than traditional single-model approaches.


