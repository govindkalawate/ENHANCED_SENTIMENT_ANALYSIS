
📌 ENHANCED SENTIMENT ANALYSIS OF TWITTER (X) DATA USING AN ENSEMBLE STACKING MODEL

This project proposes an enhanced approach to sentiment analysis using an ensemble stacking model.
The model combines the strengths of multiple machine learning classifiers to achieve improved sentiment classification on Twitter (now X) data.

-------------------------------------------------------------------------------
🔍 ABSTRACT

Sentiment analysis is vital for understanding public opinion on social media.
This study presents an ensemble stacking model with four base classifiers —
  - Random Forest
  - Logistic Regression
  - SVM
  - XGBoost
and Logistic Regression as the meta-learner.
The tweets are preprocessed using TF-IDF, and the model is evaluated using accuracy,
precision, recall, F1-score, and ROC-AUC.

-------------------------------------------------------------------------------
🧪 MODELS USED

Base Learners:
  ✔ Random Forest
  ✔ Logistic Regression
  ✔ Support Vector Machine (SVM)
  ✔ XGBoost

Meta Learner:
  ✔ Logistic Regression

-------------------------------------------------------------------------------
📝 DATASET DETAILS

- Total Tweets : 1,584
- Columns      :
    • id     - Unique tweet identifier
    • label  - Sentiment (0 = Negative, 1 = Positive)
    • tweet  - Raw tweet text

Data was collected using a keyword-based search strategy and includes a wide range of topics
including technology, services, and everyday experiences.

-------------------------------------------------------------------------------
⚙️ DATA PREPROCESSING STEPS

1. Remove URLs, mentions (@user), hashtags (#), emojis
2. Convert text to lowercase
3. Remove punctuation and extra whitespaces
4. Tokenize and remove stopwords
5. Apply TF-IDF vectorization to obtain numerical features

-------------------------------------------------------------------------------
📊 EVALUATION METRICS

The models were evaluated using the following:
  • Accuracy
  • Precision
  • Recall
  • F1-Score
  • ROC-AUC Score

-------------------------------------------------------------------------------
📈 PERFORMANCE COMPARISON

| MODEL              | ACCURACY | PRECISION | RECALL  | F1-SCORE | ROC-AUC |
|--------------------|----------|-----------|---------|----------|---------|
| SVM                | 87.71%   | 80.66     | 85.73   | 73.02    |   —     |
| Random Forest      | 86.55%   | 81.67     | 85.67   | 78.10    |   —     |
| Decision Tree      | 88.51%   | 79.62     | 84.88   | 55.57    |   —     |
| Proposed Model     | 89.71%   | 82.40     | 79.16   | 80.75    | 86.41   |

-------------------------------------------------------------------------------
🚀 HOW TO RUN THE PROJECT

Step 1: Clone the repository
  git clone https://github.com/govindkalawate/Enhanced-Sentiment-Analysis.git
  cd Enhanced-Sentiment-Analysis

Step 2: Install Dependencies
  pip install -r requirements.txt

Step 3: Run the main script
  Enhanced-Sentiment-Analysis.py

Step 4 (Optional): Run Jupyter notebook
  jupyter notebook notebooks/SentimentAnalysis_Stacking.ipynb

-------------------------------------------------------------------------------
📦 REQUIREMENTS

  - Python 3.8 or higher
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - matplotlib
  - seaborn

Install using:
  pip install -r requirements.txt

-------------------------------------------------------------------------------
🔗 IEEE XPLORE PUBLICATION

  • Title     : Enhanced Sentiment Analysis of Twitter (X) Data Using an Ensemble Stacking Model
  • Authors   : Govind Kalawate, Dr. M. Salomi, Yash Talreja
  • Published : 07-May-2025
  • DOI Link  : https://ieeexplore.ieee.org/document/10987415

-------------------------------------------------------------------------------
👨‍💻 AUTHOR INFO

  Name         : Govind Kalawate
  Degree       : B.Tech in CSE (Minor in Software Engineering)
  Institution  : SRM Institute of Science and Technology, Chennai,India
  Email        : ga3211@srmist.edu.in

-------------------------------------------------------------------------------
📃 LICENSE

This project is licensed under the MIT License.

-------------------------------------------------------------------------------
💡 STATEMENT OF PURPOSE

We developed this model to combine the strengths and complement the weaknesses of individual classifiers.
By using an ensemble approach and a robust preprocessing pipeline, the proposed model outperforms
traditional single-model sentiment analysis techniques — offering better generalization and robustness
on real-world Twitter (X) data.

-------------------------------------------------------------------------------



