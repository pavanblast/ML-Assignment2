a. Problem statement:
Ans: The dataset contains some parameters which decide whether a fetal health is normal, suspect and pathologocal based
     on the set of features provided, we have to predict the fetal health of new incoming infant when input features are given.

b. Dataset description [ 1 mark ]
Ans: Input features: 21
     features: ['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 'abnormal_short_term_variability',
       'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min', 'histogram_max',
       'histogram_number_of_peaks', 'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance', 'histogram_tendency']
       Output feature: fetal_health


c. Models used: [ 6 marks - 1 marks for all the metrics for each model ] Make a Comparison Table with the evaluation metrics calculated for all the 6 models as below:
Ans:
    ## 📊 Model Performance Comparison

    | ML Model | Accuracy | ROC AUC | Precision | Recall | F1 Score | MCC |
    |---------|---------|---------|-----------|--------|---------|------|
    | Logistic Regression | 0.8338 | 0.9174 | 0.8207 | 0.8338 | 0.8249 | 0.5123 |
    | Decision Tree | 0.9366 | 0.9559 | 0.9353 | 0.9366 | 0.9352 | 0.8226 |
    | KNN | 0.8731 | 0.9062 | 0.8661 | 0.8731 | 0.8649 | 0.6253 |
    | Naive Bayes | 0.7825 | 0.9004 | 0.8514 | 0.7825 | 0.8005 | 0.5480 |
    | Random Forest (Ensemble) | 0.9456 | 0.9855 | 0.9445 | 0.9456 | 0.9446 | 0.8483 |
    | **XGBoost (Ensemble)** | **0.9637** | **0.9884** | **0.9637** | **0.9637** | **0.9636** | **0.9002** |


- Add your observations on the performance of each model on the chosen dataset. [ 3 marks ]

    | **ML Model Name**            | **Observation about Model Performance**                                          |
    | ---------------------------- | -------------------------------------------------------------------------------- |
    | **Logistic Regression**      | Performs well on linear data but gives lower accuracy for complex patterns.      |
    | **Decision Tree**            | Fits the training data well but may overfit on unseen data.                      |
    | **kNN**                      | Works well for small datasets but is sensitive to noise and slow for large data. |
    | **Naive Bayes**              | Fast and simple but less accurate when features are correlated.                  |
    | **Random Forest (Ensemble)** | Provides higher accuracy and better generalization than a single tree.           |
    | **XGBoost (Ensemble)**       | Delivers the best performance by learning complex relationships efficiently.     |
