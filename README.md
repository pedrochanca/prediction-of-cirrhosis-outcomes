# Multi-class Prediction of Cirrhosis Outcomes
Use of different multi-class approaches to predict the the outcomes of patients with cirrhosis.

Reference to the kaggle competition: https://www.kaggle.com/competitions/playground-series-s3e26/

# File Description

train.csv - the training dataset; 
    * Status is the categorical target; C (censored) indicates the patient was alive at N_Days, CL indicates the patient was alive at N_Days due to liver a transplant, and D indicates the patient was deceased at N_Days.

test.csv - the test dataset; 
    * your objective is to predict the probability of each of the three Status values, e.g., Status_C, Status_CL, Status_D.

sample_submission.csv - a sample submission file in the correct format


# To be tested

- Decision Tree
- Random Forest
- XGBBoost
- Catboost
- lightGBM