# Multi-class Prediction of Liver Cirrhosis Outcomes
Use of different multi-class approaches to predict the survival of patients with liver cirrhosis.

Reference to the kaggle competition: https://www.kaggle.com/competitions/playground-series-s3e26/

#### **Dataset Description**

The datasets used for this competition (train and test) were generated from a deep learning model trained on the [original dataset](https://www.kaggle.com/datasets/joebeachcapital/cirrhosis-patient-survival-prediction/data).

Note by organzier: Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

#### **Files**
- cirrhosis.csv : the original dataset; this data was sourced from a Mayo Clinic study on primary biliary cirrhosis (PBC) of the liver carried out from 1974 to 1984.

- train.csv : the train dataset; the column "Status" represents the categorical target; the "Status" column assumes the following values: D, CL and C.
    - D : indicates the patient was deceased at N_Days.
    - CL : indicates the patient was alive at N_Days due to liver a transplant.
    - C : indicates the patient was alive at N_Days.

- test.csv : the test dataset; the objective is to predict the probability of each of the three "Status" values, e.g., Status_C, Status_CL, Status_D.

- sample_submission.csv : a sample submission file in the correct format.


#### **Algorithms tested so far**

- CatBoostClassifier
- Softmax Neural Network

- *to be tested: lightGBM, XGBBoost*