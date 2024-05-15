#
#
#
KAGGLE = False
CATEGORICAL_TO_NUMERICAL = True
ENCODE_LABEL = True
NUMERICAL_SCALING = True

numerical_features = [
    "N_Days",
    "Age",
    "Bilirubin",
    "Cholesterol",
    "Albumin",
    "Copper",
    "Alk_Phos",
    "SGOT",
    "Tryglicerides",
    "Platelets",
    "Prothrombin",
    "Stage",
]

categorical_features = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Edema", "Spiders"]

label_order = ["C", "CL", "D"]
