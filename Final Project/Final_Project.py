"""
Read the diamonds.csv file and describe it.
Make at least 4 different analysis on Exploratory Data Analysis section.
Pre-process the dataset to get ready for ML application. (Check missing data and handle them, can we need to do scaling or feature extraction etc.)
Define appropriate evaluation metric for our case (classification). Hint: Is there any imbalanced problem in the label column?
Split the dataset into train and test set. (Consider the imbalanced problem if is there any). Check the distribution of labels in the subsets (train and test).
Train and evaluate Decision Trees and at least 2 different appropriate algorithm which you can choose from scikit-learn library.
Is there any overfitting and underfitting? Interpret your results and try to overcome if there is any problem in a new section.
Create confusion metrics for each algorithm and display Accuracy, Recall, Precision and F1-Score values.
Analyse and compare results of 3 algorithms.
Select best performing model based on evaluation metric you chose on test dataset.
"""
import numpy as np

import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score
from sklearn.linear_model import LinearRegression

data = pd.read_csv("project.csv");

print(data.shape)

print(data.head())
print(data.info())

print(data.describe())


print(data.duplicated().sum())


print(data.isna().sum())

#öncelikle cut  color ve clearity sıralı veriler oldukları için bu verileri numaralara dönüştürelim
#datalar sıralı olduğu için sklearnde bulduğum OrdinalEncoder Kütüphanesini kullanmayı planlıyorum.
obj_df = data.select_dtypes(include=['object']).copy()
enc = OrdinalEncoder()
enc.fit(obj_df)

print(enc.categories_)



cleanup_nums = {
    "cut":     {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5},
    "color": {"D": 7, "E": 6, "F": 5, "G": 4, "H": 3, "I": 2, "J":1 },
    "clarity": {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8 },
    "price": {"Very Low": 1, "Low": 2, "Medium": 3, "High": 4, "Very High": 5 }
    }


datanumerized = data.replace(cleanup_nums)
print(datanumerized.head())

print(datanumerized.info())

clases = list(set(data.price))

print(clases)

print(datanumerized["price"].value_counts())

# verylowun çok yüksek olması ve very high datanın çok düşük olması nedeniyle bir imbalanced data sorunu var diyebiliriz.

# ben projeyi bitirmede geç kaldığım için daha hızlı model eğitme adına downsample yöntemini denicem ve tüm dataları very higha yani 1589 a eşitlicem;

df_verylow = datanumerized[datanumerized.price==1]
df_low = datanumerized[datanumerized.price==2]
df_medium = datanumerized[datanumerized.price==3]
df_high = datanumerized[datanumerized.price==4]
df_veryhigh = datanumerized[datanumerized.price==5]

print(df_verylow.shape)

df_verylow_dsample = resample(df_verylow, replace= False, n_samples=1589, random_state=42)
df_low_dsample = resample(df_low, replace= False, n_samples=1589, random_state=42)
df_medium_dsample = resample(df_medium, replace= False, n_samples=1589, random_state=42)
df_high_dsample = resample(df_high, replace= False, n_samples=1589, random_state=42)

df_balanced = pd.concat([df_verylow_dsample, df_low_dsample, df_medium_dsample, df_high_dsample, df_veryhigh])
print(df_balanced["price"].value_counts())
print(df_balanced.head())


df_balanced.drop(["index"], axis=1, inplace=True)
X, y = df_balanced.iloc[: , :-1], df_balanced.iloc[: , -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train,y_train)
print("Accuracy of train:",clf.score(X_train,y_train))
print("Accuracy of test:",clf.score(X_test,y_test))


## birde xyz değerleri arasında bir bağlantı olduğu için xyz değeri olmadan denemek istiyorum.
df_balanced.drop(["x", "y", "z"], axis=1, inplace=True)
X_new, y_new = df_balanced.iloc[: , :-1], df_balanced.iloc[: , -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train,y_train)
print("Accuracy of train:",clf.score(X_train,y_train))
print("Accuracy of test:",clf.score(X_test,y_test))

#Önemli bir değişkilik olmadığı için xyz değeri olmadan devam etmeye karar verdim


# HyperTuning

param_dict = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2),
    'learning_rate': [0.00001, 0.001, 0.01, 0.1, 1, 2],
    'n_estimators': [10, 190, 200, 210, 500, 1000, 2000]

}
"""
xgc = XGBClassifier(booster='gbtree', learning_rate=0.01, n_estimators=200, max_depth=5,
                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                    objective='multi:softprob', num_class=2, nthread=4, scale_pos_weight=1, seed=27)

clfc = RandomizedSearchCV(xgc, param_dict, cv=3, n_jobs=-1, random_state=42).fit(X_train, y_train)

print("Tuned: {}".format(clfc.best_params_))
#'n_estimators': 190, 'min_child_weight': 1, 'max_depth': 5, 'learning_rate': 0.1}
print("Mean of the cv scores is {:.6f}".format(clfc.best_score_))
#0.852184
print("Train Score {:.6f}".format(clfc.score(X_train, y_train)))
# 0.961338
print("Test Score {:.6f}".format(clfc.score(X_test, y_test)))
# 0.852349
print("Seconds used for refitting the best model on the train dataset: {:.6f}".format(clfc.refit_time_))
#1.143040

"""
model = LinearRegression()
model.fit(X_train, y_train)

r_sq = model.score(X_train, y_train)
print('Coefficient of determination (R2):', r_sq)


r_t = model.score(X_test, y_test)
print('Coefficient of determination (R2) of test:', r_t)


# fiyatların özellikler arttığını düşünürsek classification için kullanabiliriz.


# birde ExtraTreeClassifier  ve RandomForestClassifier kullaıcaktım ama neyazık ki zamanım yetmedi . Saat 23:53.
