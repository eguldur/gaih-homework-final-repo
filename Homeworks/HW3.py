#Explain your work
"""
Generate dataset using make_classification function in the sklearn.datasets class. Generate 10000 samples with 8 features (X) with one label (y). Also, use following parameters
n_informative = 5
class_sep = 2
random_state = 42
Explore and analyse raw data.
Do preprocessing for classification.
Split your dataset into train and test test (0.7 for train and 0.3 for test).
Try Decision Tree and XGBoost Algorithm with different hyperparameters. (Using GridSearchCV is a plus)
Evaluate your result on both train and test set. Analyse if there is any underfitting or overfitting problem. Make your comments.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.rcParams['figure.figsize'] = (20.0, 10.0)
pd.set_option("display.max_rows", None, "display.max_columns", None)

X, y = make_classification(n_samples=10000, n_features=8, n_informative=5, n_classes=2, class_sep=2.0, random_state=42)
print(X.shape)
# 8 Adet Featuredan oluşan 10000 satırlık bir veri seti olduğunu söyleyebiliriz.
df = pd.DataFrame(X)

df_graph = pd.DataFrame(X)
df_graph['Class'] = y
# Grafik oluşturmayı kolaylaştırmak için dataset içine class sütunu ekledim


print(df.head())
print(df.info())

# Featureları incelediğimizde, tümünün devamlı veri olduğunu söyleyebiliriz.

print(df.describe())

# ilk çeyrek ve son çeyrek değerlerine outliersı temizlemeyi deneyebiliriz.

print(df.duplicated().sum())

# Tekrar eden verinin olmadığını söyleyebiliriz

print(df.isna().sum())
# Boş verinin olmadığını söyleyebiliriz



sns.pairplot(df_graph, hue="Class")
plt.show()

# Graifği incelediğimde Index3 ve Index5 teki futureların iyi bir başlangıç noktası olacağını söyleyebiliriz. Daha detaylı incelemek için bu iki futureın grafiğini çizdiriyorum;

sns.distplot(df[df_graph.Class == 0][3])
sns.distplot(df[df_graph.Class == 1][3])
plt.show()

sns.distplot(df[df_graph.Class == 0][5])
sns.distplot(df[df_graph.Class == 1][5])
plt.show()


# train ve test data oalrak datamızı bölelim;

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clfa = DecisionTreeClassifier( random_state=42)
clfa.fit(X_train,y_train)
print("Accuracy of train:",clfa.score(X_train,y_train))
#1
print("Accuracy of test:",clfa.score(X_test,y_test))
#0.97433

#Bias for training = 1-1 = 0
#Bias for test = 1-0.974 = 0.026

#Variance = test bias - training bias = 0.026


#burde bir overfitting problemi olabiliceğini düşünebiliriz. Bu yüzden max_depth paremetresi ekleyerek ağacımızı budayabiliriz.

clfb = DecisionTreeClassifier(max_depth=4, random_state=42)
clfb.fit(X_train,y_train)
print("Accuracy of train:",clfb.score(X_train,y_train))
#0.963
print("Accuracy of test:",clfb.score(X_test,y_test))
#0.9603333333333334

clfc = DecisionTreeClassifier(max_depth=8, random_state=42)
clfc.fit(X_train,y_train)
print("Accuracy of train:",clfc.score(X_train,y_train))
#0.988
print("Accuracy of test:",clfc.score(X_test,y_test))
#0.982
#hem variance hemde bias düşük olduğu için max_depth=8 kullanmaya karar verdim.

pred = clfc.predict(X_test)
print(classification_report(y_test,pred))


print("Precision = {}".format(precision_score(y_test, pred, average='macro')))
print("Recall = {}".format(recall_score(y_test, pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, pred)))
print("F1 Score = {}".format(f1_score(y_test, pred,average='macro')))

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(12, 8))
ax =sns.heatmap(cm, square=True, annot=True, cbar=False)
ax.xaxis.set_ticklabels([0,1], fontsize = 12)
ax.yaxis.set_ticklabels([0,1], fontsize = 12, rotation=0)
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
plt.show()

param_dict = {
    'max_depth': range(3, 8, 2),
    'min_child_weight': range(1, 5, 2),
    'learning_rate': [0.00001, 0.001, 0.01, 0.1],
    'n_estimators': [10, 190, 200, 210, 500]

}

xgc = XGBClassifier(booster='gbtree', learning_rate=0.01, n_estimators=200, max_depth=5,
                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                    objective='multi:softprob', num_class=5, nthread=4, scale_pos_weight=1, seed=27)

clfc = GridSearchCV(xgc, param_dict, cv=3, n_jobs=-1).fit(X_train, y_train)

print("Tuned: {}".format(clfc.best_params_))
#{'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 500}
print("Mean of the cv scores is {:.6f}".format(clfc.best_score_))
#0.985571
print("Train Score {:.6f}".format(clfc.score(X_train, y_train)))
#1.000000
print("Test Score {:.6f}".format(clfc.score(X_test, y_test)))
#0.990667
print("Seconds used for refitting the best model on the train dataset: {:.6f}".format(clfc.refit_time_))
# 2.412217

