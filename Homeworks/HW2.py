#Explain your work


#Import Boston Dataset from sklearn dataset class.
#Explore and analyse raw data.
#Do preprocessing for regression.
#Split your dataset into train and test test (0.7 for train and 0.3 for test).
#Try Ridge and Lasso Regression models with at least 5 different alpha value for each.
#Evaluate the results of all models and choose the best performing model.

#Question 1
import numpy as np
from scipy import stats

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso

import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)


import seaborn as sns
import matplotlib.pyplot as plt


def adj_r2 (X,y,model):
    """
    X: input
    y: output
    model: regression model
    """
    r_squared = model.score(X,y)
    return(1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1))



Xb,yb =load_boston(return_X_y=True)

df_boston = pd.DataFrame(Xb,columns = load_boston().feature_names)
print(Xb.shape)
# 13 Adet Featuredan oluşan 506 satırlık bir veri seti olduğunu söyleyebiliriz.
print(df_boston.head())
print(df_boston.info())

# Featureları incelediğimizde, tümünün devamlı veri olduğunu söyleyebiliriz.

print(df_boston.describe())

# ilk çeyrek ve son çeyrek değerlerine bakarak CRIM sütununda bir outliers durumu sözkonusu diyebiliriz.

print(df_boston.duplicated().sum())

# Tekrar eden verinin olmadığını söyleyebiliriz

print(df_boston.isna().sum())

# Boş verinin olmadığını söyleyebiliriz
plt.figure(figsize=(100, 100))
sns.pairplot(df_boston)



corr = df_boston.corr()

plt.figure(figsize=(14, 14))
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True, annot = True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
ax.set_ylim(len(corr)+0.5, -0.5);
plt.show()

#Isı Grafiğine göre dis indus age nox ve düşükte olsa rm featurenın bazı featurelar ile ters yönlü bir ilişikisi olduğunu söyleyebiliriz.


#Öncelikle veri seti üzerinde işlem yapmadan model ve test denemesi yapalım.

X_train, X_test, y_train, y_test = train_test_split(Xb,yb, test_size=0.3, random_state=301)

modela = LinearRegression(normalize=False)
modela.fit(X_train,y_train)

print("Score of the train set",modela.score(X_train,y_train))
#0.7846011639863926
print("Score of the test set",modela.score(X_test,y_test))
#0.580378038900507


print("Adj. R2 of the train set",adj_r2(X_train,y_train,modela))
#0.7763653261388135
print("Adj. R2 of the test set",adj_r2(X_test,y_test,modela))
#0.5408484338693953


#Geliştirdiğimiz modelin featurelarının önem seviyesine bakalım

importance = modela.coef_
for i in range(len(importance)):
    print("Feature", df_boston.columns[i], "Score:", importance[i])

#Çıkan sonuca göre INDUS  0.006578555090487692 ve AGE -0.010820835184929944 önemlerinin düşük olduğunu söyleyebiliriz. Şimdi bu verileri datamızdan temizleyerek tekrar test edelim


new_df = df_boston.drop(["AGE","INDUS"],axis=1)

X_train, X_test, y_train, y_test = train_test_split(new_df,yb, test_size=0.3, random_state=301)
modelb = LinearRegression(normalize=True)
modelb.fit(X_train,y_train)
print("Score of the train set",modelb.score(X_train,y_train))
# 0.7845290321246626
print("Score of the test set",modelb.score(X_test,y_test))
# 0.581135821919665


print("Adj. R2 of the train set",adj_r2(X_train,y_train,modelb))
# 0.7775986793567424
print("Adj. R2 of the test set",adj_r2(X_test,y_test,modelb))
# 0.5482250650704958

#bu işlemin modelimiizin train data scorunu düşürüken test data skorunu üzerinde büyük bir değişiklik yapmadığınızı söyleyebiliriz.

#Birde age ve indus olmayan veri setimiz için Z scorea göre bir outlier detection deneyelim


z = np.abs(stats.zscore(new_df))
print(len(np.where(z > 3)[0]))
#100 Adet Outlierımız var


outliers = list(set(np.where(z > 3)[0]))
new_df_wooutliers = new_df.drop(outliers,axis = 0).reset_index(drop = False)
print(new_df_wooutliers.shape)
#415 Satır ve 12 sütünluk bir veri seti oluşturduk şimdi bunu deneyelim

y_new = yb[list(new_df_wooutliers["index"])]
X_new = new_df_wooutliers.drop('index', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X_new,y_new, test_size=0.3, random_state=301)
modelc = LinearRegression(normalize=True)
modelc.fit(X_train,y_train)
print("Score of the train set",modelc.score(X_train,y_train))
#  0.7632012188974524
print("Score of the test set",modelc.score(X_test,y_test))
# 0.6776969037516847


print("Adj. R2 of the train set",adj_r2(X_train,y_train,modelc))
# 0.7538314829545458
print("Adj. R2 of the test set",adj_r2(X_test,y_test,modelc))
# 0.6463222660637956

#yeni oluşturuduğumuz veri setinin daha iyi çalıştığını görebiliriz. Bu tam anlamıyla iyi bir model geliştirdik demek olmaz. Sonuç olarak datamızı yeni bir veri seti ve veri ile eğittik. Random state sayısı ile oynarakta bu sonuca varabilirdik

#Şimdide son oluşturduğumuz train ve test verisini kullanarak, ridge ve lasso Regulation ile 10 farklı model oluşturalım ve bu modelleri son modelimizle karşılaştıralım
print(f'\n\n')
print(f'******** At The And ********\n\n')
ridges = [1, 10, 50, 100, 150]
lassos = [0.001, 0.01, 0.1, 1, 10]
print('******** Model ******** ')
print("Adj. R2 of the train set",adj_r2(X_train,y_train,modelc))
print("Adj. R2 of the test set",adj_r2(X_test,y_test,modelc))


for ridge in ridges:
    print(f'*************Ridge Model For {ridge} ************')
    ridge_model = Ridge(ridge)
    ridge_model.fit(X_train, y_train)
    print("Adj. R2 of the train set", adj_r2(X_train, y_train, ridge_model))
    print("Adj. R2 of the test set", adj_r2(X_test, y_test, ridge_model))

for lasso in lassos:
    print(f'*************Lasso Model For {lasso} ************')
    lasso_model = Lasso(lasso)
    lasso_model.fit(X_train, y_train)
    print("Adj. R2 of the train set", adj_r2(X_train, y_train, lasso_model))
    print("Adj. R2 of the test set", adj_r2(X_test, y_test, lasso_model))

#Sonuç olarak en iyi çalışan modelimizi regulazisasyon yapmadığımı model olarak değerlendirebiliriz.


