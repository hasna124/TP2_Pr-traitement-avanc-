##Partie 1 — Explorer et vérifier les données
##Chargez le fichier CSV avec pandas.
import pandas as pd
df=pd.read_csv(r"C:\Users\dell\Downloads\transactions_TP2.csv")
## Affichez les cinq premières lignes
print(df.head())
## le nombre de lignes et de
colonnes
print(df.shape)
# les types de variables.
print(df.dtypes)
##le nombre et le pourcentage de valeurs manquantes
manquant=df.isna().sum()
print(manquant)
pourcentage=df.isna().mean()*100
print(pourcentage)
T= pd.DataFrame({'manquant': manquant , 'pourcentage': pourcentage})
T=T.sort_values('pourcentage',ascending=False)
print(T)
##Combien de doublons trouvez-vous?
print("nombres de doublons est:",df["id_operation"].duplicated().sum())
##le nombre de lignes restantes.
nbr_doub=df["id_operation"].duplicated().sum()
df=df.drop_duplicates(subset="id_operation")
print("le nombre de lignes restantes:",len(df))
##un histogramme du montant 
import  matplotlib.pyplot as plt 
import seaborn as sns
plt.figure(figsize=(8,5))
sns.histplot(df["montant_dh"])
plt.title("histogramme du montant")
plt.show()

##boxplot 
plt.figure(figsize=(8,5))
sns.boxplot(x=df["montant_dh"])
plt.title("Montant d' operation")
plt.show()
## la proportion de fraudes 
print(df["fraude"].value_counts())
print( "la proportion:",df["fraude"].value_counts(normalize=True)*100)
##un diagramme des deux classes
df['fraude'].value_counts().plot(kind='bar')
plt.title("Répartition des classes")
plt.show()
##Partie2:Préparer les données sans fuite

Y=df["fraude"]
X=df.drop(columns=["fraude","id_operation","id_client","resultat_enquete"])
##séparer les données d'entrainement et les données de test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,stratify=Y,random_state=42)
## Lister les variables numerique  
cols_numr=X.select_dtypes(include=["int64","float64"]).columns
cols_catego=X.select_dtypes(include=["object"]).columns
print ("les colonnes numerique sont:",list(cols_numr))
print("les colonnes calégorie sont:",list(cols_catego)) 

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
num_pepline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
     ("scaler",StandardScaler())
 ])
 ## imputation par la valeur la plus fréquente, puis OneHotEncoder(handle_unknown="ignore").

rom sklearn.preprocessing import OneHotEncoder
cat_pipeline=Pipeline([
('imputer',SimpleImputer(strategy='most_frequent')),
('onehot',OneHotEncoder(handle_unknown="ignore"))
])
##Transformez les groupes de référence et de vérification
from sklearn.compose import ColumnTransformer
prep=ColumnTransformer(
    transformers=[
        ("categorie",cat_pipeline,cols_catego),
        ("numeric", num_pepline,cols_numr)
    ])

x_train_n=prep.fit_transform(x_train)
x_test_n=prep.transform(x_test)
print(x_train_n.shape)
print(x_test_n.shape)
##Partie 3 — Comparer les méthodes de mise à l’échelle

from sklearn.preprocessing import RobustScaler
Robust_pipline=Pipeline([("imputer",SimpleImputer(strategy='median')),
("scaler",RobustScaler())])
x_train_r=Robust_pipline.fit_transform(x_train[cols_numr])
x_test_r=Robust_pipline.transform(x_test[cols_numr])


q1=x_train["montant_dh"].quantile(0.25)
q3=x_train["montant_dh"].quantile(0.75)
IQR=q3-q1

print("Q1=",q1)
print("Q3=",q3)
print("IQR=",IQR)
##Combien de lignes sont concernées
L=q1-1.5*IQR
U=q3+1.5*IQR
med_remp=x_train["montant_dh"].median()
x_m=x_train[(x_train["montant_dh"]<L )|( x_train["montant_dh"]>U)]
print("median est",med_remp)
print("L=",L)
print("U=",U)
print(len(x_m))
##tracer deux boxplot
x_train_cl=x_train.copy()
x_train_cl['montant_dh']=x_train_cl['montant_dh'].clip(lower=L,upper=U)
plt.figure(figsize=(11,5))
plt.subplot(1,2,1)
sns.boxplot(x=x_train['montant_dh'])
plt.title ("conserver les montants extrêmes")

plt.subplot(1,2,1)
sns.boxplot(x=x_train_cl["montant_dh"])
plt.title("aprés limitation des valeurs  extremes")
plt.show()


##appliquez séparément MinMaxScaler, StandardScaler et RobustScaler

from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaleer={}
Result_train={}
for n,scaler in[
 ('scaler_mM',MinMaxScaler()),('scaler_std',StandardScaler()),('scaler_robu',RobustScaler())]:
    x_train_sca=scaler.fit_transform(x_train[['montant_dh']])
    x_test_sca=scaler.transform(x_test[['montant_dh']])
    Result_train[n]=scaler.transform(x_train[['montant_dh']])
    scaleer[n]=scaler

montant_test=pd.DataFrame({'montant_dh':[100,150,600,33281.230]})
tableau=pd.DataFrame({
    'Monatant':montant_test['montant_dh'],
    'MinMaxScaler':scaleer['scaler_mM'].transform(montant_test).ravel(),
    'StandardScaler':scaleer['scaler_std'].transform(montant_test).ravel(),
    'RobustScaler':scaleer['scaler_robu'].transform(montant_test).ravel()
})
print(tableau)
###
import matplotlib.pyplot as plt
plt.figure(figsize=(13,2))
#histogramme MnMaxScaler
plt.subplot(1,3,1)
plt.hist(Result_train['scaler_mM'],bins=50)
plt.title("MinMaxScaler")
plt.xlabel("Valeur transformé")

#histogramme standarisation
plt.subplot(1,3,2)
plt.hist(Result_train['scaler_std'],bins=50)
plt.title("StandardScaler")
plt.xlabel("Valeur transformé")
#histogramme de RobustScaler
plt.subplot(1,3,3)
plt.hist(Result_train['scaler_robu'],bins=50)
plt.title("RobustScaler")
plt.xlabel("Valeur transformé")

plt.show()

##Partie 4 — Préparer des classes déséquilibrées
##Calculez le rapport entre les deux classes.
print(y_train.value_counts())
print( "la proportion:",y_train.value_counts(normalize=True)*100)

##RandomOverSample
from imblearn.over_sampling import RandomOverSampler
Over=RandomOverSampler(random_state=True)
x_Over,y_Over=Over.fit_resample(x_train,y_train)
print(y_train.value_counts(),y_Over.value_counts())


##RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler

under=RandomUnderSampler(random_state=True)
x_un,y_un=under.fit_resample(x_train,y_train)
print(y_train.value_counts(),y_un.value_counts())
##un petit exemple
import numpy as np
X=[16,20]
Y=[20,60]
coef=0.6
syst=[]
for i in range (len(X)):
 valr_synt=X[i]+coef*(Y[i]-X[i])
 syst.append(valr_synt)
print( "valeur synthétique :",syst)

##un tableau les effectifs des classes
Tableau_compa=pd.DataFrame({
    'Avant':y_train.value_counts(),
    'Aprés(OverSampling)':y_Over.value_counts(),
    'Aprés(Undersampling)':y_un.value_counts()
})
print(" tableau les effectifs des classes ")
print(Tableau_compa)

##Partie 5 — Vérifier et enregistrer les données préparées
## une version finale de ColumnTransformer 
num_pepline_end=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
     ("scaler",RobustScaler())
 ])
cat_pepline_end=Pipeline([
('imputer',SimpleImputer(strategy='most_frequent')),
('encode',OneHotEncoder(handle_unknown="ignore"))
])
prepro_final=ColumnTransformer([
     ("numer",num_pepline_end,cols_numr),
     ("catego",cat_pepline_end,cols_catego)
 ])

x_train_end=prepro_final.fit_transform(x_train)
x_test_end=prepro_final.transform(x_test)

print(x_train_end.shape)
print(x_test_end.shape)

import numpy as np
print(np.isnan(x_train_end).sum())
print(np.isnan(x_test_end).sum())
##Travail complémentaire


