import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#eksik veriler

veriler = pd.read_csv('eksikveriler.txt')
#pd.read_csv("eksikveriler.txt")

print(veriler)

boy = veriler[['boy']]
print(boy)

boykilo= veriler[['boy','kilo']]
print(boykilo)

class Insan:
    boy = 180
    def kosmak(self,b):
        return b + 10
    
ali = Insan()
print(ali.boy)
print(ali.kosmak(90))    

#eksik veriler
#sci - kit learn
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Yas = veriler.iloc[:, 1:4].values
print(Yas)
Yas[:, 1:4] = imputer.fit_transform(Yas[:, 1:4])
print(Yas)

ulke = veriler.iloc[:, 0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:, 0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


sonuc = pd.DataFrame(data=ulke, index=range(22), columns = ['fr', 'tr', 'us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy', 'kilo', 'yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:, -1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index = range(22), columns = ['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)


    
