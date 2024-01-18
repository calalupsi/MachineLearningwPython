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

ulke = veriler.iloc[:, 0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:, 0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

print(Yas)
Yas[:, 1:4] = imputer.fit_transform(Yas[:, 1:4])
print(Yas)

