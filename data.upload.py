
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#kodlar
#verı yukleme

veriler = pd.read_csv('veriler.txt')
#pr.read_csv("veriler.csv")

print(veriler)

#verı on isleme
boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)

x = 10

class insan:
    boy = 180
    
    def kosmak(self,b):
        return b + 10
    # y = f(x)
    # f(x) = x + 10
    
ali = insan()
print(ali.boy)
print(ali.kosmak(90))
    
l = [1,3,4] #liste

