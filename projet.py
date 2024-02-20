"""
PROJET TIDYTUESDAY
Groupe PASSWORDS
"""

########## cd D:/TRAVAIL/ETUDES/UTC/GI04/SY09/Projet/

# ETAPE 0 - IMPORT DES LIBRAIRES

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

# ETAPE 0' - QUELQUES FONCTIONS UTILES

def number_of_alpha_char(string):
    count = 0
    for i in string:
        if(i.isalpha()):
            count=count+1
    return count

def number_of_num_char(string):
    count = 0
    for i in string:
        if(i.isnumeric()):
            count=count+1
    return count

def ratio_num(row):
    alp = row['nb_alpha']
    num = row['nb_num']
    if alp == 0:
        return 0
    elif num == 0:
        return 0
    elif alp > num:
        return num/alp
    elif num > alp:
        return alp/num
    else:
        return 1

########## ETAPE 1 - OUVERTURE ET NETTOYAGE DES DONNÉES

df = pd.read_csv("passwords.csv")

# veut-on considérer la colonne "rank" comme l'index ? si oui :
# df = pd.read_csv("passwords.csv", index_col=0)
# j'aurais tendance à dire que non car le rank peut-être utile dans des viz (il représente la popularité)

df.dropna(axis=0, how='all', inplace=True)

df2 = df.drop([405, 25, 335, 499], axis=0)

########## ETAPE 2 - LA FORME DES DONNÉES

## la taille du tableau
print(df.shape)

## les types de données de chaque colonnes
print(df.dtypes)

## les infos générales
print(df.info())

########## ETAPE 3 - NOUVELLES COLONNES UTILES POUR L'ANALYSE 

## longueur du mot de passe
df['length'] = df.apply(lambda row: len(row['password']), axis=1)

## nombre de caractères alphabétiques
df['nb_alpha'] = df.apply(lambda row: number_of_alpha_char(row['password']), axis=1)

## nombre de caractères numériques
df['nb_num'] = df.apply(lambda row: number_of_num_char(row['password']), axis=1)

## ratio du mélange lettres/numériques
df['ratio_melange'] = df.apply(lambda row: ratio_num(row), axis=1)

## log-temps de crack offline du mot de passe
df['log_offline_crack_sec'] = df.apply(lambda row: np.log10(row['offline_crack_sec']), axis=1)

########## ETAPE 4 - PREMIÈRE EXPLORATION SIMPLE

## les différentes catégories de passwords triés par fréquence d'utilisation décroissante
print(df['category'].value_counts())

## les 10 mots de passe les plus utilisés
print(df['password'].head(10))

## les 10 mots de passe les plus résistants utilisés
print(df.nlargest(n=10, columns=['strength']))

## les 10 mots de passe les plus rapides à cracker en offline
print(df.nsmallest(n=10, columns=['offline_crack_sec']))
## les 10 mots de passe les plus longs à cracker en offline
print(df.nlargest(n=10, columns=['offline_crack_sec']))

########## ETAPE 5 - PREMIERS GRAPHES
# utile : plot.set( xlabel = "Longueur du mot de passe", ylabel = "Log-temps de crack offline du mot de passe")

## distribution de la catégorie des mots de passe
sns.countplot(y=df.category)

# distribution de la longueur des mots de passe selon la catégorie des mdp
sns.boxplot(x="length", y="category", data=df)

## distribution de la puissance des mots de passe selon la catégorie des mdp
sns.barplot(x="strength", y="category", data=df)

########## ETAPE 6 - GRAPHES PLUS AVANCÉS
##### SCATTERPLOT ET REGPLOT (scatter avec courbe de tendance)
# dispersion du log_temps de crackage par rapport à la longueur du mot de passe
sns.regplot(x=df.length, y=df.log_offline_crack_sec, scatter=True)

# corrélation entre la longueur et la puissance du mot de passe
sns.scatterplot(x=df.length, y=df.strength)

# corrélation entre le ratio lettres/chiffres et la puissance du mot de passe
sns.scatterplot(x=df.ratio_melange, y=df.strength, size=df.length)

# corrélation entre le ratio lettres/chiffres*longueur et la puissance du mot de passe
df_ratio = df.drop(df[df.ratio_melange == 0].index)
sns.regplot(x=(df.length*df.ratio_melange), y=df.strength, scatter=True).set( xlabel = "Longueur*ratio melange alp/num avec les ratio=0", ylabel = "Puissance")
sns.regplot(x=(df_ratio.length*df_ratio.ratio_melange), y=df_ratio.strength, scatter=True).set( xlabel = "Longueur*ratio melange alp/num sans les ratio=0", ylabel = "Puissance")

##### HEAT MAP
df2 = df.drop(['rank', 'password', 'category', 'value', 'time_unit', 'rank_alt', 'offline_crack_sec'], axis=1)
corr = df2.corr()
sns.heatmap(corr, square=True)

########## ETAPE 7 - TEST D'ACP
df_acp = df.drop(['rank', 'password', 'category', 'value', 'time_unit', 'offline_crack_sec', 'rank_alt'], axis=1)
cls = PCA(n_components=7)
pcs = cls.fit_transform(df_acp)


df_pwd = pd.DataFrame(pcs, columns=[f"PC{i}" for i in range(1, 8)])
sns.scatterplot(x="PC1", y="PC2", hue=df.category, data=df_pwd)

# df_quant = df.iloc[:,7:14]

