import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
from wordcloud import WordCloud
import collections
from collections import Counter

#lecture des donnees 
data = pd.read_csv('D:/articles.csv')

#exploration de Dataset 
data.shape
data.head(5)

# Supprimer les colonnes inutiles
colonnes_inutiles = ['id', 'date', 'year', 'year', 'month', 'url']
data = data.drop(colonnes_inutiles, axis=1)
data.head(5)

# verifier s'il y a des valeurs qui manque
data.isnull().values.any()

#utiliser missingo pour visualiser les valeurs qui manque
missingno.matrix(data)

#supprimer les valeurs manquantes 
data.dropna(inplace=True)

#verifier s'il y a des valeurs en double
duplicates = data[data.duplicated()]
print(duplicates)

# Groupement des données par publication et comptage des articles
article_count = data.groupby('publication').size().reset_index(name='article_count')

# Affichage du DataFrame "article_count"
print(article_count)

#nombre d'article par publiction
plt.figure(figsize=(6, 3))
plt.bar(article_count['publication'], article_count['article_count'])
plt.xlabel('Publication')
plt.ylabel('Nombre d\'articles')
plt.title('Nombre d\'articles par publication')
plt.xticks(rotation=90)
plt.show()

# Calculer le décompte des auteurs
class_counts = data['author'].value_counts().head(10)

# Tracer le graphique des auteurs les plus fréquents
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Auteur')
plt.ylabel('Nombre d\'articles')
plt.title('Top 10 des auteurs les plus fréquents')
plt.xticks(rotation=45)
plt.show()

# Concaténer le contenu de toutes les lignes
all_text = " ".join(data['content'])

# separer les mots
words = all_text.split()

# Compter les fréquences des mots
word_counts = collections.Counter(words)

# Créer le nuage de mots
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

# Afficher le nuage de mots
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#faire une courbe zipf avec les articles non nettoyer 
#Concaténez tous les articles
all_articles = ' '.join(data['content'])
#Divisez la chaîne de texte en mots individuels :
words = all_articles.split()
#Comptez la fréquence de chaque mot 
word_counts = Counter(words)
#Triez les mots par fréquence décroissante :
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
#Extrayez les fréquences et les rangs des mots :
frequencies = [count for word, count in sorted_word_counts]
ranks = list(range(1, len(sorted_word_counts) + 1))
#Tracez le graphique de la courbe de Zipf
plt.plot(ranks, frequencies)
plt.xlabel('Rang')
plt.ylabel('Fréquence')
plt.title('Courbe de Zipf')
plt.show()

#faire une courbe zipf avec des donnees nottoyer 
all_articles = ' '.join(data['articles_limm'])
words = all_articles.split()
word_counts = Counter(words)
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
frequencies = [count for word, count in sorted_word_counts]
ranks = list(range(1, len(sorted_word_counts) + 1))
plt.plot(ranks, frequencies)
plt.xlabel('Rang')
plt.ylabel('Fréquence')
plt.title('Courbe de Zipf')
plt.show()