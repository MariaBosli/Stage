import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.stem import PorterStemmer


data = pd.read_csv('D:/articles.csv')

# Supprimer les colonnes inutiles
colonnes_inutiles = ['id', 'date', 'year', 'year', 'month', 'url']
data = data.drop(colonnes_inutiles, axis=1)

#supprimer les valeurs manquantes 
data.dropna(inplace=True)

# Téléchargement des ressources nécessaires de NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Liste des stopwords
stopwords = set(stopwords.words('english'))

# Initialisation du lemmatiseur
lemmatizer = WordNetLemmatizer()

# Fonction de nettoyage du texte
def clean_text(text):
    # Suppression des balises HTML
    cleaned_text = re.sub('<.*?>', '', text)
    
    # Conversion en minuscules
    cleaned_text = cleaned_text.lower()
    
    # Tokenisation des mots
    tokens = word_tokenize(cleaned_text)
    
    # Suppression de la ponctuation
    tokens = [token for token in tokens if token not in punctuation]
    
    # Suppression des stopwords
    tokens = [token for token in tokens if token not in stopwords]
    
    # Lemmatisation des mots
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Reconstitution du texte nettoyé
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

# Exemple d'utilisation
text = "<html><body><h1>Hello, world!</h1></body></html>"
cleaned_text = clean_text(text)
print(cleaned_text)

# Appliquer la fonction clean_text sur la colonne 'article' de data
data['cleaned_article'] = data['content'].apply(clean_text)

# Afficher le résultat
print(data['cleaned_article'])

def clean_text(text):
    # Supprimer les caractères spéciaux et la ponctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convertir le texte en minuscules
    text = text.lower()
    
    # Supprimer les mots vides (stop words)
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Rejoindre les tokens en une chaîne de caractères
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text


data['cleaned_article'] = data['content'].apply(clean_text)

print(data['cleaned_article'].iloc[0])

# Définir la fonction de lemmatisation
def lemmatize_text(text):
    lemmatized_text = []
    for word in nltk.word_tokenize(text):
        lemma = lemmatizer.lemmatize(word)
        lemmatized_text.append(lemma)
    return ' '.join(lemmatized_text)

# Appliquer la lemmatisation à la colonne 
data['articlelim'] = data['cleaned_article'].apply(lemmatize_text)

#stemming
stemmer = PorterStemmer()
def stem_column(column):
    stemmed_column = column.apply(lambda text: ' '.join([stemmer.stem(word) for word in word_tokenize(text)]))
    return stemmed_column

data['articlelim'] = stem_column(data['articlelim'])

data.rename(columns={'articlelim': 'articles_limm'}, inplace=True)