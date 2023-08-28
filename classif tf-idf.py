import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.stem import PorterStemmer
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier


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

# Appliquer la lemmatisation 
data['articlelim'] = data['cleaned_article'].apply(lemmatize_text)

#stemming
stemmer = PorterStemmer()
def stem_column(column):
    stemmed_column = column.apply(lambda text: ' '.join([stemmer.stem(word) for word in word_tokenize(text)]))
    return stemmed_column

data['articlelim'] = stem_column(data['articlelim'])

data.rename(columns={'articlelim': 'articles_limm'}, inplace=True)



#Divisez vos données en variables indépendantes (X) et variable cible (y) :
X = data['articles_limm']
y = data['author']

#Appliquez la vectorisation TF-IDF sur les données textuelles :
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)


#Effectuez le suréchantillonnage des données pour équilibrer les classes à l'aide de SMOTE avec la verctorisation tf_idf
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

#Divisez les données en ensembles d'apprentissage et de test :
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

#Initialisez et entraînez le modèle SVM :
svm_model = SVC()
svm_model.fit(X_train, y_train)

#Faites des prédictions sur l'ensemble de test :
y_pred = svm_model.predict(X_test)

# Calculer la précision, le rappel et la f-mesure
classification_rep = classification_report(y_test, y_pred)
print("Rapport de classification :")
print(classification_rep)

#Évaluez les performances du modèle en utilisant les mesures de classification appropriées :
print(classification_report(y_test, y_pred))

# Initialiser le modèle de régression logistique
lr_model = LogisticRegression()

# Entraîner le modèle sur l'ensemble d'apprentissage
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)
# Calculer la précision, le rappel et la f-mesure
classification_rep = classification_report(y_test, y_pred)
print("Rapport de classification :")
print(classification_rep)

# Création et entraînement du modèle Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Calculer la précision, le rappel et la f-mesure
y_pred = model.predict(X_test)
classification_rep = classification_report(y_test, y_pred)
print("Rapport de classification :")
print(classification_rep)

# Prédiction sur l'ensemble de test
predictions = model.predict(X_test)

# Évaluation de la précision du modèle
accuracy = accuracy_score(y_test, predictions)
print("Précision du modèle : {:.2f}%".format(accuracy * 100))

#MLP
model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Calculer la précision, le rappel et la f-mesure
classification_rep = classification_report(y_test, y_pred)
print("Rapport de classification :")
print(classification_rep)

print(classification_report(y_test, y_pred))

# Créez un objet GradientBoostingClassifier
gradient_boosting_classifier = GradientBoostingClassifier()

# Entraînez le modèle sur les données d'apprentissage
gradient_boosting_classifier.fit(X_train, y_train)

# Faites des prédictions sur les données de test
predictions = gradient_boosting_classifier.predict(X_test)

# Évaluez les performances du modèle en utilisant la précision
accuracy = accuracy_score(y_test, predictions)
print("Précision du Gradient Boosting Classifier : {:.2f}".format(accuracy))

y_pred = gradient_boosting_classifier.predict(X_test)
# Calculer la précision, le rappel et la f-mesure
classification_rep = classification_report(y_test, y_pred)
print("Rapport de classification :")
print(classification_rep)


