# Genre-Klassifikation anhand von Liedtexten
Dieses Programm klassifiziert Liedtexte in verschiedene Musikgenres. Es nutzt maschinelles Lernen, um Wörter in den Texten zu analysieren und eine Vorhersage über das Genre des Liedes zu treffen.

# Voraussetzungen
Python 3

# Unser Datensatz
Unseren Datensatz haben wir von Kaggle von dem User MATEI BEJAN. Hier ist der Link zu diesem Datensatz: https://www.kaggle.com/datasets/mateibejan/multilingual-lyrics-for-genre-classification?select=train.csv

Trainingsdatensatz: 
- 250,000 Lieder in 5 verschiedenen Sprachen
- 10 verschiedene Genres, wobei Pop und Rock am häufigsten vorkommen (ca. 80% des Datensatzes)

![image](https://user-images.githubusercontent.com/122549143/212567403-57df8172-dd50-4821-8d95-27eed309e7eb.png)


Testdatensatz: 
- 8000 Liedtexte, alle auf Englisch
- Genres im Vergleich zum Trainingsdatensatz einigermaßen gleich verteilt
![image](https://user-images.githubusercontent.com/122549143/212567482-706c2818-e41c-4e2f-9893-12d2bc732bb5.png)

Ziel: ausgewogen verteilter Datensatz mit englischen Liedtexten
- Vermeidung von Mehrsprachigkeit, um sich auf Hauptaufgabe zu konzentrieren
- Sortierung des Datensatzes nach englischen Liedern
- Betrachtung des Genres mit den wenigsten Liedtexten(Electronic: 1809 Lieder)

Zusammengeafasst haben wir einen Trainingsdatensatz mit 10 Genre mit jeweils 1890 Liedtexten in englisch.

![image](https://user-images.githubusercontent.com/122549143/212699898-b840bf5c-03c8-4230-be11-f9efb4116226.png)


# Unser Preprocessing
- Entfernung von Sonderzeichen und unnötigen Wörtern aus Liedtexten
- Entfernung von Zeilenumbrüchen (\n)
- Anpassung des Trainingsdatensatzes auf "Genre", "Lyrics" und "cleaned_lyrics"

- Anwendung von Stemming auf "cleaned_lyrics" und Speicherung in "stemmed_lyrics"
- Entfernung von Stopwörter auf "cleaned_lyrcs"  und Speicherung in "wosw_lyrics"
- Anwendung von Stemming auf "wosw_lyrics" und Speicherung in "wosw_stemmed_lyrics"
- Entfernung von Stopwörter auf "stemmed_lyrics"  und Speicherung in "stemmed_wosw_lyrics"
- Ergebnis: Spalten "Genre", "Lyrics", "cleanded_lyrics", "stemmed_lyrics", "wosw_lyrics", "stemmed_wosw_lyrics" und "wosw_stemmed_lyrics" zur Verfügung für Feature-Extraktion und Anwendung von Machine-Learning-Algorithmen.

# Feature Extrahierung
- Anwendung von CountVectorizer(CouVec), TF-IDF Vectozier(Tfidf) und SpaCy-Vektorizer(SV) auf 5 Liedtext-Spalten
- CountVectorizer: Übersetzung von Texten in numerische Darstellung durch Zählen der Häufigkeit von Wörtern; Erstellung einer Sparse Matrix als Eingabe für maschinelle Lernalgorithmen
- TF-IDF Vectorizer: Übersetzung von Texten in numerische Darstellung durch Berechnung von TF-Werten und Multiplikation mit IDF-Werten; Berücksichtigung von Wichtigkeit von Wörtern im Kontext des gesamten Dokuments
- Spacy-Vectorizer: Verarbeitung von Textdokumenten mit Spacy-Modell, Erstellung von Vektoren durch Tokenisierung, Lemmatisierung und POS-Tagging; Umwandlung von Texten in numerische Darstellung als Eingabe für maschinelle Lernmodelle.

# Unsere Machine-Learning-Algorithmen (MLA)
- Wir haben uns für SVM, LGBM und RFC als Machine Learning Algorithmen (MLA) entschieden.
- SVM versucht, eine Entscheidungsgrenze zu finden, die die verschiedenen Klassen möglichst gut trennt und den größtmöglichen Abstand zu den nächsten Beispielen hat.
- LGBM ist eine schnelle und effiziente Implementierung von Gradient Boosting Decision Trees (GBDTs) die speziell für große Datenmengen und hohe Dimensionen optimiert ist.
- RFC besteht aus mehreren Entscheidungsbäumen, die unabhängig voneinander trainiert und dann zusammengeführt werden, um eine endgültige Vorhersage zu treffen.

# Unsere Hyperparamter-Tuning
- Durchführung einer GridSearchCV für SVM und Arbeit mit FLAML für LGBM, bevor finaler Durchlauf
- GridSearchCV: Automatische Optimierung von Hyperparametern durch Ausprobieren von verschiedenen Kombinationen, Auswahl der besten Ergebnisse
- FLAML: Erweiterung von LightGBM, automatisierte Methode zur Optimierung von Hyperparametern und Feature Engineering, verbessert Leistung von LightGBM-Modellen durch Automatisierung von Hyperparameter-Tuning und Feature-Engineering-Prozessen

# Unsere Ergebnisse
Mit den Optimalen Hyperparametern haben wir unsere MLA trainiert und folgende Ergebnisse bekommen:

![image](https://user-images.githubusercontent.com/122549143/212345700-96bd628a-3d1d-4f2a-90d0-1ff25a7228f2.png)

Die besten Resultate pro Feature:
- CountVectorizer:    xxx mit folgenden Hyperparamtern: Y:XX, Y:xx, Y:xx
- TF-IDF Vectorizer:  xxx mit folgenden Hyperparamtern: Y:XX, Y:xx, Y:xx
- Spacy-Vectorizer:   xxx mit folgenden Hyperparamtern: Y:XX, Y:xx, Y:xx

# Was hätten wir besser machen können?/ Woran sollte man weiterarbeiten beim nächsten Mal?
- Weitere verschiedene MLA ausprobieren

- Weitere verschiedene Features-Extrahierer ausprobieren, z.B. Reim-Muster, xxx, xxx, etc. 

- Mit mehr Liedtexten arbeiten

- Möglicherweise mit multilabel classification arbeiten statt mit binary classification

