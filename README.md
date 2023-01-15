# Genre-Klassifikation anhand von Liedtexten
Dieses Programm klassifiziert Liedtexte in verschiedene Musikgenres. Es nutzt maschinelles Lernen, um Wörter in den Texten zu analysieren und eine Vorhersage über das Genre des Liedes zu treffen.

# Voraussetzungen
Python 3

Die in der requirements.txt aufgelisteten Libraries


# Unser Datensatz
Unseren Datensatz haben wir von Kaggle von dem User MATEI BEJAN. Hier ist der Link zu diesem Datensatz: https://www.kaggle.com/datasets/mateibejan/multilingual-lyrics-for-genre-classification?select=train.csv

Trainingsdatensatz: 
- 250,000 Lieder in 5 verschiedenen Sprachen
- 10 verschiedene Genres, wobei Pop und Rock am häufigsten vorkommen (ca. 80% des Datensatzes)

Testdatensatz: 
- 8000 Liedtexte, alle auf Englisch
- Genres im Vergleich zum Trainingsdatensatz einigermaßen gleich verteilt

![image](https://user-images.githubusercontent.com/122549143/212372632-271b5f95-5e41-454c-874d-c4b9e3b15822.png)

Ziel: ausgewogen verteilter Datensatz mit englischen Liedtexten
- Vermeidung von Mehrsprachigkeit, um sich auf Hauptaufgabe zu konzentrieren
- Sortierung des Datensatzes nach englischen Liedern
- Betrachtung des Genres mit den wenigsten Liedtexten(Electronic: 1809 Lieder)

Zusammengeafasst haben wir einen Trainingsdatensatz mit 10 Genre mit jeweils 1800 Liedtexten in englisch.

![image](https://user-images.githubusercontent.com/122549143/212351104-926d15f9-77f4-41d7-812b-137237664cc6.png)

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
Mit unseren Features bzw. Vektoren können wir nun unsere MLA trainieren. Wir haben uns für folgende MLA entschieden: Eine SupportVectorMaschine(SVM), die LightGradientBoostingMachine(LGBM) und den RandomForestClassifier(RFC). Ich habe die SVM und LGBM mit CouVev, Tfidf und SV trainiert und die Christina den RFC mit CouVec und Tfidf. 

# Unsere Hyperparamter-Tuning
Bevor wir allerdings den finalen Durchlauf machen können welcher dann in unseren Ergebnissen mündet, haben "wir eine GridSearchCV für die SVM gemacht  und mit FLAML für die LGBM gearbeitet". Jeder MLA hat bestimmte Parameter wie diese Trainiert werden. Ohne Hyperparamater-Tunig lässt sich nur sehr schwer eine Aussage über die Qualität einer MLA machen. 

GridSearchCV ist eine Klasse in scikit-learn, einer bekannten Python-Library für maschinelles Lernen. Sie ermöglicht es, verschiedene Hyperparameter eines Modells automatisch zu optimieren, indem sie systematisch verschiedene Kombinationen von Hyperparametern ausprobiert und diejenige auswählt, die die besten Ergebnisse liefert. Dazu wird ein "Grid" von möglichen Hyperparameter-Werten angegeben, die durchprobiert werden sollen, und eine Methode zur Bewertung der Modellleistung (z.B. Cross-Validation). GridSearchCV führt dann automatisch eine Suche durch, indem es alle möglichen Kombinationen ausprobiert und diejenige auswählt, die die besten Ergebnisse liefert. Es ist ein sehr nützliches Werkzeug, um die besten Hyperparameter für ein Modell zu finden, ohne dass man diese manuell durchprobieren muss. Es kann jedoch auch Zeitintensiv sein, da es eine große Anzahl an Modellen trainieren und bewerten muss, je nachdem wie viele Hyperparameter und deren Werte angegeben wurden.

FLAML steht für "Fast Light AutoML" und ist eine Erweiterung des LightGBM, einem schnellen und effizienten Gradient Boosting Framework. FLAML ist eine automatisierte Methode zur Optimierung von Hyperparametern und Feature Engineering für LightGBM-Modelle. Es ermöglicht es, die Leistung von LightGBM-Modellen durch die Automatisierung von Hyperparameter-Tuning und Feature-Engineering-Prozessen zu verbessern.FLAML AutoML bei LightGBM automatisiert die Suche nach den besten Hyperparametern für ein LightGBM-Modell und die Auswahl der besten Merkmale, um eine Vorhersage zu treffen. Es kombiniert Algorithmen wie GridSearchCV, RandomSearchCV, Hyperopt und Optuna, um eine effiziente und schnelle Suche durch den Hyperparameterraum durchzuführen. Es beinhaltet auch Funktionen wie die Automatisierung von One-Hot-Encoding, die Erstellung von Interaktionen und die Automatisierung von normalisierten und standardisierten Merkmalen.

# Unsere Ergebnisse
Mit den Optimalen Hyperparametern haben wir unsere MLA trainiert und folgende Ergebnisse bekommen:

![image](https://user-images.githubusercontent.com/122549143/212345700-96bd628a-3d1d-4f2a-90d0-1ff25a7228f2.png)

Wie man hier sehen kann, hat die MLA "xxx" mit den Features "xxx" die besten Resultate erzielt. Mit einer Wahrscheinlichkeit von "xxx" klassifiert unser Programm ein Lied zum richtigen Genre.

# Was hätten wir besser machen können?/ Woran sollte man weiterarbeiten beim nächsten Mal?
Weitere verschiedene MLA ausprobieren

Weitere verschiedene Features-Extrahierer ausprobieren, z.B. Reim-Muster, xxx, xxx, etc. 

Mit mehr Liedtexten arbeiten

Möglicherweise mit multilabel classification arbeiten statt mit binary classification

xxx
