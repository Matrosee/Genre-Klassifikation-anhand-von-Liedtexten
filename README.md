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
Als erstes haben wir alle Sonderzeichen aus den Liedtexten, die Spalte mit dem Namen "Lyrics" entfernt. Außerdem alle unnötigen Wörter, die auch in Klammern stehen wie z.B. "Chorus" oder "Refrain", da diese für unser Projekt irrelevant sind. Auch Zeichenketten wie "\n" die für einen Zeilenumbruch in Python stehen, wurden auch entfernt. 
Danach wurde der Trainingsdatensatz so angepasst, dass nur noch die Spalten "Genre", "Lyriks" und "cleaned_lyrics" übrig blieben. Auf der Basis von "cleaned_lyrics" haben wir diese gereinigten Liedtexte einmal "gestemmed". Das heißt die Wortendungen entfernt, z.B. "He always runs to the supermarket" wird dann in "He always run to the supermarket". Damit haben wir nur noch die Wortstämme und möglicherweise liefern Songtexte mit nur Wortstämmen, bessere Ergebnisse als die normalen gereingten Liedtexte. Diese Liedtexte sind in der Spalte "stemmed_lyrics". Aus dem gleichen Grund haben wir die Spalte "wosw_lyrics" hinzugefügt. Wir haben englische Stopwörter wie "but", "be","after", etc. herausgefiltert, da diese oft nur eine sehr kleine Rolle in der Sprache spielen und somit möglicherweise das Ergebnis am Ende verschlechtern könnten.  Diese beiden Fälle haben wir dann miteinander kombiniert, indem wir zum einem aus "stemmed_lyrics" die Stopwörter herausgefiltert haben und in "stemmed_wosw_lyrics" gespeicher haben. Als nächstes haben wir von "wosw_lyrics" die Endungen abgeschnitten, das ist dann die Spalte "wosw_stemmed_lyrics. 

Am Ende haben wir dann die Spalten "Genre", "Lyrics", "cleanded_lyrics", "stemmed_lyrics", "wosw_lyrics", "stemmed_wosw_lyrics" und "wosw_stemmed_lyrics". Auf diesem normalen und den verschieden gereinigten Liedtexten werden wir die Features extrahieren und unsere Machine-Learning-Algorithmen anwenden. 

-Entfernung von Sonderzeichen und unnötigen Wörtern aus Liedtexten
-Entfernung von Zeilenumbrüchen (\n)
-Anpassung des Trainingsdatensatzes auf "Genre", "Lyrics" und "cleaned_lyrics"

- Anwendung von Stemming auf "cleaned_lyrics" und Speicherung in "stemmed_lyrics"
- Entfernung von Stopwörter auf "cleaned_lyrcs"  und Speicherung in "wosw_lyrics"
- Anwendung von Stemming auf "wosw_lyrics" und Speicherung in "wosw_stemmed_lyrics"
- Entfernung von Stopwörter auf "stemmed_lyrics"  und Speicherung in "stemmed_wosw_lyrics"
- Ergebnis: Spalten "Genre", "Lyrics", "cleanded_lyrics", "stemmed_lyrics", "wosw_lyrics", "stemmed_wosw_lyrics" und "wosw_stemmed_lyrics" zur Verfügung für Feature-Extraktion und Anwendung von Machine-Learning-Algorithmen.

# Feature Extrahierung
Wir haben auf unsere 5 Liedtext-Spalten jeweils einmal einen CountVectorizer(CouVec), TF-IDF Vectozier(Tfidf), SpaCy-Vektorizer(SV), xxx und xxx angewendet. 

Der CountVectorizer ist ein in der NLP (Natural Language Processing) verwendetes Tool, das verwendet wird, um Texte in eine numerische Darstellung zu übersetzen. Es zählt die Häufigkeit von Wörtern in einem gegebenen Text und erstellt daraus eine Sparse Matrix. Diese Matrix kann dann als Eingabe für maschinelle Lernalgorithmen verwendet werden.


Der TF-IDF Vectorizer ist ein weiteres Tool, das in der NLP (Natural Language Processing) verwendet wird, um Texte in eine numerische Darstellung zu übersetzen. Es basiert auf der Idee des Term-Frequency-Inverse-Document-Frequency (TF-IDF). TF (Term Frequency) misst die Häufigkeit, mit der ein bestimmtes Wort in einem gegebenen Text vorkommt. IDF (Inverse Document Frequency) misst, wie wichtig ein Wort im Vergleich zu allen Texten in einer gegebenen Sammlung von Texten ist. Ein Wort, das in vielen Texten vorkommt, hat einen niedrigeren IDF-Wert als ein Wort, das nur in wenigen Texten vorkommt. Der TF-IDF Vectorizer berechnet zunächst die TF-Werte für jedes Wort in einem Text und multipliziert sie mit dem IDF-Wert des Wortes. Dies erzeugt eine numerische Darstellung des Textes, die die Wichtigkeit jedes Wortes im Kontext des gesamten Dokuments berücksichtigt.


Ein Spacy-Vectorizer ist ein Skript oder eine Funktion, die die Verarbeitung von Textdokumenten mit dem Spacy-Modell durchführt und dabei Vektoren erstellt, die als Eingabe für maschinelle Lernmodelle verwendet werden können. Spacy bietet Funktionen zur Tokenisierung, Lemmatisierung und POS-Tagging von Texten und durch die Erstellung von Vektoren werden diese Texten in numerischen Darstellungen umgewandelt. Diese Vektoren können dann als Eingabe für maschinelle Lernmodelle verwendet werden, um Textklassifikationsaufgaben zu lösen.


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
