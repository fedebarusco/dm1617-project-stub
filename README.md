Progetto suggerito di Data Mining - Goal B
===================

Membri del gruppo
-----------------

Barusco Federico - 1159762\
Cavallin Massimo - 1159787\
Fabian Emanuele - 1156449\
Mangano Jessica - 1132062\
Zampieri Giovanni - 1151614

Struttura del progetto
-----------------

Questo progetto contiene classi già implementate forniteci come base di partenza
e classi create da noi. Le classi già implementate servono per: l'utilizzo di 
WikiPage, la Lemmatizzazione, l'Input/Output, il modello TfIdf, la Cosine 
Distance e altro ancora. Invece, le classi implementate da noi sono:
 - Word2VecOurModel: implementa l'utilizzo del modello Word2Vec, effettua il preprocessing 
 descritto nella relazione sulle pagine, parole e categorie, crea il cluster k-means per un
 dato valore di k (impostato attraverso la variabile numClusters), crea il cluster random
 per un sottoinsieme del dataset in input, 
 calcola i valori necessari per 1/c(cat), per l'Entropia e per il Silhouette Coefficient.
 - entropia: contiene i metodi per il calcolo dell'Entropia dei cluster e delle cateogorie.
 - RandomCluster: contiene i metodi per creare un cluster random su un insieme ridotto 
dei dati del dataset di input.
 - Silhouette: contiene i metodi per il calcolo del Slhouette Coefficient.
 - SilhouetteOnRandom: contiene i metodi per il calcolo del Slhouette Coefficient nel 
cluster random.
 - Analyzer: contiene tutte le funzioni che attraverso l'utilizzo dei 
metodi di Spark restituiscono i valori necessari per il calcolo di 1/c(cat), 
dell'Entropia e di Silhouette.

Inoltre, nella classe TfIdfTransformation che implementa l'utilizzo del modello TfIdf sono state aggiunte tutte
le istruzioni e i metodi presenti anche nella classe Word2VecOurModel. Questo ci ha permesso
di ottenere lo stesso output da due modelli diversi (Word2Vec e TfIdf) che abbiamo opportunamente
confrontato e argomentato nella relazione.\
Infine, nella classe Distance è stata aggiunta la funzione che calcola la distanza
euclidea tra due oggetti Vector.

Le classi entropia, RandomCluster, Sample, Silhouette, SilhouetteOnRandom, 
TfIdfTransformation e Word2VecOurModel sono contenute nel package `it.unipd.dei.dm1617.examples`.
Mentre le classi Analyzer, CountVectorizer, Distance, InputOutput, Lemmatizere WikiPage sono
contenute nel package `it.unipd.dei.dm1617`.

Descrizione dell'output
-----------------

Sia in TfIdfTransformation che in Word2VecOurModel otteniamo in output la stessa tipologia
di dati:
 - numero di pagine presenti nel dataset;
 - numero di pagine presenti nel dataset dopo il preprocessing (va ricordato che con Word2Vec
 non sono comprese le parole che si ripetono meno di 2 volte);
 - numero di categorie totali (distinte) presenti nel dataset;
 - media dei cluster contenenti una stessa categoria (usata per il calcolo di 1/c(category));
 - il Silhouette Coefficient sul cluster generato con kmeans;
 - il silhouette coefficient sul cluster random;
 - media dell'Entropia dei cluster di kmeans;
 - media dell'Entropia delle categorie di kmeans;
 - media dell'Entropia dei cluster del cluster random;
 - media dell'Entropia delle categorie del cluster random;
 - differenza delle Entropie Kmeans-ClusterRandom sia per quanto riguarda le categoria sia per
 i cluster.
 
 Le sezioni per il salvataggio del modello di Word2Vec e per l'impostazione della
 cartella di hadoop (sotto Windows) sono state commentate.
 Nella relazione sono argomentati i risultati ottenuti e la differenza tra i due modelli.