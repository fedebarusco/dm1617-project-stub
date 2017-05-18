package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

/**
 * Example program to show the basic usage of some Spark utilities.
 * La funzione di peso tf-idf (term frequency–inverse document frequency)
 * è una funzione utilizzata per misurare l'importanza di un termine
 * rispetto ad un documento o ad una collezione di documenti.
 * Tale funzione aumenta proporzionalmente al numero di volte che
 * il termine è contenuto nel documento, ma cresce in maniera inversamente
 * proporzionale con la frequenza del termine nella collezione.
 * L'idea alla base di questo comportamento è di dare più importanza
 * ai termini che compaiono nel documento, ma che in generale sono poco frequenti.
 */
public class TfIdfTransformation {

    public static void main(String[] args) {
        String dataPath = args[0];

        /*
            Usual setup. Configuration for a Spark application.
            Used to set various Spark parameters as key-value pairs.
        */
        SparkConf conf = new SparkConf(true).setAppName("Tf-Ifd transformation");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Load dataset of pages
        JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);

        // Get text out of pages
        JavaRDD<String> texts = pages.map((p) -> p.getText());

        // Get the lemmas. It's better to cache this RDD since the
        // following operation, lemmatization, will go through it two
        // times.
        /*
            La lemmatizzazione è il processo di riduzione di una forma flessa
            di una parola alla sua forma canonica (non marcata), detta lemma.
            Nell'elaborazione del linguaggio naturale, la lemmatizzazione è
            il processo algoritmico che determina automaticamente il lemma
            di una data parola.
         */
        JavaRDD<ArrayList<String>> lemmas = Lemmatizer.lemmatize(texts).cache();

        // Transform the sequence of lemmas in vectors of counts in a
        // space of 100 dimensions, using the 100 top lemmas as the vocabulary.
        // This invocation follows a common pattern used in Spark components:
        //
        //  - Build an instance of a configurable object, in this case CountVectorizer.
        //  - Set the parameters of the algorithm implemented by the object
        //  - Invoke the `transform` method on the configured object, yielding
        //  - the transformed dataset.
        //
        // In this case we also cache the dataset because the next step,
        // IDF, will perform two passes over it.
        /*
            CountVectorizer extracts a vocabulary from document collections and generates a CountVectorizerModel.
            CountVectorizerModel Converts a text document to a sparse vector of token counts. param: vocabulary
            An Array over terms. Only the terms in the vocabulary will be counted.
         */
        JavaRDD<Vector> tf = new CountVectorizer()
                .setVocabularySize(100)
                .transform(lemmas)
                .cache();

        // Same as above, here we follow the same pattern, with a small
        // addition. Some of these "configurable" objects configure their
        // internal state by means of an invocation of their `fit` method
        // on a dataset. In this case, the Inverse Document Frequence
        // algorithm needs to know about the term frequencies across the
        // entire input dataset before rescaling the counts of the single
        // vectors, and this is what happens inside the `fit` method invocation.
        JavaRDD<Vector> tfidf = new IDF()
                .fit(tf)
                .transform(tf);

        // In this last step we "zip" toghether the original pages and
        // their corresponding tfidf vectors. We can perform this
        // operation safely because we did no operation changing the order
        // of pages and vectors within their respective datasets,
        // therefore the first vector corresponds to the first page and so
        // on.
        JavaPairRDD<WikiPage, Vector> pagesAndVectors = pages.zip(tfidf);

        // Cluster the data into two classes using KMeans
        /*
            Clustering basato su centri e assumendo di sapere quanti cluster vogliamo.
            Il k-means ha come funzione obiettivo quello di: minimizzare la somma dei quadrati
            delle distanze di ogni punto dal centro del suo cluster. E' più tollerante al rumore
            perché il contributo degli outlier viene un po' mascherato dal contributo di tutti
            gli altri.
            Problema che studiamo facendo riferimento in particolare al caso Euclideo, cioè
            al caso in cui i punti siano punti in uno spazio Euclideo di dimensione R^d, in cui
            si faccia uso della distanza Euclidea. Centroide e permettiamo ai centri dei cluster
            di non appartenere ai punti iniziali. I centri dei cluster saranno dati dai centroidi,
            cioè un "punto medio" rispetto all'insieme dei punti del cluster.
            Il punto medio di un insieme di punti in R^d è dato dalla somma dei punti,
            vista come somma di vettori, diviso n.
            Il centroide di un insieme di punti di R^d è quello che minimizza la somma
            dei quadrati delle distanze di tutti i punti di P.
            Una volta trovato il clustering, quindi la partizione di punti in k cluster, per ciascun
            cluster, il miglior centro ai fini della funzione obiettivo di k-means è il centroide.
        */
        int numClusters = 100;
        int numIterations = 20;
        KMeansModel clusters = KMeans.train(tfidf.rdd(), numClusters, numIterations);

        // KMeansModel: A clustering model for K-means.
        // Each point belongs to the cluster with the closest center.

        System.out.println("Cluster centers:");
        for (Vector center : clusters.clusterCenters()) {
            System.out.println(" " + center);
        }
        // here is what I added to predict data points that are within the clusters
        List<Integer> L = clusters.predict(tfidf).collect();
        for (Integer i : L) {
            System.out.println(i);
        }

        /*
            Map delle coppie (pagina, vettore) utilizzando il modello creato prima e il metodo predict
            che prendendo come argomento il vettore corrispondente alla pagina restituisce il cluster, l'RDD
            restituita alla fine è la coppia (pagina, indice del cluster corrispondente)
         */
        JavaPairRDD<WikiPage, Integer> clustersNew = pagesAndVectors.mapToPair(pav -> {
            return new Tuple2<WikiPage, Integer>(pav._1(), clusters.predict(pav._2()));
        });

        for (Tuple2<WikiPage, Integer> p : clustersNew.collect()) {
            System.out.println(p._1().getTitle() + ", cluster: " + p._2());
        }

        // Finally, we print the distance between the first two pages
        List<Tuple2<WikiPage, Vector>> firstPages = pagesAndVectors.take(2);
        double dist = Distance.cosineDistance(firstPages.get(0)._2(), firstPages.get(1)._2());
        System.out.println("Cosine distance between `" +
                firstPages.get(0)._1().getTitle() + "` and `" +
                firstPages.get(1)._1().getTitle() + "` = " + dist);


    }

}