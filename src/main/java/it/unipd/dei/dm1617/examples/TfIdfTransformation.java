package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.linalg.Vector;
import scala.Serializable;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * Example program to show the basic usage of some Spark utilities.
 */
public class TfIdfTransformation {


    public static void main(String[] args) {
        String dataPath = args[0];

        // Usual setup
        SparkConf conf = new SparkConf(true).setAppName("Tf-Ifd transformation");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Load dataset of pages
        JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);

        // Get text out of pages
        JavaRDD<String> texts = pages.map((p) -> p.getText());

        // Get the lemmas. It's better to cache this RDD since the
        // following operation, lemmatization, will go through it two
        // times.
        JavaRDD<ArrayList<String>> lemmas = Lemmatizer.lemmatize(texts).cache();

        Broadcast<Set<String>> stopWords = sc.broadcast(
                new HashSet<>(Arrays.asList(StopWordsRemover.loadDefaultStopWords("english")) )
        );

        lemmas = lemmas.map(ls -> {
            ArrayList<String> filtered = new ArrayList<>();
            for (String s : ls){
                if(!stopWords.getValue().contains(s)) {
                    filtered.add(s);
                }
            }
            return filtered;
        });

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
        int numClusters = 100;
        int numIterations = 20;
        KMeansModel clusters = KMeans.train(tfidf.rdd(), numClusters, numIterations);

        System.out.println("Cluster centers:");
        for (Vector center : clusters.clusterCenters()) {
            System.out.println(" " + center);
        }
        // here is what I added to predict data points that are within the clusters
        List<Integer> L = clusters.predict(tfidf).collect();
        for (Integer i : L) {
            System.out.println(i);
        }

        /* Map delle coppie (pag, vettore) utilizzando il modello creato prima e il metodo predict
        che prendendo come argomento il vettore corrispondente alla pg restituisce il cluster, l'RDD
        restituita alla fine è la coppia (pagina, indice del cluster corrispondente)
         */
        JavaPairRDD<WikiPage, Integer> clustersNew = pagesAndVectors.mapToPair(pav -> {
            return new Tuple2<WikiPage, Integer>(pav._1(), clusters.predict(pav._2()));
        });

        //categorie per cluster
        JavaPairRDD<Integer, List<String>> groupedCategoriesByCluster = Analyzer.getCategoriesDistribution(clustersNew);
        for (Map.Entry<Integer, List<String>> e : groupedCategoriesByCluster.collectAsMap().entrySet()) {
            int clusterId = e.getKey();
            List<String> categories = e.getValue();
            System.out.println(categories.size() + " distinct categories found in cluster " + clusterId);
        }

        //numero categorie distinte
        int size = Analyzer.getCategoriesFrequencies(clustersNew).collect().size();
        System.out.println("numero di categorie totali distinte:" + size);

        //media di categorie presenti in ciascun cluster
        double average = size/clusters.k();
        System.out.println("k dovrebbe essere 100: " + clusters.k());
        System.out.println("media di categorie presenti in ciascun cluster: " + average);

        //per ciascuna categoria restituisco in quanti cluster si trova
        //work in progress

        /*
        for (Tuple2<WikiPage, Integer> p : clustersNew.collect()) {
            System.out.println(p._1().getTitle() + ", cluster: " + p._2());
        }
*/
        // Finally, we print the distance between the first two pages
        List<Tuple2<WikiPage, Vector>> firstPages = pagesAndVectors.take(2);
        double dist = Distance.cosineDistance(firstPages.get(0)._2(), firstPages.get(1)._2());
        System.out.println("Cosine distance between `" +
                firstPages.get(0)._1().getTitle() + "` and `" +
                firstPages.get(1)._1().getTitle() + "` = " + dist);


        // Get text out of pages
        JavaRDD<String[]> cat = pages.map((p) -> p.getCategories());

        List<String[]> catlist = cat.collect();
        int idx = 0;
        for(String[] list : catlist){
            System.out.println("Cats of doc" + idx++);
            for(String c : list){
                System.out.println(c);
            }
            System.out.println();
        }


        // Now we can apply the MR algorithm for word count.
        // Note that we are using `mapToPair` instead of `map`, since
        // it returns a `JavaPairRDD` object, which has methods specialized
        // to work on key-value pairs, like the `reduceByKey` operation we use here.
        JavaPairRDD<String[], Integer> dCounts = cat
                .mapToPair((w) -> new Tuple2<>(w, 1))
                .reduceByKey((x, y) -> x + y);


/*
        class TupleComparator implements Comparator<Tuple2<String, Integer>>, Serializable {
            @Override
            public int compare(Tuple2<String, Integer> t1, Tuple2<String, Integer> t2) {
                return t1._2().compareTo(t2._2());
            }
        }


        // Instead of sorting and collecting _all_ the values on the master
        // machine, we take only the top 100 words by count.
        // In general this operation is safer, since we can bound the number
        // of elements that are collected by the master, thus avoiding OutOfMemory errors
        List<Tuple2<String[], Integer>> lTopCounts = dCounts.top(500, new TupleComparator());
        lTopCounts.forEach((tuple) -> {
            String[] word = tuple._1();
            int count = tuple._2();
            System.out.println(word + " :: " + count);
        });

*/
    }

}
