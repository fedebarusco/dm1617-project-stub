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
import org.apache.spark.rdd.RDD;
import scala.Serializable;
import scala.Tuple2;

import java.util.*;

/**
 * Example program to show the basic usage of some Spark utilities.
 */
/*
* La classe TfIdfTransformation:
* implementa l'utilizzo del modello TfIdf e di k-means;
* calcola ciò che serve per il calcolo di 1/c(cat);
* richiama le funzioni per il calcolo di Entropia e di Silhouette per k-means e per il cluster random.
* */
public class TfIdfTransformation {

    public static void main(String[] args) {
        String dataPath = args[0];

        //Imposto la 'hadoop distribution directory'
        //percorso di Giovanni:
        //System.setProperty("hadoop.home.dir", "C:\\Users\\Giovanni\\Documents\\unipd\\magistrale\\Mining\\progetto");
        //percorso di Emanuele;
        System.setProperty("hadoop.home.dir", "C:\\Users\\Emanuele\\Desktop\\hadoop");

        // Usual setup
        SparkConf conf = new SparkConf(true).setAppName("Tf-Ifd transformation");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Load dataset of pages
        JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);

        //quante pagine ci sono nel dataset
        long num_pages = pages.count();
        System.out.println("numero di pagine presenti nel dataset: " + num_pages);

        //Inizio preprocessing
        pages = Analyzer.cleanCategories(pages, 1, 10000, sc);

        // Get text out of pages
        JavaRDD<String> texts = pages.map((p) -> p.getText());

        // Get the lemmas. It's better to cache this RDD since the
        // following operation, lemmatization, will go through it two
        // times.
        JavaRDD<ArrayList<String>> lemmas = Lemmatizer.lemmatize(texts).cache();

        //StopWords
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
        //Fine preprocessing

        long num_pages1 = pages.count();
        System.out.println("numero di pagine presenti nel dataset dopo il preprocessing: " + num_pages1);

        //String path_model = "C:\\Users\\Emanuele\\Desktop\\data\\model_tfidf";
        //String path_model = "C:\\Users\\Giovanni\\Documents\\unipd\\magistrale\\Mininig\\progetto\\modelw2v";

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

        //savataggio del modello TfIdf
        //tfidf.saveAsTextFile(path_model);
        //System.out.println("modello salvato tfidf");

        // In this last step we "zip" toghether the original pages and
        // their corresponding tfidf vectors. We can perform this
        // operation safely because we did no operation changing the order
        // of pages and vectors within their respective datasets,
        // therefore the first vector corresponds to the first page and so
        // on.
        JavaPairRDD<WikiPage, Vector> pagesAndVectors = pages.zip(tfidf);

        // Cluster the data into two classes using KMeans
        int numClusters = 2;
        int numIterations = 20;
        KMeansModel clusters = KMeans.train(tfidf.rdd(), numClusters, numIterations);

        /*
        System.out.println("Cluster centers:");
        for (Vector center : clusters.clusterCenters()) {
            System.out.println(" " + center);
        }
        // here is what I added to predict data points that are within the clusters
        List<Integer> L = clusters.predict(tfidf).collect();
        for (Integer i : L) {
            System.out.println(i);
        }
        */

        /* Map delle coppie (pag, vettore) utilizzando il modello creato prima e il metodo predict
        che prendendo come argomento il vettore corrispondente alla pg restituisce il cluster, l'RDD
        restituita alla fine è la coppia (pagina, indice del cluster corrispondente)
         */
        JavaPairRDD<WikiPage, Integer> clustersNew = pagesAndVectors.mapToPair(pav -> {
            return new Tuple2<WikiPage, Integer>(pav._1(), clusters.predict(pav._2()));
        });

        //numero categorie distinte
        int size = Analyzer.getCategoriesFrequencies(clustersNew).collect().size();
        System.out.println("numero di categorie totali distinte:" + size);

        //Calcolo del valore della funzione obiettivo
        RDD<Vector> data2= tfidf.rdd();
        double f_obiettivo = clusters.computeCost(data2);
        System.out.println("Esito di compute cost: " + f_obiettivo);

        //creo un cluster random
        RandomCluster random = new RandomCluster(pagesAndVectors, numClusters);

        //compute in how many clusters a category is split
        ArrayList<Integer> size_cluster = new ArrayList<>();
        JavaPairRDD<String, List<Integer>> tmp = Analyzer.getNumberOfClustersPerCat(clustersNew);
        for (Map.Entry<String, List<Integer>> e : tmp.collectAsMap().entrySet()) {
            String cat = e.getKey();
            List<Integer> clustersList = e.getValue();
            size_cluster.add(clustersList.size());
            System.out.println("Category \"" + cat + "\" was found in " + clustersList.size() + " clusters.");
        }
        //in media una categoria è stata trovata in tot cluster
        int size_cu = 0;
        int max_cu = size_cluster.get(0);
        for(int i= 0; i < size_cluster.size(); i++){
            if(max_cu < size_cluster.get(i)){
                max_cu = size_cluster.get(i);
            }
            size_cu+=size_cluster.get(i);
        }
        //metto in ordine crescente il numero di cluster
        size_cluster.sort(Integer::compareTo);
        int mediam_cu = size_cluster.get((int)(size_cluster.size()/2));
        System.out.println("mediana dei cluster contenenti una stessa categoria: " + mediam_cu);
        double average_cu = size_cu/size;
        System.out.println("somma del numero di cluster contenti una stessa categoria: " + size_cu);
        System.out.println("k: " + clusters.k());
        System.out.println("media dei cluster contenenti una stessa categoria: " + average_cu);
        System.out.println("il massimo numero di cluster che contengono una stessa categoria: " + max_cu);

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

        //calcola il silhouette coefficient sul cluster generato con kmeans
        double s = Silhouette.getSilhouette(pagesAndVectors, clusters, 10);
        System.out.printf("Total Silhouette: %f\n", s);

        //calcola il silhouette coefficient sul cluster random
        double sr = SilhouetteOnRandom.getSilhouette(pagesAndVectors, random, 10);
        System.out.printf("Total Silhouette: %f\n", sr);

        JavaPairRDD<WikiPage, Integer> clustersRand = pagesAndVectors.mapToPair(pav -> {
            return new Tuple2<WikiPage, Integer>(pav._1(), RandomCluster.predict(pav._2()));
        });

        //Calcolo dell'entropia e confronto con entropia di un cluster casuale
        JavaPairRDD<String, Integer[]> catfreperclu = Analyzer.getCategoryFreqInAllClusters(clustersNew, numClusters);
        JavaPairRDD<String, Integer[]> catfreperrand = Analyzer.getCategoryFreqInAllClusters(clustersRand, numClusters);

        Map<Integer, Double> EntropiaClusters = entropia.EntrClu(clustersNew, catfreperclu);
        System.out.println("la media dell'entropia dei cluster kmeans vale: " + entropia.mediaEntrClu(EntropiaClusters));

        Map<String, Double>EntropiaCategorie = entropia.EntrCat(catfreperclu, numClusters);
        System.out.println("la media dell'entropia delle categorie kmeans vale: " + entropia.mediaEntrCat(EntropiaCategorie, clustersNew));

        Map<Integer, Double> EnCluRand = entropia.EntrClu(clustersRand, catfreperrand);
        System.out.println("la media dell'entropia dei cluster random vale: " + entropia.mediaEntrClu(EnCluRand));

        Map<String, Double>EnCatRand = entropia.EntrCat(catfreperrand, numClusters);
        System.out.println("la media dell'entropia delle categorie vale: " + entropia.mediaEntrCat(EnCatRand, clustersRand));

        System.out.println("Differenza Entropie Kmeans-Random");
        System.out.println("Clusters: " + (entropia.mediaEntrClu(EntropiaClusters)-entropia.mediaEntrClu(EnCluRand)));
        System.out.println("Categorie: " + (entropia.mediaEntrCat(EntropiaCategorie,clustersNew)-entropia.mediaEntrCat(EnCatRand, clustersRand)));

        /*
        JavaPairRDD<WikiPage, Integer> clustersRand = pagesAndVectors.mapToPair(pav -> {
            return new Tuple2<WikiPage, Integer>(pav._1(), RandomCluster.predict(pav._2()));
        });

        //Calcolo del WCSS che valuta il clustering k-means
        //Map.Entry<String, List<Integer>> e : tmp.collectAsMap().entrySet()
        Map<WikiPage, Vector> temp = pagesAndVectors.collectAsMap();
        double media = 0.0;
        double randmedia = 0.0;
        double tosquare = 0.0;
        //da definizione è complesso assai, k^2*wikipages
        for(int i=0; i<numClusters; i++){
            for(Vector center : clusters.clusterCenters()){
                for (Tuple2<WikiPage, Integer> wp : clustersNew.collect()) {
                    if (wp._2() == i){
                        try{
                            tosquare = Distance.euclidianDistance(center, temp.get(wp._1()));
                            media = media + (tosquare*tosquare);
                        }catch(java.lang.NullPointerException e){
                            System.out.println(e);
                            System.out.println("center:" + center);
                            System.out.println(temp.get(wp._1()));
                            media = media + 0.0;
                        }
                    }
                }
            }
        }//fine for più esterno
        System.out.println("WCSS - Media calcolata cluster k-means = " + media);

        for(int i=0; i<numClusters; i++){
            for(Vector center : random.clusterCenters()){
                for (Tuple2<WikiPage, Integer> wp : clustersRand.collect()) {
                    if (wp._2() == i){
                        try {
                            tosquare = Distance.euclidianDistance(center, temp.get(wp._1()));
                            randmedia = randmedia + (tosquare * tosquare);
                        }catch(java.lang.NullPointerException e){
                            System.out.println(e);
                            randmedia = randmedia + 0.0;
                        }
                    }
                }
            }
        }//fine for più esterno

        System.out.println("WCSS - Media calcolata cluster random = " + randmedia);
        */

        // Now we can apply the MR algorithm for word count.
        // Note that we are using `mapToPair` instead of `map`, since
        // it returns a `JavaPairRDD` object, which has methods specialized
        // to work on key-value pairs, like the `reduceByKey` operation we use here.
        //JavaPairRDD<String[], Integer> dCounts = cat
        //        .mapToPair((w) -> new Tuple2<>(w, 1))
        //        .reduceByKey((x, y) -> x + y);
    }
}