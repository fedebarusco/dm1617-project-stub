package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.Word2Vec;
import org.apache.spark.mllib.feature.Word2VecModel;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.File;
import java.util.ArrayList;
import java.util.List;


/**
 * Created by Emanuele on 11/05/2017.
 */
public class Word2VecOurModel {
    public static void main(String[] args) {
        String dataPath = args[0];

        // Usual setup
        SparkConf conf = new SparkConf(true)
                .setMaster("local")
                .setAppName("Word2VecModel");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Load dataset of pages
        JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);

        // Get text out of pages
        JavaRDD<String> texts = pages.map((p) -> p.getText());

        // Get the lemmas. It's better to cache this RDD since the
        // following operation, lemmatization, will go through it two
        // times.
        JavaRDD<ArrayList<String>> lemmas = Lemmatizer.lemmatize(texts).cache();

        JavaPairRDD<WikiPage, ArrayList<String>> pageAndLemma = pages.zip(lemmas);

        Word2Vec word2vec = new Word2Vec();

        Word2VecModel model = word2vec
                .setVectorSize(100)
                .fit(lemmas);


        JavaPairRDD<WikiPage, Vector> pageAndVector = pageAndLemma.mapToPair(pair -> {
            Vector docvec = null;
            for(String lemma : pair._2()){
                Vector tmp = model.transform(lemma);
                if(docvec==null){
                    docvec = tmp;
                }else{
                    docvec = sumVectors(tmp,docvec);
                }
            }
            return new Tuple2<WikiPage,Vector>(pair._1(),docvec);
        });

        for(Tuple2<WikiPage, Vector> el : pageAndVector.collect()){
            System.out.println(el._1().getTitle());
            System.out.println(el._2());
        }
        System.out.println();
        //Tuple2<String, Object>[] synonyms = model.findSynonyms("age", 5);
        //synonyms.
        model.save(sc.sc(), "datapath");

        JavaRDD<Vector> data = pageAndVector.map(pair -> pair._2());

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
        KMeansModel clusters = KMeans.train( data.rdd(), numClusters, numIterations);

        System.out.println("Cluster centers:");
        for (Vector center : clusters.clusterCenters()) {
            System.out.println(" " + center);
        }
        // here is what I added to predict data points that are within the clusters
        List<Integer> L = clusters.predict(data).collect();
        for (Integer i : L) {
            System.out.println(i);
        }

        /*
            Map delle coppie (pagina, vettore) utilizzando il modello creato prima e il metodo predict
            che prendendo come argomento il vettore corrispondente alla pagina restituisce il cluster, l'RDD
            restituita alla fine è la coppia (pagina, indice del cluster corrispondente)
         */
        JavaPairRDD<WikiPage, Integer> clustersNew = pageAndVector.mapToPair(pav -> {
            return new Tuple2<WikiPage, Integer>(pav._1(), clusters.predict(pav._2()));
        });

        for (Tuple2<WikiPage, Integer> p : clustersNew.collect()) {
            System.out.println(p._1().getTitle() + ", cluster: " + p._2());
        }

        // Finally, we print the distance between the first two pages
        List<Tuple2<WikiPage, Vector>> firstPages = pageAndVector.take(2);
        double dist = Distance.cosineDistance(firstPages.get(0)._2(), firstPages.get(1)._2());
        System.out.println("Cosine distance between `" +
                firstPages.get(0)._1().getTitle() + "` and `" +
                firstPages.get(1)._1().getTitle() + "` = " + dist);
    }

    public static Vector sumVectors(Vector v1, Vector v2){
        int size = v1.toArray().length;
        double[] sum = new double[size];
        for(int i = 0; i < size; i++){
            sum[i]=v1.toArray()[i] + v2.toArray()[i];
        }
        return Vectors.dense(sum);
    }
}