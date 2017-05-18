package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.CountVectorizer;
import it.unipd.dei.dm1617.InputOutput;
import it.unipd.dei.dm1617.Lemmatizer;
import it.unipd.dei.dm1617.WikiPage;
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

        int i = 0;
        for(Tuple2<WikiPage, Vector> el : pageAndVector.collect()){
            System.out.println(el._1().getTitle());
            System.out.println(el._2());
        }
        System.out.println();

        model.save(sc.sc(), "datapath");

        //JavaRDD<Vector> data = pageAndVector.map();

        // Cluster the data into two classes using KMeans
        int numClusters = 100;
        int numIterations = 20;
        //KMeansModel clusters = KMeans.train(data.rdd(), numClusters, numIterations);
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