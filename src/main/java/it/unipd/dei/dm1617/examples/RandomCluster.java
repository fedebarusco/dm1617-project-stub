package it.unipd.dei.dm1617.examples;


import it.unipd.dei.dm1617.WikiPage;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.clustering.StreamingKMeans;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.util.Utils;
import scala.Tuple2;

import static java.lang.Math.*;


public class RandomCluster {

    JavaPairRDD<WikiPage, Integer> RandClus;

    public RandomCluster(KMeansModel clusters, JavaPairRDD<WikiPage, Vector> pageAndVector) {
        RandClus = pageAndVector.mapToPair(pav -> {
            return new Tuple2<WikiPage, Integer>(pav._1(), (int)(Math.random()*100));
        });
    }

    public JavaPairRDD<WikiPage, Integer> getRandClus(){
        return RandClus;
    }


}
