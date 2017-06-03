package it.unipd.dei.dm1617.examples;


import it.unipd.dei.dm1617.Distance;
import it.unipd.dei.dm1617.WikiPage;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.clustering.StreamingKMeans;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.util.Utils;
import scala.Tuple2;

import static java.lang.Math.*;

import java.util.*;


public class RandomCluster {

    static int nodes;
    static List<Tuple2<Integer, Vector>> centers;

    public RandomCluster(JavaPairRDD<WikiPage, Vector> pageAndVector, int k) {
        nodes = k;
        centers = findCenters(pageAndVector);
    }

    //Metodo che restituisce una lista di coppie indice, vettore. Dove l'indice è l'indice del cluster
    //Il vettore rappresenta le coordinate del centro
    private static List<Tuple2<Integer, Vector>> findCenters(JavaPairRDD<WikiPage, Vector> pageAndVector)
    {
        int i = 0;
        List<Tuple2<Integer, Vector>> centers = new ArrayList<>();
        //Prendiamo dei punti a caso dal dataset
        List<Tuple2<WikiPage, Vector>> samples = pageAndVector.takeSample(false, nodes);
        //Ogni punto dei samples viene eletto a centro del cluster i
        for(Tuple2<WikiPage, Vector> spl : samples)
        {
            centers.add(new Tuple2<Integer, Vector>(i, spl._2()));
            i++;
        }
        return centers;
    }

    //metodo che ci restituisce un vettore contenente i centri dei cluster
    public  Vector[] clusterCenters(){
        Vector[] c = new Vector[nodes];
        for(Tuple2<Integer, Vector> p : centers)
            c[p._1()] = p._2();
        return c;
    }

    //Metodo che assegna ad un cluster il punto point
    public static int predict(Vector point){
        Random rnd = new Random();
        double prob = 0.2;
        double dist = 0.0;
        double minDist = Double.POSITIVE_INFINITY;
        double stat = 0.0;
        int index = Integer.MAX_VALUE;

        //Cicliamo su ogni singolo centro e ci calcoliamo la distanza euclidea
        for(Tuple2<Integer, Vector> c : centers){
            dist = Distance.euclidianDistance(point, c._2());
            //Se la distanza dal centro in esame è minore di minDist allora lo assegnamo momentaneamente al cluster index
            if (dist <= minDist){
                minDist = dist;
                index = c._1();
            }
            stat = rnd.nextDouble();

            if ((minDist == 0.0) && (stat < prob))
                break;
        }

        return index;
    }


}