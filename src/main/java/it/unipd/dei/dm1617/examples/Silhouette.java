package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import org.apache.commons.collections.map.HashedMap;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.feature.Word2Vec;
import org.apache.spark.mllib.feature.Word2VecModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.util.*;

/*
* Classe che implementa i metodi per il calcolo del Silhouette Coefficient per k-means.
* */
public class Silhouette {

    static double[] valuesOnCluster;
    static double[] valuesOnPoints;

    protected static double getSilhouette(JavaPairRDD<WikiPage, Vector> pageAndVector, KMeansModel clusters, double prob) {
        valuesOnCluster = new double[clusters.clusterCenters().length];
        valuesOnPoints = new double[pageAndVector.collect().size()];
        //Mappa che contiene come chiave il valore del cluster, come valore una lista dei punti associati a tale cluster
        Map<Integer,List<Vector>> pointsByID = new HashMap<>();
        //Per ogni coppia page and vector
        for (Tuple2<WikiPage, Vector> pav : pageAndVector.collect()) {
            int clusterID = clusters.predict(pav._2());

            List<Vector> l = pointsByID.getOrDefault(clusterID, null);
            if (l == null)
                l = new ArrayList<>();
            //Scarta dei punti con probabilità 1 - prob. Utile qualora non volessimo calcolare ai e bi su tutti i punti del cluster
            //Troppo cattivo con i cluster piccoli. WIP
            //if (Math.random() < prob)
            l.add(pav._2());
            pointsByID.put(clusterID, l);
        }

        return silhouetteCoefficient(clusters, pointsByID);
    }

    static double silhouetteCoefficient(KMeansModel clusters, Map<Integer, List<Vector>> pointsByID) {
        double totalSilhouetteCoefficient = 0.0;
        long numPoints = 0L;
        int j = 0;

        for (Map.Entry<Integer, List<Vector>> cluster : pointsByID.entrySet()) {

            List<Vector> pointsInCluster = cluster.getValue();
            long clusterSize = pointsInCluster.size();
            numPoints += clusterSize;

            //se c'è solo un punto nel cluster, il silhouette coefficient vale 0 come visto nella teoria
            if (clusterSize > 1) {

                double clusterAvg = 0.0;
                for (Vector point : pointsInCluster) {
                    double pointAiValue = getPointAiValue(point, pointsInCluster, true);
                    double pointBiValue = getPointBiValue(clusters, cluster.getKey(), point, pointsByID);
                    double s = silhouetteCoefficient(pointAiValue, pointBiValue);
                    totalSilhouetteCoefficient += s;
                    clusterAvg += s/clusterSize;
                    valuesOnPoints[j++] = s;
                    //System.out.println(s);
                }
                System.out.println("Il silhouette coefficent del cluster " + cluster.getKey() + " vale: " + clusterAvg);
                valuesOnCluster[cluster.getKey()] = clusterAvg;
            }
        }

        double medianOnCluster = medianValueOnCluster();
        System.out.println("Il valore della mediana sui cluster è: " + medianOnCluster);

        double medianOnPoints = medianValueOnPoints();
        System.out.println("Il valore della mediana sui punti è: " + medianOnPoints);

        if(numPoints == 0) {
            return 0.0;
        }
        else
            return totalSilhouetteCoefficient / numPoints;
    }

    private static double getPointBiValue(KMeansModel clusters, int otherClusterID, Vector point,
                                          Map<Integer, List<Vector>> clusteredPointsMap) {
        double minDist = Double.POSITIVE_INFINITY;
        for (Map.Entry<Integer, List<Vector>> entry : clusteredPointsMap.entrySet()) {
            // lo applichiamo solo sui cluster diversi da quelli a cui appartiene dil punto
            if (entry.getKey().equals(otherClusterID))
                continue;

            Vector center = clusters.clusterCenters()[entry.getKey()];
            double centerDist = Distance.euclidianDistance(center, point);
            // se il centro del cluster in esame dista più di 2minDist rinunciamo a fare questo conto
            // passiamo all'iterazione successiva
            if (centerDist * .5 > minDist)
                continue;

            double dist = getPointAiValue(point, entry.getValue(), false);
            if (dist < minDist)
                minDist = dist;
        }

        return minDist;
    }

    static double silhouetteCoefficient(double ai, double bi) {
        //Applichiamo le formule viste in teoria.
        if (ai < bi) {
            return 1.0 - (ai / bi);
        }
        if (ai > bi) {
            return (bi / ai) - 1.0;
        }
        return 0.0;
    }

    private static double getPointAiValue(Vector point, List<Vector> pointsInCluster, boolean sameCluster) {
        double totalValue = 0.0;
        for (Vector p : pointsInCluster) {
            totalValue += Distance.euclidianDistance(point, p);
        }

        if (sameCluster) {
            //Siamo sullo stesso cluster del punto, calcoliamo ai.
            return totalValue / (pointsInCluster.size() - 1);
        } else {
            //Stiamo vacendo la stessa valutazione su un cluster differente, quindi bi.
            return totalValue / pointsInCluster.size();
        }
    }

    //Metodo che restituisce la mediana del silhouette coefficient calcolato sui cluster
    private static double medianValueOnCluster(){
        Arrays.sort(valuesOnCluster);
        if ((valuesOnCluster.length % 2) == 0)
            return ((valuesOnCluster[valuesOnCluster.length/2] + valuesOnCluster[(valuesOnCluster.length/2) - 1]) / 2);
        else
            return valuesOnCluster[valuesOnCluster.length/2];
    }

    //Metodo che restituisce la mediana del silhouette coefficient calcolato sui punti.
    private static double medianValueOnPoints(){
        Arrays.sort(valuesOnPoints);
        if ((valuesOnPoints.length % 2) == 0)
            return ((valuesOnPoints[valuesOnPoints.length/2] + valuesOnPoints[(valuesOnPoints.length/2) - 1]) / 2);
        else
            return valuesOnPoints[valuesOnPoints.length/2];
    }
}