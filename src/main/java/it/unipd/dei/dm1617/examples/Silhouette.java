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
import scala.Tuple3;

import java.util.*;


public class Silhouette {

    protected static double getSilhouette(JavaPairRDD<WikiPage, Vector> pageAndVector, KMeansModel clusters, double prob) {
        //BREAKPOINT QUI
        Map<Integer,List<Vector>> clusterPointsByID = new HashMap<>();
        for (Tuple2<WikiPage, Vector> pav : pageAndVector.collect()) {
            int cluster = clusters.predict(pav._2());
            List<Vector> l = clusterPointsByID.getOrDefault(cluster, null);
            if (l == null)
                l = new ArrayList<>();
            //
            if (Math.random() < prob)
                l.add(pav._2());
            clusterPointsByID.put(cluster, l);
        }

        return silhouetteCoefficient(clusterPointsByID);
    }

    public static Vector sumVectors(Vector v1, Vector v2) {
        int size = v1.toArray().length;
        double[] sum = new double[size];
        for (int i = 0; i < size; i++) {
            sum[i] = v1.toArray()[i] + v2.toArray()[i];
        }
        return Vectors.dense(sum);
    }

    static double silhouetteCoefficient(Map<Integer, List<Vector>> clusterPointsByID) {

        double totalSilhouetteCoefficient = 0.0;
        long sampleCount = 0L;


        for (Map.Entry<Integer, List<Vector>> cluster : clusterPointsByID.entrySet()) {
            List<Vector> clusteredPoints = cluster.getValue();
            long clusterSize = clusteredPoints.size();
            // Increment the total sample count for computing silhouette coefficient
            sampleCount += clusterSize;
            // if there's only one element in a cluster, then assume the silhouetteCoefficient for
            // the cluster = 0, this is an arbitrary choice per Section 2: Construction of Silhouettes
            // in the referenced paper
            if (clusterSize > 1) {

                double clusterAvg = 0.0;
                for (Vector point : clusteredPoints) {
                    double pointIntraClusterDissimilarity = clusterDissimilarityForPoint(point, clusteredPoints, true);
                    double pointInterClusterDissimilarity = minInterClusterDissimilarityForPoint(cluster.getKey(), point, clusterPointsByID);
                    double s = silhouetteCoefficient(pointIntraClusterDissimilarity, pointInterClusterDissimilarity);
                    totalSilhouetteCoefficient += s;
                    clusterAvg += s/clusterSize;

                    //System.out.printf("%f\n", s);
                }
                System.out.println("Il silhouette coefficent del cluster " + cluster.getKey() + " vale " + clusterAvg);
            }

        }

        return sampleCount == 0 ? 0.0 : totalSilhouetteCoefficient / sampleCount;
    }

    private static double minInterClusterDissimilarityForPoint(
            int otherClusterID,
            Vector point,
            Map<Integer, List<Vector>> clusteredPointsMap) {
        return clusteredPointsMap.entrySet().stream().mapToDouble(entry -> {
            // only compute dissimilarities with other clusters
            if (entry.getKey().equals(otherClusterID)) {
                return Double.POSITIVE_INFINITY;
            }
            return clusterDissimilarityForPoint(point, entry.getValue(), false);
        }).min().orElse(Double.POSITIVE_INFINITY);
    }

    static double silhouetteCoefficient(double ai, double bi) {
        if (ai < bi) {
            return 1.0 - (ai / bi);
        }
        if (ai > bi) {
            return (bi / ai) - 1.0;
        }
        return 0.0;
    }

    private static double clusterDissimilarityForPoint(Vector point,
                                                       List<Vector> clusterPoints,
                                                       boolean ownCluster) {
        double totalDissimilarity = 0.0;
        for (Vector clusterPoint : clusterPoints) {
            totalDissimilarity += Distance.euclidianDistance(point, clusterPoint);
        }

        if (ownCluster) {
            // (points.size -1) because a point's dissimilarity is being measured with other
            // points in its own cluster, hence there would be (n - 1) dissimilarities computed
            return totalDissimilarity / (clusterPoints.size() - 1);
        } else {
            // point dissimilarity is being measured with all points of one of other clusters
            return totalDissimilarity / clusterPoints.size();
        }
    }
}