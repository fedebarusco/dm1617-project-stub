package it.unipd.dei.dm1617.examples;


import org.apache.spark.mllib.clustering.StreamingKMeans;
import org.apache.spark.util.Utils;

/**
 * Created by Piccy on 27/05/2017.
 */
public class RandomCluster {

    public static void main(String[] args) {


        int numDimensions = 100;
        int numClusters = 100;
        StreamingKMeans RandomCLuster = new StreamingKMeans()
                .setK(numClusters)
                .setDecayFactor(1.0)
                .setRandomCenters(numDimensions, 0.0, Utils.random().nextLong());

    }
}
