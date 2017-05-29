package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import org.apache.commons.collections.map.HashedMap;
import org.apache.commons.lang3.ObjectUtils;
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
import java.lang.Math;

public class entropia {

    public static Map<Integer, Double> calcolaEntrCluster(JavaPairRDD<WikiPage, Integer> clustersNew, JavaSparkContext sc){

        double entr=0.0;
        int mci;
        int mc;

        JavaPairRDD<String, Integer> catfreq = Analyzer.getCategoriesFrequencies(clustersNew);

        //int L = Analyzer.getCategoriesFrequencies(clustersNew).collect().size(); //numero totale di categorie

        JavaPairRDD<Integer, Integer> arr = Analyzer.getNumberOfPagePerCluster(clustersNew);
        //con questo for ottengo gli mc = # di punti in ogni cluster

        Map<Integer, Double> entropy = new HashMap<>();
        String temp;

        for (Tuple2<Integer, Integer> e : arr.collect()) {
            mc = e._2();

            //mancano gli mci, trova gli mci in funzione di e._1(), = ID del cluster,  e chiama
            for (Tuple2<String, Integer> cat : catfreq.collect()) {
                temp = cat._1();
                mci = Analyzer.getNumberOfDocsInClusterPerCat(temp, e._1(), clustersNew);
                entr = entr + formulacluster(mci, mc);
            }
            entr = -entr; //l'entropia è negativa

            //ora inserisco nel'RDD in uscita
            entropy.put(e._1(), entr);
            entr=0; //reset entr
        }

            //per ottenere Rdd invece di mappa
            /*
            List<Tuple2<Integer, Double>> list = new ArrayList<Tuple2<Integer, Double>>();
            for(Map.Entry<Integer, Double> entry : entropy.entrySet()){
                list.add(new Tuple2<Integer, Double>(entry.getKey(),entry.getValue()));
            }

            JavaPairRDD<Integer, Double> RddEntropia = sc.parallelizePairs(list);
            */
            return entropy;
        }

    public static Map<String, Double> calcolaEntrCat(JavaPairRDD<WikiPage, Integer> clustersNew, int k, JavaSparkContext sc){
        //k è il numero di cluster, lo prendiamo da input
        //mi in teoria del preprocessing, poi vediamo se lo prendo da una struttura dati
        double entr=0.0;
        int mci;
        int mi;
        String temp;
        JavaPairRDD<String, Integer> catfreq = Analyzer.getCategoriesFrequencies(clustersNew);//ma gli mi li tiriamo fuori da qui in realtà

        Map<String, Double> entropy = new HashMap<>();


            for (Tuple2<String, Integer> e : catfreq.collect()) {
                mi = e._2();
                temp = e._1();
                for (int i = 0; i < k; i++) {

                    mci = Analyzer.getNumberOfDocsInClusterPerCat(temp, i, clustersNew);
                    entr = entr + formulacategorie(mci, mi);
                }
                entr = -entr; //l'entropia è negativa
                //ora inserisco nel'RDD in uscita
                entropy.put(e._1(), entr);
                entr = 0; //reset entr
            }

        return entropy;
    }

    private static double formulacluster(int mci, int mc){

        return (double)(mci/mc)*log2((double)(mci/mc));
    }

    private static double formulacategorie(int mci, int mi){

        return (double)(mci/mi)*log2((double)(mci/mi));
    }

    private static double log2(double num){

        return (Math.log(num)/Math.log(2));
    }

    }//fine entropia


