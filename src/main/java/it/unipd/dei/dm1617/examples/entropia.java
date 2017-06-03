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

/*
* Nella classe entropia sono implementati i metodi utlizzati per il calcolo sia dell'entropia dei cluster sia delle categorie.
* */
public class entropia {

    public static Map<Integer, Double> EntrClu(JavaPairRDD<WikiPage, Integer> clustersNew, JavaPairRDD<String, Integer[]> catfreperclu){

        double entr=0.0;
        int mci;
        int mc;

        JavaPairRDD<Integer, Integer> arr = Analyzer.getNumberOfPagePerCluster(clustersNew);

        Map<Integer, Double> entropy = new HashMap<>();

        for(int i =0; i<arr.count(); i++){
            entropy.put(i, 0.0);
        }

        Integer temp[];
        for(Tuple2<Integer, Integer> b : arr.collect()) {

            mc=b._2();

            for(Tuple2<String, Integer[]> a : catfreperclu.collect()){
                temp=a._2();
                mci=temp[b._1()];
                if(mci!=0) {
                    entr = entr + formulacluster(mci, mc);
                }
            }
            if(entr!=0) entr = -entr;

            entropy.put(b._1(),entr);
            entr=0.0;
        }

        return entropy;

    }

    public static Map<String, Double> EntrCat(JavaPairRDD<String, Integer[]> catfreperclu, int k){
        double entr=0.0;
        int mci;
        int mi;
        String temp;

        Map<String, Double>entropy = new HashMap<>();

        //inizializzo la mappa
        for(Tuple2<String, Integer[]> a: catfreperclu.collect())
            entropy.put(a._1(), 0.0);

        for (Tuple2<String, Integer[]> a : catfreperclu.collect()){
            temp=a._1();
            mi=0;
            for(int i=0; i<k; i++) {
                mci=a._2()[i];
                if(mci!=0) {//solo se mci è !=0 calcolo mi
                    for (int j = 0; j < a._2().length; j++) mi = mi + a._2()[i];
                    //poichè le componenti dell'array sono il numero di occorrenze della categoria nei vari cluster
                    //calcolo mi sommando tutte le componenti dell'array
                    entr = entr + formulacategorie(mci, mi);
                }
            }
            //l'entropia è negativa, entr =0 significa TUTTI i punti del cluster appartengono alla stessa categoria
            if(entr!=0) entr = -entr;
            //ora inserisco nel'RDD in uscita
            entropy.put(temp, entr);
            entr = 0.0; //reset entr
        }
        return entropy;
    }

    public static double mediaEntrClu(Map<Integer, Double> entr){
        double media=0.0;
        for (int i=0; i<entr.size(); i++){
            media = media + entr.get(i);
        }
        return media/entr.size();
    }

    public static double mediaEntrCat(Map<String, Double> entr, JavaPairRDD<WikiPage,Integer> clustersNew){

        double media=0.0;

        JavaPairRDD<String, Integer> catfreq = Analyzer.getCategoriesFrequencies(clustersNew);

        for (Tuple2<String, Integer> a : catfreq.collect()){
            media = media + entr.get(a._1());
        }

        return media/entr.size();
    }

    //metodo chiamato solo quando mci!=0 per il calcolo del'entropia dei cluster
    private static double formulacluster(int mci, int mc){
        double num = (double)mci/mc;
        return num*log2(num);
    }

    //metodo chiamato solo quando mci!=0 per il calcolo del'entropia delle categorie
    private static double formulacategorie(int mci, int mi){
        double num=(double)mci/mi;
        return num*log2(num);
    }

    //metodo che mi restituisce il logaritmo in base 2
    private static double log2(double num){

        return (Math.log(num)/Math.log(2));
    }

}