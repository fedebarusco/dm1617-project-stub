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

    public static Map<String, Double> EntrCat(/*JavaPairRDD<WikiPage, Integer> clustersNew, alla fine non serve nemmeno*/ JavaPairRDD<String, Integer[]> catfreperclu, int k){
        double entr=0.0;
        int mci;
        int mi;
        String temp;
        //JavaPairRDD<String, Integer> catfreq = Analyzer.getCategoriesFrequencies(clustersNew);
        //riesco addirittura a fare una chiamata in meno! swag

        Map<String, Double>entropy = new HashMap<>();

        //inizializzare la mappa, necessario altrimenti da errore
        for(Tuple2<String, Integer[]> a: catfreperclu.collect())
            entropy.put(a._1(), 0.0);

        for (Tuple2<String, Integer[]> a : catfreperclu.collect()){
            temp=a._1();
            mi=0;
            for(int i=0; i<k; i++) {
                mci=a._2()[i];
                if(mci!=0) {//solo se mci è !=0 calcolo mi, grande risparmio
                    for (int j = 0; j < a._2().length; j++) mi = mi + a._2()[i];
                    //poichè le componenti dell'array sono il numero di occorrenze della categoria nei vari cluster
                    //calcolo mi sommando tutte le componenti dell'array
                    entr = entr + formulacategorie(mci, mi);
                }
            }
            if(entr!=0) entr = -entr; //l'entropia è negativa, entr =0 significa TUTTI i punti del clusterino appartengono alla stessa categoria
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


    //****************************************************************************************************************
    //Sbagliatissimi non usare

    public static Map<Integer, Double> NONUSAREcalcolaEntrCluster(JavaPairRDD<WikiPage, Integer> clustersNew){

        double entr=0.0;
        int mci;//rappresenta il numero di occorrenze della categoria i nel cluster (piccolo) C
        int mc;//rappresenta il numero di punti in un cluster (piccolo) C

        //dalla teoria: Entropia dei singoli Cluster C in clustersNew:

        // - (somma per tutte le categorie) mci/mc * log2(mci/mc)

        // ovvero devo sommare per tutte le categorie, quante volte mi compare nei punti del cluster C e dividere per il numero di punti

        //alla fine cambio di segno al tutto


        int stop = 0;//variabile di debug, lo faccio fermare dopo che ha trovato solo 2 mci diversi da 0, ci mette 20 minuti (se va bene)

        //mi dice per ogni categoria la frequenza GLOBALE ovvero quante volte compare nel "cluster grande"
        //le categorie che compaiono una sola volta vengono filtrate
        //usato perchè così ho un RDD con tutte le categorie
        // in un for, cerco per ogni categoria il numero di occorrenze nei "cluster piccoli"
        JavaPairRDD<String, Integer> catfreq = Analyzer.getCategoriesFrequencies(clustersNew);

        //con metodo ottengo gli mc = # di punti in ogni cluster piccolo
        JavaPairRDD<Integer, Integer> arr = Analyzer.getNumberOfPagePerCluster(clustersNew);

        //questa è la struttura dati che ritornerò
        //il primo campo rappresenta l'indice del cluster, il secondo il valore dell'entropia
        Map<Integer, Double> entropy = new HashMap<>();

        //rilevato errore, è indispensabile inizializzare la lista

        for(int i=0; i<arr.count(); i++){
            entropy.put(i, 0.0);
        }

        String temp; //utilizzata nel metodo getNumberOfDocsInClusterPerCat

        //per ogni ID di cluster in arr (coppie ID-# di pagine in Cluster piccolo)
        for (Tuple2<Integer, Integer> e : arr.collect()) {

            mc = e._2();//il secondo elemento della tupla è la dimensione del clusterino

            for (Tuple2<String, Integer> cat : catfreq.collect()) {//per ogni categoria dell'RDD delle categorie

                //mi prendo il nome della categoria
                temp = cat._1();

                //lo do in pasto al tuo metodo
                mci = Analyzer.getNumberOfDocsInClusterPerCat(temp, e._1(), clustersNew);
                //che per funzionare è corretto, ma non è efficiente :(
                if(mci!=0) {//spesso il numero di occorrenze di una categoria in un cluster è 0, ma in quel caso il contributo all'entropia è nullo
                    entr = entr + formulacluster(mci, mc);
                    //debug
                    stop++;
                    //ho verificato che stampa correttamente
                    System.out.println("Categoria =  " + temp + " mc= " + mc + " mci = " + mci + " Entropia calcolata = " + entr);
                }
                if (stop>1) break;//lo faccio fermare appena ne trova una, ma può metterci 40 minuti
            }

            entr = entr*(-1.0); //l'entropia è negativa, va cambiata di segno dalla definizione

            //ora inserisco nel'RDD in uscita

            entropy.put(e._1(), entr);
            //se non lo fermassi continuerebbe a manetta per giorni o forse settimane
            //comunque quando ha finito per il cluster 0 (o 1 a seconda) si resetta il valore di entr
            entr=0; //reset entr

            //debug, lo fermo subito
            if (stop>1) break;
        }
            // non importante
            //per ottenere Rdd invece di mappa
            /*
            List<Tuple2<Integer, Double>> list = new ArrayList<Tuple2<Integer, Double>>();
            for(Map.Entry<Integer, Double> entry : entropy.entrySet()){
                list.add(new Tuple2<Integer, Double>(entry.getKey(),entry.getValue()));
            }

            JavaPairRDD<Integer, Double> RddEntropia = sc.parallelizePairs(list);
            */
            //mi faccio sputare la mappa

            return entropy;
        }

    public static Map<String, Double> NONUSAREcalcolaEntrCat(JavaPairRDD<WikiPage, Integer> clustersNew, int k){
        //k è il numero di cluster, lo prendiamo da input
        //mi in teoria del preprocessing, poi vediamo se lo prendo da una struttura dati
        double entr=0.0;
        int mci;
        int mi;

        String temp;
        JavaPairRDD<String, Integer> catfreq = Analyzer.getCategoriesFrequencies(clustersNew);//ma gli mi li tiriamo fuori da qui in realtà

        Map<String, Double> entropy = new HashMap<>();
        //necessario inizializzare la mappa in output
        for(Tuple2<String, Integer> e : catfreq.collect()){
            entropy.put(e._1(), 0.0);
        }


            for (Tuple2<String, Integer> e : catfreq.collect()) {
                mi = e._2();
                temp = e._1();
                for (int i = 0; i < k; i++) {

                    mci = Analyzer.getNumberOfDocsInClusterPerCat(temp, i, clustersNew);
                    if(mci!=0)
                    entr = entr + formulacategorie(mci, mi);
                }
                entr = -entr; //l'entropia è negativa
                //ora inserisco nel'RDD in uscita
                entropy.put(e._1(), entr);
                entr = 0; //reset entr

            }

        return entropy;
    }

    //*****************************************************************************************************************
    //metodi che fanno solo un semplice calcolo, tutti O(1) e chiamati solo quando mci!=0
    private static double formulacluster(int mci, int mc){
        double num = (double)mci/mc;
        return num*log2(num);
    }

    private static double formulacategorie(int mci, int mi){
        double num=(double)mci/mi;
        return num*log2(num);
    }

    private static double log2(double num){

        return (Math.log(num)/Math.log(2));
    }

    }//fine entropia


