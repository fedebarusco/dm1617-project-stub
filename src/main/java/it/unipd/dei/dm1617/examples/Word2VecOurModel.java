package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.Word2Vec;
import org.apache.spark.mllib.feature.Word2VecModel;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;
import org.apache.spark.mllib.clustering.StreamingKMeans;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.io.File;
import java.util.*;


public class Word2VecOurModel {
    public static void main(String[] args) {
        String dataPath = args[0];

        // Usual setup
        SparkConf conf = new SparkConf(true)
                .setMaster("local")
                .setAppName("Word2VecModel");
        JavaSparkContext sc = new JavaSparkContext(conf);

        //Set hadoop distribution directory
        System.setProperty("hadoop.home.dir", "C:\\Users\\Emanuele\\Desktop\\hadoop");

        // Load dataset of pages
        JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);

        //quante pagine (dataset) ci sono nel dataset
        long num_pages = pages.count();
        System.out.println("numero di pagine presenti nel dataset: " + num_pages);

        //preprocessing category
        pages = Analyzer.cleanCategories(pages, 1, 10000, sc);

        //quante pagine ci sono nel dataset dopo il preprocessing sul numero di pagine nelle categorie
        long num_pages1 = pages.count();
        System.out.println("numero di pagine presenti nel dataset dopo: " + num_pages1);

        // Get text out of pages
        JavaRDD<String> texts = pages.map((p) -> p.getText());

        // Get the lemmas. It's better to cache this RDD since the
        // following operation, lemmatization, will go through it two
        // times.
        JavaRDD<ArrayList<String>> lemmas = Lemmatizer.lemmatize(texts).cache();

        //StopWords
        Broadcast<Set<String>> stopWords = sc.broadcast(
                new HashSet<>(Arrays.asList(StopWordsRemover.loadDefaultStopWords("english")) )
        );

        lemmas = lemmas.map(ls -> {
            ArrayList<String> filtered = new ArrayList<>();
            for (String s : ls){
                if(!stopWords.getValue().contains(s)) {
                    filtered.add(s);
                }
            }
            return filtered;
        });

        JavaPairRDD<WikiPage, ArrayList<String>> pageAndLemma = pages.zip(lemmas);

        String path_model = "C:\\Users\\Emanuele\\Desktop\\data\\model_word2vec";
        Word2Vec word2vec = new Word2Vec();

        //Word2VecModel model = Word2VecModel.load(sc.sc(), path_model);

        //Word2VecModel sameModel = Word2VecModel.load(sc.sc(), path_model);

        Word2VecModel model = word2vec
                .setVectorSize(100)
                //nel modello consideriamo solo le parole che si ripetono più di 2 volte (>=2) questo per la legge di Zipf (da approfondire)
                .setMinCount(2) // il default è 5 se si vuole lasciare 5 bisogna levare le pagine che danno problemi
                .fit(lemmas);

        //savataggio del modello
        model.save(sc.sc(), path_model);

        JavaPairRDD<WikiPage, Vector> pageAndVector = pageAndLemma.mapToPair(pair -> {
            int i = 0;
            Vector docvec = null;
            for (String lemma : pair._2()) {
                Vector tmp = null;
                try {
                    tmp = model.transform(lemma); //= sameModel.transform();
                } catch (java.lang.IllegalStateException e) {
                    i++;
                    tmp = null;
                    System.out.println(e);
                    continue;
                }
                if (docvec == null) {
                    docvec = tmp;
                } else {
                    docvec = sumVectors(tmp, docvec);
                }
            }
            System.out.println("Parole perse perchè non contenute nel vocabolario di scala:" + i);
            //i=0;
            return new Tuple2<WikiPage, Vector>(pair._1(), docvec);
        });

        pageAndVector = pageAndVector.filter(pair -> {
            return pair != null && pair._2() != null && pair._1() != null;
        }).cache();
        // provare vari valori di k e vedere la qualità del cluster
        //influenza di k sul clustering

        for (Tuple2<WikiPage, Vector> el : pageAndVector.collect()) {
            System.out.println(el._1().getTitle());
            System.out.println(el._2());
        }
        System.out.println();

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
        KMeansModel clusters = KMeans.train(data.rdd(), numClusters, numIterations);

        System.out.println("Cluster centers:");
        for (Vector center : clusters.clusterCenters()) {
            System.out.println(" " + center);
        }
        // here is what I added to predict data points that are within the clusters
        List<Integer> L = clusters.predict(data).collect();
        /*for (Integer i : L) {
            System.out.println(i);
        }*/

        /*
            Map del le coppie (pagina, vettore) utilizzando il modello creato prima e il metodo predict
            che prendendo come argomento il vettore corrispondente alla pagina restituisce il cluster, l'RDD
            restituita alla fine è la coppia (pagina, indice del cluster corrispondente)
         */
        JavaPairRDD<WikiPage, Integer> clustersNew = pageAndVector.mapToPair(pav -> {
            return new Tuple2<WikiPage, Integer>(pav._1(), clusters.predict(pav._2()));
        });
/*
        for (Tuple2<WikiPage, Integer> p : clustersNew.collect()) {
            System.out.println(p._1().getTitle() + ", cluster: " + p._2());
        }
*/
        //per ogni categoria vedere in quanti cluster è spezzata
        //numero categorie distinte
        int size = Analyzer.getCategoriesFrequencies(clustersNew).collect().size();
        System.out.println("numero di categorie totali distinte:" + size);

        //compute in how many clusters a category is split
        ArrayList<Integer> size_cluster = new ArrayList<>();
        JavaPairRDD<String, List<Integer>> tmp = Analyzer.getNumberOfClustersPerCat(clustersNew);
        for (Map.Entry<String, List<Integer>> e : tmp.collectAsMap().entrySet()) {
            String cat = e.getKey();
            List<Integer> clustersList = e.getValue();
            size_cluster.add(clustersList.size());
            System.out.println("Category \"" + cat + "\" was found in " + clustersList.size() + " clusters.");
        }
        //in media una categoria è stata trovata in tot cluster
        int size_cu = 0;
        int max_cu = size_cluster.get(0);
        for(int i= 0; i < size_cluster.size(); i++){
            if(max_cu < size_cluster.get(i)){
                max_cu = size_cluster.get(i);
            }
            size_cu+=size_cluster.get(i);
        }
        //metto in ordine crescente il numero di cluster
        size_cluster.sort(Integer::compareTo);
        int mediam_cu = size_cluster.get((int)(size_cluster.size()/2));
        System.out.println("mediana dei cluster contenenti una stessa categoria: " + mediam_cu);
        double average_cu = size_cu/size;
        System.out.println("somma del numero di cluster contenti una stessa categoria: " + size_cu);
        System.out.println("k: " + clusters.k());
        System.out.println("media dei cluster contenenti una stessa categoria: " + average_cu);
        System.out.println("il massimo numero di cluster che contengono una stessa categoria: " + max_cu);

        //categorie per cluster
        ArrayList<Integer> size_categories = new ArrayList<>();
        JavaPairRDD<Integer, List<String>> groupedCategoriesByCluster = Analyzer.getCategoriesDistribution(clustersNew);
        for (Map.Entry<Integer, List<String>> e : groupedCategoriesByCluster.collectAsMap().entrySet()) {
            int clusterId = e.getKey();
            List<String> categories = e.getValue();
            size_categories.add(categories.size());
            System.out.println(categories.size() + " distinct categories found in cluster " + clusterId);
        }

        //media delle categorie (con ripetizioni) presenti in ciascun cluster
        int size_c = 0;
        int max_cat = size_categories.get(0);
        for(int i= 0; i < size_categories.size(); i++){
            if(max_cat < size_categories.get(i)){
                max_cat = size_categories.get(i);
            }
            size_c+=size_categories.get(i);
        }
        size_categories.sort(Integer::compareTo);
        for(int i= 0; i < size_categories.size(); i++){
            System.out.println("categorie ordinate: " + size_categories.get(i));
        }
        int mediam = size_categories.get((int)(size_categories.size()/2));
        System.out.println("mediana: " + mediam);
        double average = size_c/clusters.k();
        System.out.println("categorie (con ripetizioni) presenti nei cluster: " + size_c);
        System.out.println("k: " + clusters.k());
        System.out.println("media di categorie presenti in ciascun cluster: " + average);
        System.out.println("il massimo numero di categorie presenti in un cluster: " + max_cat);

        // Finally, we print the distance between the first two pages
        List<Tuple2<WikiPage, Vector>> firstPages = pageAndVector.take(2);
        double dist = Distance.cosineDistance(firstPages.get(0)._2(), firstPages.get(1)._2());
        System.out.println("Cosine distance between `" +
                firstPages.get(0)._1().getTitle() + "` and `" +
                firstPages.get(1)._1().getTitle() + "` = " + dist);

        // Get Silhouette coefficient
        // Restituisce il silhouette coefficient relativo al dataset
        // Prob è un parametro che permette di decidere che percentuale di punti intendiamo utilizzare, utile se vogliamo snellire il procedimento

        double s = Silhouette.getSilhouette(pageAndVector, clusters, 10);
        System.out.printf("Total Silhouette: %f\n", s);

        //Calcolo dell'entropia e confronto con entropia di un cluster casuale

        //Inizializzare un cluster casuale che io qui chiamerò Gianfranco.
        //Gianfranco deve essere di tipo JavaPairRDD<Wikipage, Integer>
        //Buon lavoro jessica

        //JavaPairRDD<WikiPage, Integer> Gianfranco;
        Map<Integer, Double> EntropiaClusters = entropia.calcolaEntrCluster(clustersNew);
        Map<String, Double> EntropiaCategorie = entropia.calcolaEntrCat(clustersNew, numClusters);//che cacchio è k = num di cluster
        //Map<Integer, Double> EntropiaRandomClusters = entropia.calcolaEntrCluster(Gianfranco);
        //Map<String, Double> EntropiaRandomCategorie = entropia.calcolaEntrCat(Gianfranco, numClusters);
        //Eseguire il confronto






    }







    public static Vector sumVectors(Vector v1, Vector v2) {
        int size = v1.toArray().length;
        double[] sum = new double[size];
        for (int i = 0; i < size; i++) {
            sum[i] = v1.toArray()[i] + v2.toArray()[i];
        }
        return Vectors.dense(sum);
    }
}