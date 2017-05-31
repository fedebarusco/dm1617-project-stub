package it.unipd.dei.dm1617;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import scala.Tuple12;
import scala.Tuple2;

import java.util.*;

public class Analyzer {

    public static int getNumberOfDocsInClusterPerCat(String cat, int clusterIdx, JavaPairRDD<WikiPage, Integer> clusters) {
        return clusters.filter(t -> (t._2() == clusterIdx) && (Arrays.asList(t._1().getCategories()).contains(cat))).collect().size();
    }//controlla che non sovrascrive il cluster
    //chiamare size() prima e dopo

    /**
     * Elimina da ciascun oggetto wikipage le categorie che compaiono nel dataset con frequenza al di fuori  delle soglie specificate.
     * Se un oggetto wikipage resta con 0 categorie, l'oggetto viene eliminato.
     *
     * @param data
     * @param lThr
     * @param hThr
     * @param sc
     * @return
     */
    public static JavaRDD<WikiPage> cleanCategories(JavaRDD<WikiPage> data, int lThr, int hThr, JavaSparkContext sc) {
        JavaPairRDD<String, Integer> catsFreqs = getCategoriesFrequencies(data);
        Broadcast<Map<String, Integer>> bcf = sc.broadcast(catsFreqs.collectAsMap());
        return data.map(p -> {
            List<String> newCategories = new ArrayList<>();
            for (String s : p.getCategories()) {
                try {
                    int currFreq = bcf.getValue().get(s);
                    if (currFreq > lThr && currFreq < hThr) {
                        newCategories.add(s);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            String[] cToadd = new String[newCategories.size()];
            for (int i = 0; i < newCategories.size(); i++) {
                cToadd[i] = newCategories.get(i);
            }
            p.setCategories(cToadd);
            return p;
        }).filter((Function<WikiPage, Boolean>) p -> p.getCategories().length > 0);
    }

    /**
     * Counts how many pages are associated to each cluster.
     * We first collect the wikipages associated to a specific cluster index and then count them.
     *
     * @param clusters
     * @return
     */
    public static JavaPairRDD<Integer, Integer> getNumberOfPagePerCluster(JavaPairRDD<WikiPage, Integer> clusters) {
        return clusters.mapToPair(t -> new Tuple2<Integer, WikiPage>(t._2(), t._1())).groupByKey().mapToPair(p -> {
            int size = 0;
            Iterator i = p._2().iterator();
            while (i.hasNext()) {
                size++;
                i.next();
            }
            return new Tuple2<>(p._1(), size);
        });
    }

    /**
     * Counts the number of clusters in which a category appears.
     *
     * @param clusters
     * @return
     */
    public static JavaPairRDD<String, List<Integer>> getNumberOfClustersPerCat(JavaPairRDD<WikiPage, Integer> clusters) {
        return clusters.flatMapToPair((PairFlatMapFunction<Tuple2<WikiPage, Integer>, String, List<Integer>>) pv -> {
            List<Tuple2<String, List<Integer>>> tmpCats = new ArrayList<>();
            List<Integer> tmpclusters = new ArrayList<>();
            for (String c : pv._1().getCategories()) {
                tmpclusters.add(pv._2());
                tmpCats.add(new Tuple2(c, tmpclusters));
            }
            return tmpCats.iterator();
        }).reduceByKey((c1, c2) -> {
            Set<Integer> l = new HashSet<>();

            l.addAll(c1);
            l.addAll(c2);

            List<Integer> retval = new ArrayList<>();
            retval.addAll(l);
            return retval;
        });
    }

    /**
     * Counts how many times a category appears in the whole dataset starting from the clustered data.
     *
     * @param clusters
     * @return
     */
    public static JavaPairRDD<String, Integer> getCategoriesFrequencies(JavaPairRDD<WikiPage, Integer> clusters) {
        return clusters.flatMapToPair((PairFlatMapFunction<Tuple2<WikiPage, Integer>, String, Integer>) pv -> {
            List<Tuple2<String, Integer>> tmpCats = new ArrayList<>();
            for (String c : pv._1().getCategories()) {
                tmpCats.add(new Tuple2<>(c, 1));
            }
            return tmpCats.iterator();
        }).reduceByKey((f1, f2) -> f1 + f2);
    }

    /**
     * Counts how many times a category appears in the whole dataset starting from the raw data.
     *
     * @param pages
     * @return
     */
    public static JavaPairRDD<String, Integer> getCategoriesFrequencies(JavaRDD<WikiPage> pages) {
        return pages.flatMapToPair((PairFlatMapFunction<WikiPage, String, Integer>) p -> {
            List<Tuple2<String, Integer>> tmpCats = new ArrayList<>();
            for (String c : p.getCategories()) {
                tmpCats.add(new Tuple2<>(c, 1));
            }
            return tmpCats.iterator();
        }).reduceByKey((f1, f2) -> f1 + f2);
    }

    /**
     * Returns the list of categories associated to each cluster.
     * The JavaRDD has the index of the cluster as key and the list of categories that appear in that cluster as values.
     *
     * @param clusters
     * @return
     */
    public static JavaPairRDD<Integer, List<String>> getCategoriesDistribution(JavaPairRDD<WikiPage, Integer> clusters) {
        JavaPairRDD<Integer, List<String>> categoriesByclusterIdx = clusters.mapToPair(el -> new Tuple2<Integer, List<String>>(el._2(), Arrays.asList(el._1().getCategories())));

        return categoriesByclusterIdx.reduceByKey((l1, l2) -> {
            ArrayList<String> l = new ArrayList<>();
            for (String s : l1) {
                if (!l.contains(s)) {
                    l.add(s);
                }
            }
            for (String s : l2) {
                if (!l.contains(s)) {
                    l.add(s);
                }
            }
            return l;
        });
    }
}