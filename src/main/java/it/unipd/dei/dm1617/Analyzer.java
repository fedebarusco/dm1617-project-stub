package it.unipd.dei.dm1617;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import scala.Tuple2;

import java.util.*;

public class Analyzer {

    public static JavaPairRDD<String, Integer> getCategoriesFrequencies(JavaPairRDD<WikiPage, Integer> clusters) {
        return clusters.flatMapToPair((PairFlatMapFunction<Tuple2<WikiPage, Integer>, String, Integer>) pv -> {
            List<Tuple2<String, Integer>> tmpCats = new ArrayList<>();
            for (String c : pv._1().getCategories()) {
                tmpCats.add(new Tuple2<>(c, 1));
            }
            return tmpCats.iterator();
        }).reduceByKey((f1, f2) -> f1 + f2);
    }

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
