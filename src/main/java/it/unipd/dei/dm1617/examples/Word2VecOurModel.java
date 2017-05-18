package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.CountVectorizer;
import it.unipd.dei.dm1617.InputOutput;
import it.unipd.dei.dm1617.Lemmatizer;
import it.unipd.dei.dm1617.WikiPage;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.Word2Vec;
import org.apache.spark.mllib.feature.Word2VecModel;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.File;
import java.util.ArrayList;


/**
 * Created by Emanuele on 11/05/2017.
 */
public class Word2VecOurModel {
    public static void main(String[] args) {
        String dataPath = args[0];

        // Usual setup
        SparkConf conf = new SparkConf(true)
                .setMaster("local")
                .setAppName("Word2VecModel");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Load dataset of pages
        JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);

        // Get text out of pages
        JavaRDD<String> texts = pages.map((p) -> p.getText());

        // Get the lemmas. It's better to cache this RDD since the
        // following operation, lemmatization, will go through it two
        // times.
        JavaRDD<ArrayList<String>> lemmas = Lemmatizer.lemmatize(texts).cache();

        JavaPairRDD<WikiPage, ArrayList<String>> pageAndLemma = pages.zip(lemmas);

        Word2Vec word2vec = new Word2Vec();

        Word2VecModel model = word2vec
                .setVectorSize(100)
                .fit(lemmas);


        JavaPairRDD<WikiPage, Vector> pageAndVector = pageAndLemma.mapToPair(pair -> {
            Vector docvec = null;
            for(String lemma : pair._2()){
                Vector tmp = model.transform(lemma);
                if(docvec==null){
                    docvec = tmp;
                }else{
                    docvec = sumVectors(tmp,docvec);
                }
            }
            return new Tuple2<WikiPage,Vector>(pair._1(),docvec);
        });

        for(Tuple2<WikiPage, Vector> el : pageAndVector.collect()){
            System.out.println(el._1().getTitle());
            System.out.println(el._2());
        }
        System.out.println();
        //Tuple2<String, Object>[] synonyms = model.findSynonyms("age", 5);
        //synonyms.
        model.save(sc.sc(), "datapath");

        int numClusters = 100;
        int numIterations = 20;
        //KMeansModel clusters = KMeans.train(tfidf.rdd(), numClusters, numIterations);
    }

    public static Vector sumVectors(Vector v1, Vector v2){
        int size = v1.toArray().length;
        double[] sum = new double[size];
        for(int i = 0; i < size; i++){
            sum[i]=v1.toArray()[i] + v2.toArray()[i];
        }
        return Vectors.dense(sum);
    }
}
/*
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}

val input = sc.textFile("data/mllib/sample_lda_data.txt").map(line => line.split(" ").toSeq)

val word2vec = new Word2Vec()

val model = word2vec.fit(input)

val synonyms = model.findSynonyms("1", 5)

for((synonym, cosineSimilarity) <- synonyms) {
  println(s"$synonym $cosineSimilarity")
}

// Save and load model
model.save(sc, "myModelPath")
val sameModel = Word2VecModel.load(sc, "myModelPath")

log.info("Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(dataPath);

        log.info("Load data....");
        SentenceIterator iter = new LineSentenceIterator(new File("/Users/cvn/Desktop/file.txt"));
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();
*/