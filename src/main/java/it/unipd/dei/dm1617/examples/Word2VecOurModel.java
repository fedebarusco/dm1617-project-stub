package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.CountVectorizer;
import it.unipd.dei.dm1617.InputOutput;
import it.unipd.dei.dm1617.Lemmatizer;
import it.unipd.dei.dm1617.WikiPage;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
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
/*
    ENG: The purpose and usefulness of Word2vec is to group the vectors
    of similar words together in vectorspace. That is, it detects
    similarities mathematically. Word2vec creates vectors that are
    distributed numerical representations of word features, features
    such as the context of individual words.
    Given enough data, usage and contexts, Word2vec can make highly
    accurate guesses about a word’s meaning based on past appearances.
    Those guesses can be used to establish a word’s association with
    other words (e.g. “man” is to “boy” what “woman” is to “girl”),
    or cluster documents and classify them by topic.
    The output of the Word2vec neural net is a vocabulary in which
    each item has a vector attached to it, which can be fed into
    a deep-learning net or simply queried to detect relationships
    between words.
    ITA: Word2vec è una semplice rete neurale artificiale a due strati
    progettata per elaborare il linguaggio naturale, l'algoritmo
    richiede in ingresso un corpus e restituisce un insieme di vettori
    che rappresentano la distribuzione semantica delle parole nel testo.
    Viene costruito un vettore per ogni parola contenuta nel corpus
    e ogni parola, rappresentata come un punto nello spazio
    multidimensionale creato. In questo spazio le parole saranno più
    vicine se riconosciute come semanticamente più simili.
 */
/*
    Word2Vec e` un modello che, dato un corpus di documenti, associa a
    ogni parola un vettore in uno spazio di dimensionalita` decisa dall'utente.
    Spark fornisce un'implementazione che funziona come segue.
    - Dato un dataset di testi, di tipo JavaRDD<String>
    - Trasformare ogni documento del dataset in una sequenza di token
      (potenzialmente con qualche ulteriore preprocessing). L'importante e` ottenere
      un dataset di tipo JavaRDD<Iterable<String>>.
    - Creare un oggetto di tipo org.apache.spark.mllib.feature.Word2Vec, che va configurato
      usando i vari metodi "set...".
    - Una volta configurato l'oggetto Word2Vec, allenare il modello invocando il metodo
      Word2Vec.fit con argomento il dataset di sequenze di token.
    - Questa invocazione restituisce un oggetto di tipo Word2VecModel. E' possibile usare
      questo oggetto per trovare il vettore che rappresenta una data parola usando il
      metodo transform(String).

    Manca la funzionalita` per trasformare i documenti in vettori usando questo modello.
    Un modo semplice di farlo e` di trasformare ogni parola di un documento nel vettore
    corrispondente e poi fare la media:

    Distribuire ai vari worker il modello allenato usando il metodo del broadcast visto a lezione.
    Per ogni documento, inizializzare il vettore somma usando il metodo statico Vectors.zeros
    Per ogni parola del documento ottenere il vettore corrispondente usando il metodo transform
    del modello, e sommare questo vettore al vettore somma. Si può usare il metodo BLAS.axpy
    Riscalare il vettore somma per il numero di parole del documento, usando ad esempio il metodo BLAS.scal
    A questo punto si ottiene il vettore che rappresenta il documento.

    Suggerimento: allenare Word2Vec puo` richiedere tempo. E' consigliato salvare il modello
    su file dopo averlo allenato usando il metodo save.
    E' possibile caricare un modello salvato con il metodo load.
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