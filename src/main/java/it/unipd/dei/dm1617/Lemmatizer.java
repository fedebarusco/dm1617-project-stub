package it.unipd.dei.dm1617;

import edu.stanford.nlp.simple.Document;
import edu.stanford.nlp.simple.Sentence;
import org.apache.spark.api.java.JavaRDD;

import javax.print.DocFlavor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.regex.Pattern;

/**
 * Collection of functions that allow to transform texts to sequence
 * of lemmas using lemmatization. An alternative process is
 * stemming. For a discussion of the difference between stemming and
 * lemmatization see this link: https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
 */
/*
  La LEMMATIZZAZIONE è il processo di riduzione di una forma flessa
  di una parola alla sua forma canonica (non marcata), detta lemma.
  Nell'elaborazione del linguaggio naturale, la lemmatizzazione è il
  processo algoritmico che determina automaticamente il lemma di una
  data parola. Il processo può coinvolgere altre attività di elaborazione
  del linguaggio, quali ad esempio l'analisi morfologica e grammaticale.
  In molte lingue, le parole appaiono in diverse forme flesse.
  Per esempio, in italiano il verbo camminare può apparire come cammina,
  camminò, camminando e così via. La forma canonica, camminare, è il
  lemma della parola ed è la forma di riferimento per cercare la parola
  all'interno di un dizionario.
 */
/*
  Lo STEMMING è il processo di riduzione della forma flessa di una parola
  alla sua forma radice, detta tema. Il tema non corrisponde necessariamente
  alla radice morfologica (lemma) della parola: normalmente è sufficiente
  che le parole correlate siano mappate allo stesso tema (ad esempio, che
  andare, andai, andò mappino al tema and), anche se quest'ultimo non è una
  valida radice per la parola.
 */
public class Lemmatizer {

  /**
   * Some symbols are interpreted as tokens. This regex allows us to exclude them.
   */
  public static Pattern symbols = Pattern.compile("^[',\\.`/-_]+$");

  /**
   * A set of special tokens that are present in the Wikipedia dataset
   */
  public static HashSet<String> specialTokens =
    new HashSet<>(Arrays.asList("-lsb-", "-rsb-", "-lrb-", "-rrb-", "'s", "--"));

  /**
   * Transform a single document in the sequence of its lemmas.
   */
  public static ArrayList<String> lemmatize(String doc) {
    Document d = new Document(doc.toLowerCase());
    // Count spaces to allocate the vector to the right size and avoid trashing memory
    int numSpaces = 0;
    for (int i = 0; i < doc.length(); i++) {
      if (doc.charAt(i) == ' ') {
        numSpaces++;
      }
    }
    ArrayList<String> lemmas = new ArrayList<>(numSpaces);

    for (Sentence sentence : d.sentences()) {
      for (String lemma : sentence.lemmas()) {
        // Remove symbols
        if (!symbols.matcher(lemma).matches() && !specialTokens.contains(lemma)) {
          lemmas.add(lemma);
        }
      }
    }

    return lemmas;
  }

  /**
   * Transform an RDD of strings in the corresponding RDD of lemma
   * sequences, with one sequence for each original document.
   */
  public static JavaRDD<ArrayList<String>> lemmatize(JavaRDD<String> docs) {
    return docs.map((d) -> lemmatize(d));
  }

  /**
   * Transform an RDD of WikiPage objects into an RDD of WikiPage
   * objects with the text replaced by the concatenation of the lemmas
   * in each page.
   */
  public static JavaRDD<WikiPage> lemmatizeWikiPages(JavaRDD<WikiPage> docs) {
    return docs.map((wp) -> {
      ArrayList<String> lemmas = lemmatize(wp.getText());
      StringBuilder newText = new StringBuilder();
      for(String lemma : lemmas) {
        newText.append(lemma).append(' ');
      }
      wp.setText(newText.toString());
      return wp;
    });
  }

  public static void main(String[] args) {
    System.out.println(lemmatize("This is a sentence. This is another. The whole thing is a document made of sentences."));
  }

}
/*first try*/