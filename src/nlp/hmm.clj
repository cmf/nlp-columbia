(ns nlp.hmm
  "Implementation of the Hidden Markov Model tagger
   Assignment 1

   The main data structures are the word/tag map, which has the form:
   {word {tag count ...} ...}

   and the n-gram count map, which is a map of n-gram to count. The
   n-gram is a vector of the (string) items."

  (:require [clojure.java.io :as io]
            [clojure.string :as str])
  (:import [java.io BufferedReader StringReader]))

(defn sentences
  "Lazy sequence of sentences given a sequence of lines.
   Sentences are separated by blank lines"
  [lines]
  (if (seq lines)
    (let [[sentence rest] (split-with (complement str/blank?) lines)]
      (cons (apply vector sentence) (lazy-seq (sentences (drop-while str/blank? rest)))))))

(defn update-tags
  "Given a sentence of [word tag] vectors and a word/tag map,
   returns a word/tag map with the tag counts for each word
   in the sequence updated."
  [items tags]
  (reduce (fn [tags item]
            (let [[word tag] item
                  tag-counts (get tags word {})
                  count (get tag-counts tag 0)]
              (assoc tags word (assoc tag-counts tag (inc count)))))
          tags
          items))

(defn subvecs
  "Given a vector and a length, returns a sequence of all n-grams
   from the vector of the passed length."
  [vec length]
  (if (<= length (count vec))
    (map #(subvec vec % (+ length %)) (range (inc (- (count vec) length))))))

(defn update-counts
  "Given a sentence of [word tag] vectors and an n-gram count
   map, return a map with the n-gram counts updated for n-grams
   up to length 3."
  [items counts]
  (let [tags (conj (into [:* :*] (map second items)) :STOP)
        subsequences (apply concat (map #(subvecs tags %) (range 1 4)))]
    (reduce (fn [counts subsequence]
              (let [count (get counts subsequence 0)]
                (assoc counts subsequence (inc count))))
            counts
            subsequences)))

(defn calculate-totals
  "Given a word/tag map, calculate the total counts for each
   tag. Returns a map of tag to total."
  [tags]
  (reduce (fn [totals tag-counts]
            (reduce (fn [totals [tag count]]
                      (assoc totals tag (+ count (get totals tag 0))))
                    totals
                    (seq tag-counts)))
          {}
          (vals tags)))

(defn only-rare
  "Map an uncommon word to the constant symbol :RARE"
  [word]
  :RARE)

(defn common-symbol
  "Map an uncommon word to its common symbol"
  [word]
  (cond
    (re-find #"\d" word) :NUMERIC
    (re-matches #"[A-Z]+" word) :ALL-CAPITALS
    (re-matches #"^.*[A-Z]$" word) :LAST-CAPITAL
    :else :RARE))

(defn replace-rare
  "Given a word/tag map, returns a map with all instances of
   uncommon words mapped to their corresponding common symbol."
  [tags common-symbol]
  (reduce (fn [tags word]
            (let [counts (get tags word)
                  count (apply + (vals counts))]
              (if (> 5 count)
                (let [symbol (common-symbol word)
                      rare-counts (get tags symbol {})]
                  (assoc (dissoc tags word)
                    symbol
                    (reduce (fn [rare-counts [tag count]]
                              (assoc rare-counts tag (+ count (get rare-counts tag 0))))
                            rare-counts
                            (seq counts))))
                tags)))
          tags
          (keys tags)))

(defn get-tags
  ([]
   (with-open [reader (io/reader "/Users/colin/dev/nlp-columbia/data/hmm/gene.train")]
     (get-tags reader)))
  ([reader]
   (loop [sentences (sentences (line-seq reader))
          tags {}
          counts {}]
     (if (seq sentences)
       (let [items (map #(str/split (str/trim %) #"\s+") (first sentences))
             tag-sentence (map second items)]
         (recur (rest sentences) (update-tags items tags) (update-counts items counts)))
       [tags (dissoc counts [:*] [:STOP])]))))

(defn q
  "Calculates conditional probability q(yi | yi-1, yi-2)"
  [trigram counts]
  (assert (< 0 (get counts (subvec trigram 0 2) 0)) (str trigram))
  (/ (get counts trigram 0)
     (get counts (subvec trigram 0 2) 0)))

(defn e [tag count totals]
  (/ count (get totals tag)))

(defn probability [k u v w v-tag-count pi counts totals]
  (let [pi-val (get pi [(dec k) w u])]
    (if (= pi-val 0)
      0
      (* pi-val
         (q [w u v] counts)
         (e v v-tag-count totals)))))

(defn calc-max
  "Returns a function useful for calculating max with reduce"
  [f]
  (fn [item1 item2]
    (if (< (f item1) (f item2)) item2 item1)))

(defn update-pi-bp
  "Updates pi and bp for a given u, v at position k"
  [[pi bp] [u v] k sentence tags counts totals K-2 common-symbol]
  (let [word (nth sentence (dec k))
        tag-counts (get tags word (get tags (common-symbol word)))
        tag-count (get tag-counts v 0)
        max-item (reduce (calc-max first)
                         (map (fn [w]
                                [(probability k u v w tag-count pi counts totals) w])
                              K-2))]
    [(assoc pi [k u v] (first max-item))
     (assoc bp [k u v] (second max-item))]))

(defn calculate-pi-bp
  "Calculate pi and bp for the given sentence"
  [sentence tags counts totals all-tags common-symbol]
  (let [n (count sentence)]
    (loop [k 1
           [K-2 K-1 K :as Ks] [#{:*} #{:*} all-tags]
           [pi bp] [{[0 :* :*] 1} {}]]
      (if (<= k n)
        (recur (inc k)
               (conj (subvec Ks 1) all-tags)
               (reduce #(update-pi-bp %1 %2 k sentence tags counts totals K-2 common-symbol)
                       [pi bp]
                       (for [u K-1 v K] [u v])))
        [pi bp]))))

(defn third [coll]
  (second (rest coll)))

(defn get-tag-sequence [sentence pi bp all-tags counts]
  (let [n (count sentence)
        K-1 (if (= 1 n) #{:*} all-tags)
        [u v prob] (reduce (calc-max third)
                           (for [u K-1
                                 v all-tags]
                             [u v (* (get pi [n u v])
                                     (q [u v :STOP] counts))]))]
    (loop [k (- n 2)
           reversed [v u]
           yk+2 v
           yk+1 u]
      (if (<= 1 k)
        (let [next (get bp [(+ k 2) yk+1 yk+2])]
          (recur (dec k)
                 (conj reversed next)
                 yk+1
                 next))
        (reverse reversed)))))


(defn part-1 [file-name]
  (let [tags (replace-rare (first (get-tags)) only-rare)
        totals (calculate-totals tags)]
    (with-open [reader (io/reader (str "/Users/colin/dev/nlp-columbia/data/hmm/" file-name))
                writer (io/writer (str "/Users/colin/dev/nlp-columbia/data/hmm/" (str/replace file-name \. \_) ".p1
                .out"))]
      (doseq [word (map str/trim (line-seq reader))]
        (if-not (str/blank? word)
          (let [word-tags (get tags word (get tags (only-rare word)))
                tag (reduce (fn [item1 item2]
                              (let [[tag1 count1] item1
                                    [tag2 count2] item2]
                                (if (< (e tag1 count1 totals)
                                       (e tag2 count2 totals))
                                  item2
                                  item1)))
                            (seq word-tags))]
            (.write writer (str word " " (first tag) "\n")))
          (.write writer "\n"))))))


(defn part-2-3 [file-name common-symbol part-name]
  (let [[tags counts] (get-tags)
        tags (replace-rare tags common-symbol)
        totals (calculate-totals tags)
        all-tags (set (keys totals))]
    (with-open [reader (io/reader (str "/Users/colin/dev/nlp-columbia/data/hmm/" file-name))
                writer (io/writer (str "/Users/colin/dev/nlp-columbia/data/hmm/"
                                       (str/replace file-name \. \_)
                                       "."
                                       part-name
                                       ".out"))]
      (doseq [sentence-seq (sentences (map str/trim (line-seq reader)))]
        (let [sentence (apply vector sentence-seq)
              [pi bp] (calculate-pi-bp sentence tags counts totals all-tags common-symbol)
              sentence-tags (get-tag-sequence sentence pi bp all-tags counts)]
          (assert (= (count sentence) (count sentence-tags)))
          (doseq [line (map #(str %1 " " %2 "\n") sentence sentence-tags)]
            (.write writer line))
          (.write writer "\n"))))))

(defn part-2 [file-name]
  (part-2-3 file-name only-rare "p2"))

(defn part-3 [file-name]
  (part-2-3 file-name common-symbol "p3"))

(defn get-counts
  "Unused now that we calculate our own counts"
  []
  (with-open [reader (io/reader "/Users/colin/dev/nlp-columbia/data/hmm/gene.counts")]
    (loop [lines (line-seq reader)
           tags {}
           counts {}]
      (if (seq lines)
        (let [next (first lines)
              items (str/split next #"\s+")
              count (Long/parseLong (first items))]
          (case (second items)
            "WORDTAG" (let [tag (nth items 2)
                            word (nth items 3)
                            tag-counts (get tags word {})]
                        (recur (rest lines)
                               (assoc tags word (assoc tag-counts tag count))
                               counts))
            ("1-GRAM" "2-GRAM" "3-GRAM") (recur (rest lines)
                                                tags
                                                (assoc counts (apply vector (drop 2 items)) count))
            (throw (RuntimeException. (str "Unexpected tag " (second items))))))
        [tags counts]))))
