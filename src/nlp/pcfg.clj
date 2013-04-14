(ns nlp.pcfg
  (:require [clojure.edn :as edn]
            [clojure.string :as str]
            [clojure.java.io :as io]))

(defn inc-in
  "Increments the count in map for key"
  [map key]
  (assoc map key (inc (get map key 0))))

(defn add-in
  "Increments the count in map for key by count"
  [map key count]
  (assoc map key (+ count (get map key 0))))

(defn third [coll]
  (second (rest coll)))

(defn unary-rule? [item]
  (and (vector? item)
       (= 2 (count item))))

(defn binary-rule? [item]
  (and (vector? item)
       (= 3 (count item))))

(defn update-counts
  "Returns a new map with all the counts from sentence updated.
  For all sentences, the non-terminal count is increased. For
  unary rules the count using the sentence itself as the key
  is incremented. For binary rules the count using the vector
  representing the rule applied (e.g. [S NP VP]) is
  incremented."
  [counts sentence]
  (let [updated (inc-in counts (first sentence))]
    (case (count sentence)
      2 (inc-in updated sentence)
      3 (let [phrase1 (second sentence)
              phrase2 (third sentence)
              updated (update-counts updated phrase1)
              updated (update-counts updated phrase2)]
          (inc-in updated [(first sentence) (first phrase1) (first phrase2)])))))

(defn update-rare-counts
  "Returns a new map containing counts updated for rare words.
  All words with a total count of less than five have their
  unary rules removed, and their counts added to the corresponding
  unary rule using _RARE_."
  [counts word-counts]
  (reduce (fn [counts [[non-terminal word :as rule] count]]
            (if (< (word-counts word) 5)
              (let [new-rule [non-terminal "_RARE_"]]
                (add-in (dissoc counts rule) new-rule count))
              counts))
          counts
          (filter #(unary-rule? (first %)) counts)))

(defn update-rare-word-counts
  "Returns a new map containing counts updated for rare words.
  All words with a total count of less than five have their
  unary rules removed, and their counts added to the corresponding
  unary rule using _RARE_."
  [word-counts]
  (reduce (fn [word-counts [word count]]
            (add-in (dissoc word-counts word) "_RARE_" count))
          word-counts
          (filter #(< (second %) 5) word-counts)))

(defn counts
  "Calculates all the counts for the sentences in the given file,
  or from parse_train.dat by default."
  ([training-file]
   (with-open [reader (io/reader (str "/Users/colin/dev/nlp-columbia/data/pcfg/" training-file))]
     (reduce update-counts
             {}
             (map (comp edn/read-string str/trim) (line-seq reader))))))

(defn word-counts
  "Using the calculated counts from the sentences, return a map
  of words to their counts."
  [counts]
  (reduce (fn [word-counts [[_ word] count]]
            (add-in word-counts word count))
          {}
          (filter #(unary-rule? (first %)) counts)))

(defn write-counts
  ([counts]
   (with-open [writer (io/writer "/Users/colin/dev/nlp-columbia/data/pcfg/parse_train.counts.out")]
     (write-counts counts writer)))
  ([counts writer]
   (doseq [[item count] counts]
     (if (vector? item)
       (.write writer (str count " "
                           (if (= 2 (clojure.core/count item)) "UNARYRULE" "BINARYRULE") " "
                           (str/join " " item) "\n"))
       (.write writer (str count " NONTERMINAL " item "\n"))))))

(defn initial-pi [counts word-counts non-terminals sentence]
  (reduce (fn [pi [i non-terminal]]
            (let [raw-word (get sentence (dec i))
                  word (if (nil? (get word-counts raw-word))
                         "_RARE_"
                         raw-word)
                  value (/ (counts [non-terminal word] 0)
                           (counts non-terminal))]
              (if (> value 0)
                (assoc pi [i i non-terminal] value)
                pi)))
          {}
          (for [i (range 1 (inc (count sentence)))
                non-terminal non-terminals]
            [i non-terminal])))

(defn binary-rules-by-nonterminal [counts]
  (reduce (fn [rules [non-terminal _ _ :as rule]]
            (assoc rules non-terminal (conj (get rules non-terminal []) rule)))
          {}
          (apply vector (filter binary-rule? (keys counts)))))

(defn calc-max
  "Returns a function useful for calculating max with reduce"
  [f]
  (fn [item1 item2]
    (if (> (f item1) (f item2)) item1 item2)))

(defn calculate-pi-bp [counts word-counts sentence]
  (let [non-terminals (apply vector (filter string? (keys counts)))
        binary-rules (binary-rules-by-nonterminal counts)
        n (count sentence)
        initial-pi (initial-pi counts word-counts non-terminals sentence)]
    (reduce (fn [[pi bp] [i j non-terminal]]
              (let [[value rule s] (reduce
                                     (calc-max first)
                                     [0 nil 0]
                                     (for [[X Y Z :as rule] (get binary-rules non-terminal [])
                                           s (range i j)]
                                       (let [q (/ (counts rule 0)
                                                  (counts X))
                                             value (* q (pi [i s Y] 0) (pi [(inc s) j Z] 0))]
                                         [value rule s])))]
                (if (> value 0)
                  [(assoc pi [i j non-terminal] value)
                   (assoc bp [i j non-terminal] [rule s])]
                  [pi bp])))
            [initial-pi {}]
            (for [length (range 1 n)
                  i (range 1 (inc (- n length)))
                  non-terminal non-terminals]
              [i (+ i length) non-terminal]))))

(defn create-tree-node [bp i j non-terminal sentence]
  (if (= i j)
    [non-terminal (get sentence (dec i))]
    (let [[[X Y Z] s] (bp [i j non-terminal])]
      [X
       (create-tree-node bp i s Y sentence)
       (create-tree-node bp (inc s) j Z sentence)])))

(defn parse-sentence [counts word-counts sentence]
  (let [[pi bp] (calculate-pi-bp counts word-counts sentence)]
    (create-tree-node bp 1 (count sentence) "SBARQ" sentence)))

(defn jsonify [item]
  (if (vector? item)
    (str "[" (str/join \, (map jsonify item)) "]")
    (pr-str item)))

(defn part1 []
  (let [counts (counts "parse_train.dat")
        word-counts (word-counts counts)]
    (write-counts (update-rare-counts counts word-counts))))

(defn parse [training-file input-file output-file]
  (let [raw-counts (counts training-file)
        raw-word-counts (word-counts raw-counts)
        counts (update-rare-counts raw-counts raw-word-counts)
        word-counts (update-rare-word-counts raw-word-counts)]
    (with-open [reader (io/reader (str "/Users/colin/dev/nlp-columbia/data/pcfg/" input-file))
                writer (io/writer (str "/Users/colin/dev/nlp-columbia/data/pcfg/" output-file))]
      (doseq [sentence (map #(str/split (str/trim %) #"\s+") (line-seq reader))]
        (.write writer (jsonify (parse-sentence counts word-counts sentence)))
        (.write writer "\n")))))

(defn test-part2 []
  (parse "parse_train.dat" "parse_dev.dat" "parse_dev.out"))

(defn part2 []
  (parse "parse_train.dat" "parse_test.dat" "parse_test.p2.out"))

(defn test-part3 []
  (parse "parse_train_vert.dat" "parse_dev.dat" "parse_dev.out"))

(defn part3 []
  (parse "parse_train_vert.dat" "parse_test.dat" "parse_test.p3.out"))
