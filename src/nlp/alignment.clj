(ns nlp.alignment
  (:import (java.io Writer))
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.pprint :as pp]
            [clojure.core.reducers :as r]
            [foldable.seq :as f]))

(defn assoc-items [n word items]
  (assoc n word (into (get n word #{}) items)))

(definline add-in [c key delta]
  `(assoc ~c ~key (+ (get ~c ~key 0) ~delta)))

(defn model-1-c [t sentences]
  (reduce (fn [c [sentence-e sentence-f]]
            (reduce (fn [c f]
                      (let [denom (reduce + (map #(get-in t [% f]) sentence-e))]
                        (reduce (fn [c e]
                                  (let [delta (/ (get-in t [e f]) denom)]
                                    (if (pos? delta)
                                      (let [c (add-in c e delta)
                                            c (add-in c [e f] delta)]
                                        c))))
                                c
                                sentence-e)))
                    c
                    sentence-f))
          {}
          sentences))

(defn model-2-c [t q sentences]
  (reduce (fn [c [sentence-e sentence-f]]
            (let [l (count sentence-e)
                  m (count sentence-f)]
              (reduce (fn [c j]
                        (let [f (sentence-f j)
                              denom (reduce + (map #(* (get q [% j l m])
                                                       (get-in t [(sentence-e %) f]))
                                                   (range l)))]
                          (reduce (fn [c i]
                                    (let [e (sentence-e i)
                                          delta (/ (* (get q [i j l m])
                                                      (get-in t [e f]))
                                                   denom)]
                                      (if (pos? delta)
                                        (let [c (add-in c e delta)
                                              c (add-in c [e f] delta)
                                              c (add-in c [j l m] delta)
                                              c (add-in c [i j l m] delta)]
                                          c))))
                                  c
                                  (range l))))
                      c
                      (range m))))
          {}
          sentences))

(defn updated-t
  "Given an n-map of the form returned by calculate-n and counts
  calculated from a current t value, return an updated t map."
  [n c]
  (r/fold 128
          (r/monoid merge hash-map)
          (fn [t e]
            (let [fs (get n e)
                  t-vals (reduce (fn [t-vals f]
                                   (assoc t-vals f (/ (double (get c [e f] 0))
                                                      (double (get c e 0)))))
                                 {}
                                 fs)]
              (assoc t e t-vals)))
          (f/foldable-seq (keys n))))

(defn initial-t
  "Given an n-map of the form returned by calculate-n, calculate the
  naive initial t values using IBM model 1. Returns a sparse nested
  map as suggested in the problem description, e->f->t."
  [n]
  (r/fold 128
          (r/monoid merge hash-map)
          (fn [t e]
            (let [fs (get n e)
                  t-val (/ 1.0 (count fs))
                  t-vals (reduce (fn [t-vals f]
                                   (assoc t-vals f t-val))
                                 {}
                                 fs)]
              (assoc t e t-vals)))
          (f/foldable-seq (keys n))))

(defn updated-q [sentence-lengths c]
  (reduce (fn [q [l m]]
            (reduce (fn [q j]
                      (reduce (fn [q i]
                                (assoc q [i j l m] (/ (double (get c [i j l m] 0))
                                                      (double (get c [j l m] 0)))))
                              q
                              (range l)))
                    q
                    (range m)))
          {}
          sentence-lengths))

(defn initial-q [sentence-lengths]
  (reduce (fn [q [l m]]
            (reduce (fn [q j]
                      (reduce (fn [q i]
                                (assoc q [i j l m] (/ 1.0 l)))
                              q
                              (range l)))
                    q
                    (range m)))
          {}
          sentence-lengths))

(defn sentence-lengths
  "Returns a set of vectors representing the pairs of sentence
  lengths seen in the corpus."
  [sentences]
  (reduce (fn [lengths [sentence-e sentence-f]]
            (conj lengths [(count sentence-e) (count sentence-f)]))
          #{}
          sentences))

(defn calculate-n
  "Returns a map whose keys are all words in sentences-from,
  and whose values are sets of the words from sentences-to
  that the key word may be mapped to."
  [sentences]
  (reduce (fn [n [sentence-e sentence-f]]
            (reduce (fn [n word]
                      (assoc-items (assoc-items n :NULL sentence-f)
                                   word
                                   sentence-f))
                    n
                    sentence-e))
          {}
          sentences))

(defn calc-max
  "Returns a function useful for calculating max with reduce"
  [f]
  (fn [item1 item2]
    (if (< (f item1) (f item2)) item2 item1)))

(defn write-alignments [calc sentences filename]
  (with-open [writer (io/writer (str "/Users/colin/dev/nlp-columbia/data/alignment/" filename))]
    (doseq [k (range (count sentences))]
      (let [[sentence-e sentence-f] (sentences k)
            l (count sentence-e)
            m (count sentence-f)]
        (doseq [j (range m)]
          (let [f (sentence-f j)
                [i _] (reduce (calc-max second)
                              (map #(vector % (calc % j l m (sentence-e %) f))
                                   (range l)))]
            (.write writer (str (inc k) " " i " " (inc j) "\n"))))))))

(defn blank-sentence? [sentence]
  (and (= (count sentence) 1)
       (str/blank? (first sentence))))

(defn to-sentence [line]
  (apply vector (str/split (str/trim line) #"\s+")))

(defn read-sentences
  "Read sentences from corpus data files, returned as a vector of pairs of
   sentences. Sentences are vectors of words."
  [file-stem]
  (with-open [reader-e (io/reader (str "/Users/colin/dev/nlp-columbia/data/alignment/" file-stem ".en"))
              reader-f (io/reader (str "/Users/colin/dev/nlp-columbia/data/alignment/" file-stem ".es"))]
    (apply vector
           (map #(vector (into [:NULL] (first %))
                         (second %))
                (filter #(not (or (blank-sentence? (first %))
                                  (blank-sentence? (second %))))
                        (map #(vector %1 %2)
                             (map to-sentence (line-seq reader-e))
                             (map to-sentence (line-seq reader-f))))))))

(defn calculate-t-model-1 [sentences n]
  (reduce (fn [t _]
            (let [c (model-1-c t sentences)]
              (updated-t n c)))
          (initial-t n)
          (range 5)))

(defn calculate-t-q-model-2 [sentences n initial-t]
  (let [lengths (sentence-lengths sentences)]
    (reduce (fn [[t q] _]
              (let [c (model-2-c t q sentences)
                    t (updated-t n c)
                    q (updated-q lengths c)]
                [t q]))
            [initial-t (initial-q lengths)]
            (range 4))))

(defn part-1 []
  (let [sentences (read-sentences "corpus")
        n (calculate-n sentences)
        t (calculate-t-model-1 sentences n)
        test-sentences (read-sentences "dev")]
    (write-alignments (fn [i j l m e f]
                        (get-in t [e f]))
                      test-sentences
                      "alignment_dev.p1.out")))

(defn part-2 []
  (let [sentences (read-sentences "corpus")
        n (calculate-n sentences)
        [t q] (calculate-t-q-model-2 sentences n (calculate-t-model-1 sentences n))
        test-sentences (read-sentences "test")]
    (write-alignments (fn [i j l m e f]
                        (* (get q [i j l m])
                           (get-in t [e f])))
                      test-sentences
                      "alignment_test.p2.out")))
