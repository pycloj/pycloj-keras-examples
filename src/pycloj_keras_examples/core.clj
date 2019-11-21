(ns  pycloj-keras-examples.core
  "Constrains the weights to be non-negative."
  (:require [keras.Sequential :as Sequential]
            [keras.Model :as Model]
            [keras.datasets.fashion-mnist :as fashion-mnist]
            [keras.layers.Flatten :as Flatten]
            [keras.layers.Dense :as Dense]))

(defmacro def+
  "binding => binding-form
  internalizes binding-forms as if by def.
  See http://clojuredocs.org/clojure.core/destructure ."
  {:added "1.9", :special-form true, :forms '[(def+ [bindings*])]}
  [& bindings]
  (let [bings (partition 2 (destructure bindings))]
    (sequence cat
              ['(do)
               (map (fn [[var value]] `(def ~var ~value)) bings)
               [(mapv (fn [[var _]] (str var)) bings)]])))

(def mnist-data ( fashion-mnist/load-data ))

(def+ [[train-images , train-labels], [test-images, test-labels]] (vec mnist-data))

(def model (keras/Sequential [
                           (layers/Flatten [] {:input_shape   [28,28]})
                           (layers/Dense [128] {:activation  "relu"})
                           (layers/Dense [10] {:activation "softmax" })]))



(keras.Sequntial/compile {:optimizer "adam"
               :loss "sparse_categorical_crossentropy"
               :metrics ["accuracy"]})

(keras.Sequntial/fit  model train-images train-labels {:epochs 10})

(defn -main
 
  "I don't do a whole lot ... yet."
  [& args]

  (println "Hello, World!"))
