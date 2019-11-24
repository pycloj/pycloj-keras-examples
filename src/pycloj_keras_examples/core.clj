(ns  pycloj-keras-examples.core
  "Constrains the weights to be non-negative."
  (:require [keras]
            [keras.Sequential :as Sequential :refer [Sequential]]
            [keras.Model :as Model]
            [keras.datasets.fashion-mnist :as fashion-mnist]
            [keras.layers.Flatten :as Flatten :refer [Flatten]]
            [keras.layers.Dense :as Dense :refer [Dense]]
            [libpython-clj.python :as py]))


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

(def f (Flatten [] {:input_shape [28,28]}))

(def layers [f
             ;; (Dense [128] {:activation "relu"})
             ;; (Dense [10] {:activation "softmax"})
             ])

(def model (Sequential
            :layers
            layers))

(Sequential/compile model
                    :optimizer "adam"
                    :loss      "sparse_categorical_crossentropy"
                    :metrics   ["accuracy"])

(py/get-attr f "input_shape")

(keras.Sequential/fit model
                      :x train-images
                      :y train-labels
                      :epochs 10)


(let [f (Flatten [] {:input_shape [28,28]})]
  (py/call-attr f "input_shape"))




(defn -main
  
  "I don't do a whole lot ... yet."
  [& args]

  (println "Hello, World!"))
