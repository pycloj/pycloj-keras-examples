(ns  pycloj-keras-examples.core
  "Constrains the weights to be non-negative."
  (:require [keras]
            [keras.Sequential :as Sequential :refer [Sequential]]
            [keras.Model :as Model]
            [keras.datasets.fashion-mnist :as fashion-mnist]
            [keras.layers.Flatten :as Flatten :refer [Flatten]]
            [keras.layers.Dense :as Dense :refer [Dense]]
            [libpython-clj.python :as py]))

(defonce keras (py/import-module "keras"))
(defonce np (py/import-module "numpy"))

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

(def model (py/call-attr  keras "Sequential" [(Flatten)
                                              (Dense 128
                                                     :activation "relu")
                                              (Dense 10
                                                     :activation "softmax")]))

(Sequential/compile model
                    "adam"
                    :loss      "sparse_categorical_crossentropy"
                    :metrics   '("accuracy"))

(Sequential/fit model
                :x train-images
                :y train-labels
                :epochs 10)


(def+ [test-loss test-acc]
  (Sequential/evaluate model
                       :x test-images
                       :y test-labels
                       :verbose 2) )

(def predictions (Sequential/predict model
                                    test-images))

(py/call-attr np "argmax" (first predictions))
(first test-labels)
