Validation-by-Scibert
===============================
1 Introduction
--------------
This repository is to illustrate the difference between DComment("Deep Code-Comment Understanding and Assessment") and general text classification model. We compare our model with the state-of-the-art scientific text classification model [Scibert](https://paperswithcode.com/task/sentence-classification).

2 Dataset
--------------
We use the public code-comment [dataset](http://www2.unibas.it/gscanniello/coherence/). All data is converted into json format in `preprocess/`.

3 Result
--------------
To run Scibert on the public code-comment dataset, use the script in `script/` instead of the original.

The simplified version of the output log is in `output/`. 

```
{
  "best_epoch": 4,
  "peak_cpu_memory_MB": 1280.92,
  "peak_gpu_0_memory_MB": 422,
  "training_duration": "0:10:48.458976",
  "training_start_epoch": 0,
  "training_epochs": 13,
  "epoch": 13,
  "training_COHERENT_P": 0.958720326423645,
  "training_COHERENT_R": 0.9607031941413879,
  "training_COHERENT_F1": 0.9597106575965881,
  "training_NOT_COHERENT_P": 0.9335664510726929,
  "training_NOT_COHERENT_R": 0.9303135871887207,
  "training_NOT_COHERENT_F1": 0.9319371581077576,
  "training_average_F1": 0.9458239078521729,
  "training_accuracy": 0.9493835171966256,
  "training_loss": 0.125472488856758,
  "training_cpu_memory_MB": 1280.92,
  "training_gpu_0_memory_MB": 414,
  "validation_COHERENT_P": 0.774193525314331,
  "validation_COHERENT_R": 0.9375,
  "validation_COHERENT_F1": 0.8480564951896667,
  "validation_NOT_COHERENT_P": 0.8769230842590332,
  "validation_NOT_COHERENT_R": 0.6195651888847351,
  "validation_NOT_COHERENT_F1": 0.7261145710945129,
  "validation_average_F1": 0.7870855331420898,
  "validation_accuracy": 0.8045454545454546,
  "validation_loss": 0.7502426498436502,
  "best_validation_COHERENT_P": 0.8299319744110107,
  "best_validation_COHERENT_R": 0.953125,
  "best_validation_COHERENT_F1": 0.8872727155685425,
  "best_validation_NOT_COHERENT_P": 0.9178082346916199,
  "best_validation_NOT_COHERENT_R": 0.72826087474823,
  "best_validation_NOT_COHERENT_F1": 0.8121212124824524,
  "best_validation_average_F1": 0.8496969640254974,
  "best_validation_accuracy": 0.8590909090909091,
  "best_validation_loss": 0.3986410472009863,
  "test_COHERENT_P": 0.8203389644622803,
  "test_COHERENT_R": 0.9416342377662659,
  "test_COHERENT_F1": 0.8768116235733032,
  "test_NOT_COHERENT_P": 0.8672566413879395,
  "test_NOT_COHERENT_R": 0.6490066051483154,
  "test_NOT_COHERENT_F1": 0.7424242496490479,
  "test_average_F1": 0.8096179366111755,
  "test_accuracy": 0.8333333333333334,
  "test_loss": 0.4222671856673864
}
```

More detailed model and output can be downloaded [here]().

