# postal-parser
Probabilistic postal pyspark parser project pod.

### Tasks

- [x] Download OpenAddress data - beautiful csv
- [x] Determine data format for the training pipeline (Pyspark dataframe)
- [x] Create data pre-processing pipeline
- [x] Get formatted distributed data (parquet) ready for the training pipeline  
- [x] Prepare training pipeline
- [ ] Train on the public addresses
- [ ] Experiment with embeddings (Multi-language, ELMO, etc), and language models 

### Formatting pipeline
The NerDLApproach() is explained in our notebook [here.](https://github.com/Beaver-2020/postal-parser/blob/master/training/NERDLApproach.ipynb)

Helpful resources for Spark-nlp:
 - [Spark NLP Walkthrough, powered by TensorFlow](https://medium.com/@saif1988/spark-nlp-walkthrough-powered-by-tensorflow-9965538663fd)
 - [Spark-nlp documentation](https://nlp.johnsnowlabs.com/docs/en/quickstart)

