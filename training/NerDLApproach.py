###import packages

print('Importing required packages...')

from sklearn.metrics import classification_report, accuracy_score

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
import pyspark.sql.functions as F

import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from sparknlp.training import CoNLL


###Set variables###

training_data_path = 'test_CoNLL_addresses.txt'
test_data_path = 'test_CoNLL_addresses.txt')
path_to_model = 'NER_model1'
Test = False

###Start session###

print('Starting spark session...')

spark = sparknlp.start()
def start(gpu):
    builder = SparkSession.builder \
        .appName("Spark NLP") \
        .master("local[*]") \
        .config("spark.driver.memory", "8G") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")\
        .config("spark.kryoserializer.buffer.max", "1000M")
    if gpu:
        builder.config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.5.1")
    else:
        builder.config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.1")

    return builder.getOrCreate()

gpu_access=True 
spark = start(gpu=gpu_access)

###read data ###

print('Loading data...')
training_data = CoNLL().readDataset(spark, training_data_path)

if Test== True:
    test_data = CoNLL().readDataset(spark, test_data_path)


###Get embeddings ###

print('Loading embeddings...')

bert_annotator = BertEmbeddings.pretrained('bert_base_cased', 'en') \
 .setInputCols(["sentence",'token'])\
 .setOutputCol("embeddings")\
 .setCaseSensitive(False)\
 .setPoolingLayer(0)
 
training_data = bert_annotator.transform(training_data)

###Building NER pipeline###

print('Building DL pipeline...')

nerTagger = NerDLApproach()\
  .setInputCols(["sentence", "token", 'embeddings'])\
  .setLabelColumn("label")\
  .setOutputCol("ner")\
  .setMaxEpochs(10)\
  .setLr(0.001)\
  .setPo(0.005)\
  .setBatchSize(8)\
  .setRandomSeed(0)\
  .setVerbose(1)\
  .setValidationSplit(0.2)\
  .setEvaluationLogExtended(True) \
  .setEnableOutputLogs(True)\
  .setIncludeConfidence(True)

NER_pipeline = Pipeline(
    stages = [
    bert_annotator,
    nerTagger
  ])

print('Training model...')

Ner_model = NER_pipeline.fit(training_data)

print('Done training.')
print('Saving model...')

Ner_model.stages[1].write().overwrite().save(path_to_model)


###Predictions###

print('getting predictions...')
training_predictions = Ner_model.transform(training_data)

if Test==True:
    test_predictions = Ner_model.transform(test_data)

###Evaluation###

print('Converting predictions to pandas df...')
df_train = training_predictions.select(F.explode(F.arrays_zip('token.result','label.result','ner.result')).alias("cols")) \
.select(F.expr("cols['0']").alias("token"),
        F.expr("cols['1']").alias("ground_truth"),
        F.expr("cols['2']").alias("prediction")).toPandas()
        
if Test == True:
    df_test = test_predictions.select(F.explode(F.arrays_zip('token.result','label.result','ner.result')).alias("cols")) \
    .select(F.expr("cols['0']").alias("token"),
        F.expr("cols['1']").alias("ground_truth"),
        F.expr("cols['2']").alias("prediction")).toPandas()
        

print('Evaluation report for training data:')
print(classification_report(df_train.ground_truth, df_train.prediction))
print(accuracy_score(df_train.ground_truth, df_train.prediction))

if Test== True:
    print(classification_report(df_test.ground_truth, df_test.prediction))
    print(accuracy_score(df_test.ground_truth, df_test.prediction))

print('Done! Yay!!')
