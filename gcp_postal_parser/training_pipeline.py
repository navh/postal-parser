import sys
import pyspark as ps
import warnings
import re
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from sparknlp.training import CoNLL
import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *

bucket = sys.argv[1]
inputdir = 'gs://'+bucket+'/pyspark_nlp/Correct_format_result/de/hh'
outputfile = 'gs://'+bucket+'/pyspark_nlp/result'
modeldir = 'gs://'+bucket+'/pyspark_nlp/model2'
multilang = False
CRF = False
training_split = 0.8

def start(gpu):
    builder = SparkSession.builder \
                .appName("Spark NLP") \
                .master("yarn") \
                .config("spark.dynamicAllocation.enabled", "True") \
                .config("spark.executor.memory", "21G") \
                .config("spark.executor.instances", 129) \
                .config("spark.driver.cores",1)
    if gpu:
        builder.config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp-gpu_2.11:2.5.1")
    else:
        builder.config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.1")
    return builder.getOrCreate()
    
def embedding(multilang=False):
    print('Loading embeddings...')
    if multilang:
        bert_annotator = BertEmbeddings.pretrained('bert_multi_cased', 'xx') \
            .setInputCols(["sentence",'token'])\
            .setOutputCol("embeddings")\
            .setCaseSensitive(False)\
            .setPoolingLayer(0)
    else:
        bert_annotator = BertEmbeddings.pretrained('bert_base_cased', 'en') \
            .setInputCols(["sentence",'token'])\
            .setOutputCol("embeddings")\
            .setCaseSensitive(False)\
            .setPoolingLayer(0)
    return bert_annotator
    

def build_pipeline(bert_annotator,CRF=False):
    print('Building DL pipeline...')

    nerTagger = NerDLApproach()\
        .setInputCols(["sentence", "token", 'embeddings'])\
        .setLabelColumn("label")\
        .setOutputCol("ner")\
        .setMaxEpochs(30)\
        .setLr(0.00001)\
        .setPo(0.005)\
        .setBatchSize(8)\
        .setRandomSeed(0)\
        .setVerbose(1)\
        .setValidationSplit(0.2)\
        .setEvaluationLogExtended(False) \
        .setEnableOutputLogs(False)\
        .setIncludeConfidence(True)
    if CRF:
        nerTagger = NerCrfApproach() \
                .setInputCols(["sentence", "token", "embeddings"]) \
                .setLabelColumn("label") \
                .setOutputCol("ner") \
                .setMaxEpochs(20)

    NER_pipeline = Pipeline(
                        stages = [
                                  bert_annotator,
                                  nerTagger
                                  ])
    return NER_pipeline
    
def train(spark, data, NER_pipeline):
    print('Fitting data for training...')
    model =  NER_pipeline.fit(data)
    print('Saving model...')
    model.stages[1].write().overwrite().save(modeldir)
    
    return model

def get_metrics(data, model):
    
    print("Getting embeddings...")
    data = bert_annotator.transform(data)
    predictions = model.transform(data)
    
    df = predictions.select(F.explode(F.arrays_zip('token.result','label.result','ner.result')).alias("cols")) \
    .select(F.expr("cols['1']").alias("target"),
            F.expr("cols['2']").alias("prediction"))
            
    correct = df.filter(df.target == df.prediction).count()
    total = df.select("target").count()
    
    accuracy = 100*correct/total
    print("Accuracy = {}".format(accuracy))

if __name__ == "__main__":
    # create a SparkContext while checking if there is already SparkContext created
    try:
        spark = start(False)
        print('Created a SparkSession')
    except ValueError:
        warnings.warn('SparkSession already exists in this scope')

    print('Retrieving data from {}'.format(inputdir))
        #train_txt = spark.sparkContext.textFile(inputdir +training_data_path)
    data=spark.read.parquet(inputdir)
    print("There are {} training addresses.".format(data.count()))
    data = data.sample(False,0.45, seed=0)
    splits = data.randomSplit([training_split, 1-training_split], 24)
    training_data = splits[0]
    test_data = splits[1]
    print("Training on {} addresses...".format(training_data.count()))
    training_data.limit(10).show(3)
    
    print('Get embedding...')
    bert_annotator=embedding(multilang)
    NER_pipeline=build_pipeline(bert_annotator, CRF)
    training_data = bert_annotator.transform(training_data)
    model = train(spark,training_data,NER_pipeline)
    
    print("Training data:")
    get_metrics(training_data, model)
    
    print("Test data")
    get_metrics(test_data, model)
    
    print('Done! Yay!!')
    spark.stop()
