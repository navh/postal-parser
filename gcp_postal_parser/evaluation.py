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


### paths ###
bucket = sys.argv[1]
model_path = 'gs://'+bucket+'/pyspark_nlp/model2'
train_path = 'gs://'+bucket+'/pyspark_nlp/format_result/on' 
test_path = 'gs://'+bucket+'/pyspark_nlp/format_result/on' 

def read_data(path):
    print("Reading data...")
    
    data=spark.read.parquet(path)
    data = data.limit(10000)
    
    return data

def build():
    builder = SparkSession.builder \
    .appName("Spark NLP Licensed") \
    .master("yarn")
    
    return builder.getOrCreate()
 
def embedding():
    print('Loading embeddings...')
    bert_annotator = BertEmbeddings.pretrained('bert_base_cased', 'en') \
        .setInputCols(["sentence",'token'])\
        .setOutputCol("embeddings")\
        .setCaseSensitive(False)\
        .setPoolingLayer(0)
    return bert_annotator
    
def get_metrics(data_path):
    
    data = read_data(data_path)
    bert_annotator=embedding()
    
    print("Getting labels...")
    data = bert_annotator.transform(data)
    
    print("Loading model...")
    model = NerDLModel.load(model_path)\
        .setInputCols(["sentence", "token", 'embeddings'])\
        .setOutputCol("ner")
    
    data.show()
    predictions = model.transform(data)
    
    df = predictions.select(F.explode(F.arrays_zip('token.result','label.result','ner.result')).alias("cols")) \
    .select(F.expr("cols['1']").alias("target"),
            F.expr("cols['2']").alias("prediction"))
            
    correct = df.filter(df.target == df.prediction).count()
    total = df.select("target").count()
    
    accuracy = correct/total
    print("Accuracy = {}".format(accuracy))

if __name__ == "__main__":
    # create a SparkContext while checking if there is already SparkContext created
    try:
        spark=build()
        print('Created a SparkContext')
    except ValueError:
        warnings.warn('SparkContext already exists in this scope')
        
    get_metrics(train_path)
