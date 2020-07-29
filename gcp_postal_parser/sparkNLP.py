from sklearn.metrics import classification_report, accuracy_score
import sys
import os
import pyspark as ps
import warnings
import re
from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from sparknlp.training import CoNLL
import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *

training_dir = '/training/'
test_data_dir = '/test/'
loaded_model=False
test=False


# grab command line args and store them as variables
bucket = sys.argv[1]
inputdir = 'gs://'+bucket+'/conll-data/'
outputfile = 'gs://'+bucket+'/pyspark_nlp/result'
modeldir = 'gs://'+bucket+'/pyspark_nlp/model'


# preprocessing: Read data
def unionAll(dfs):
    return reduce(DataFrame.unionAll, dfs)

def read_data(dir_path):
    dfs = []
    files = os.listdir(dir_path)
    for filename in files:
        dfs = dfs.append(CoNLL().readDataset(spark, dir_path+filename))
    return unionAll(dfs)

# preprocessing: make embeddings
def embedding():
    print('Loading embeddings...')
    bert_annotator = BertEmbeddings.pretrained('bert_base_cased', 'en') \
        .setInputCols(["sentence",'token'])\
        .setOutputCol("embeddings")\
        .setCaseSensitive(False)\
        .setPoolingLayer(0)
    return bert_annotator


# build a pipeline following below order
def build_pipeline(bert_annotator):
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
        .setEvaluationLogExtended(False) \
        .setEnableOutputLogs(False)\
        .setIncludeConfidence(True)

    NER_pipeline = Pipeline(
                        stages = [
                                  bert_annotator,
                                  nerTagger
                                  ])
    return NER_pipeline


#training
def train(spark,data, NER_pipeline):
    
    
    
        print('fitting data for training...')
        model =  NER_pipeline.fit(data)
        print('Saving model...')
        model.stages[1].write().overwrite().save(modeldir)
        return model


#predicting
def predict(model,data):
    print('making predictions on data...')
    predictions = model.transform(data)

    print('Converting predictions to pandas df...')
    df = predictions.select(F.explode(F.arrays_zip('token.result','label.result','ner.result')).alias("cols")) \
    .select(F.expr("cols['0']").alias("token"),
            F.expr("cols['1']").alias("ground_truth"),
            F.expr("cols['2']").alias("prediction")).toPandas()

    print('Evaluation report')
    print(classification_report(df.ground_truth, df.prediction))
    print(accuracy_score(df.ground_truth, df.prediction))

#loading model
def loading_model(inputdir):
   
    loaded_ner_model = NerDLModel.load(modeldir)\
        .setInputCols(["sentence", "token", 'embeddings'])\
        .setOutputCol("ner")
    return loaded_ner_model
 
                                       
if __name__ == "__main__":
    # create a SparkContext while checking if there is already SparkContext created
    try:
        spark = sparknlp.start()
        print('Created a SparkContext')
    except ValueError:
        warnings.warn('SparkContext already exists in this scope')

    print('retrieving data from {}'.format(inputdir))
    if not loaded_model:
        
        training_data = read_data(training_data_dir)
        training_data.show(3)
    if test is True:
        test_data=read_data(test_data_dir)

    print('get embedding...')
    bert_annotator=embedding()
    NER_pipeline=build_pipeline(bert_annotator)
    if not loaded_model:
        training_data = bert_annotator.transform(training_data)
        pipelineFit = train(spark,training_data,NER_pipeline)
    if test is True:
        test_data = bert_annotator.transform(test_data)
        model=loading_model(inputdir)
        predict(model,test_data)
    print('Done! Yay!!')
    spark.stop()
