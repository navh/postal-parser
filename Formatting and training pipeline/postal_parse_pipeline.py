import sys
import sparknlp
import pyspark
import re
import warnings
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql import *
from typing import List
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.base import *


training_split = 0.8

bucket = sys.argv[1]
inputdir = 'gs://'+bucket+'/processed-data/training'

modeldir = 'gs://'+bucket+'/pyspark_nlp/model_final'
graph_dir='gs://'+bucket+'/pyspark_nlp/graph'

#spark_session
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

#getting embeddings
def embedding():
    print('Loading embeddings...')
    bert_annotator = BertEmbeddings.pretrained('bert_base_cased', 'en') \
        .setInputCols(["sentence",'token'])\
        .setOutputCol("embeddings")\
        .setCaseSensitive(False)\
        .setPoolingLayer(0)
    return bert_annotator
    
#formating model
def get_formatting_model():
    document = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
    
    sentence = SentenceDetector()\
        .setInputCols(['document'])\
        .setOutputCol('sentence')
    tokenizer = Tokenizer() \
        .setInputCols(["sentence"]) \
        .setOutputCol("token")
    
    pos = PerceptronModel.pretrained() \
        .setInputCols(["sentence", "token"]) \
        .setOutputCol("pos")
    
    
    formatting_pipeline = Pipeline(
                                   stages = [
                                             document,
                                             sentence,
                                             tokenizer,
                                             pos
                                             ]
                                   )
    empty_data = spark.createDataFrame([['']]).toDF("text")
    formatting_model = formatting_pipeline.fit(empty_data)
    return formatting_model

#buiding training_pipeline
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
def train(spark, data, NER_pipeline):
    print('Fitting data for training...')
    model =  NER_pipeline.fit(data)
    print('Saving model...')
    model.stages[1].write().overwrite().save(modeldir)
    
    return model



#prediction
def get_metrics(data):
    
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

        accuracy = 100*correct/total
        print("Accuracy = {}".format(accuracy))

#getting labels and annotate them
def createAnnotation(l):
    res=[]
    
    for i in range(len(l)):
        data=l[i].asDict()
        a=Row(
              annotatorType="named_entity",
              begin=data['begin'],
              end=data['end'],
              result=data['result'],
              metadata=data['metadata'].asDict(),
              embeddings=[0.]
              )
        res.append(a)
    
    return res

#our formatting pipeline procedure
def format(df):
    training_rdd = df.rdd.map(lambda row: row.asDict())
    data_rdd = training_rdd.map(lambda  x: {'text':x['text'], 'label':createAnnotation(x['label'])})
    
   
    Schema = StructType([StructField("text", StringType(), False),
                        StructField('label',ArrayType(
                        StructType([
                                   StructField("annotatorType", StringType(), False),
                                   StructField("begin", IntegerType(), False),
                                   StructField("end", IntegerType(), False),
                                   StructField("result", StringType(), False),
                                   StructField("metadata",  MapType(StringType(), StringType())),
                                   StructField("embeddings",  ArrayType(FloatType()), False)
                                   ])))])
    training_data= spark.createDataFrame(data_rdd, schema=Schema)
    formatting_model=get_formatting_model()
    training_data=formatting_model.transform(training_data)
    training_data=training_data.filter(size(col("sentence.end")) <= 1)
    return training_data

#our training pipeline procedure
def training_pipeline(training_data):
    print('Get embedding...')
    bert_annotator=embedding()
    NER_pipeline=build_pipeline(bert_annotator)
    training_data = bert_annotator.transform(training_data)
    pipelineFit = train(spark,training_data,NER_pipeline)
    return pipelineFit


if __name__ == "__main__":
    # create a SparkContext while checking if there is already SparkContext created
    try:
        spark=start(False)
        print('Created a SparkSession')
    except ValueError:
        warnings.warn('SparkSession already exists in this scope')
    
    print("we are in our first step, formatting pipeline")
    print('Retrieving data from {}'.format(inputdir))
    data=spark.read.parquet(inputdir)
    data=format(data)

    #change this part if you want to train on more than 0.00015 of the data
    print("we are in our next step, training pipeline")
    data =data.sample(False,0.0001, seed=0)

    #train-test split
    splits = data.randomSplit([training_split, 1-training_split], 24)
    training_data = splits[0]
    test_data = splits[1]


    print("Training on {} addresses...".format(training_data.count()))
    model=training_pipeline(training_data)

    #prediction
    get_metrics(training_data)
    print("predicting on test data")
    get_metrics(test_data)
    
    print('Done! Yay!!')
    spark.stop()
