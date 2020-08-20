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
from pyspark.sql import Row
from pyspark.sql.types import *
from typing import List

training_split = 0.8

bucket = sys.argv[1]
inputdir = 'gs://'+bucket+'/processed-data/training/'
outputfile = 'gs://'+bucket+'/pyspark_nlp/result'
modeldir = 'gs://'+bucket+'/pyspark_nlp/model_final'
graph_dir='//'+bucket+'/pyspark_nlp/graph/'
multilang = False


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
    
def with_column_index(sdf):
    new_schema = StructType(sdf.schema.fields + [StructField("ColumnIndex", LongType(), False),])
    return sdf.rdd.zipWithIndex().map(lambda row: row[0] + (row[1],)).toDF(schema=new_schema)

#getting labels and annotate them
def get_label(inp):
    x=inp['label']
    text=inp['text']
    data2=[]
    data={}
    for i in range(len(x)):
        data=x[i].asDict()
        data['metadata']=data['metadata'].asDict()
        data=Row(**data)
        data2.append(data)
    return { 'text':text, 'label':data2}
    
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

#our formatting pipeline procedure
def formating_pipeline(df):
    df=df.withColumnRenamed('data', 'label')
    df=df.withColumnRenamed('string', 'text')
    training_rdd = df.rdd.map(lambda row: row.asDict())
    label_rdd = training_rdd.map(lambda x: get_label(x))
   
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
    training_data= spark.createDataFrame(label_rdd, schema=Schema)
    formatting_model=get_formatting_model()
    training_data=formatting_model.transform(training_data)
    return training_data

#our training pipeline procedure
def training_pipeline(training_data):
    print('Get embedding...')
    bert_annotator=embedding(multilang)
    NER_pipeline=build_pipeline(bert_annotator)
    training_data = bert_annotator.transform(training_data)
    pipelineFit = train(spark,training_data,NER_pipeline)
    return pipelineFit

def build_pipeline(bert_annotator):
    print('Building DL pipeline...')

    nerTagger = NerDLApproach()\
        .setInputCols(["sentence", "token", 'embeddings'])\
        .setLabelColumn("label")\
        .setOutputCol("ner")\
        .setMaxEpochs(30)\
        .setLr(0.0001)\
        .setPo(0.005)\
        .setBatchSize(8)\
        .setRandomSeed(0)\
        .setVerbose(1)\
        .setValidationSplit(0.2)\
        .setEvaluationLogExtended(False) \
        .setEnableOutputLogs(False)\
        .setIncludeConfidence(True) \
        .setGraphFolder(graph_dir)

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
    data = data.sample(False,0.00003, seed=20000)
    data=formating_pipeline(data)
    splits = data.randomSplit([training_split, 1-training_split], 24)
    
    training_data = splits[0]
    test_data = splits[1]
    print("Training on {} addresses...".format(training_data.count()))
    training_data.limit(10).show(3)
    
    print('Get embedding...')
    bert_annotator=embedding(multilang)
    NER_pipeline=build_pipeline(bert_annotator)
    training_data = bert_annotator.transform(training_data)
    model = train(spark,training_data,NER_pipeline)
    
    print("Training data:")
    get_metrics(training_data, model)
    
    print("Test data")
    get_metrics(test_data, model)
    
    print('Done! Yay!!')
    spark.stop()
