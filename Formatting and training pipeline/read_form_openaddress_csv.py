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
from google.cloud import storage
from pyspark.sql.functions import *
from pyspark.sql.functions import lit
from pyspark.sql import Row
import random
import pyspark.sql.functions as F
from pyspark.sql.types import *

bucket = sys.argv[1]
output_dir = 'gs://'+bucket+'/pyspark_nlp/Correct_format_result/de/hh'
def start():
    builder = SparkSession.builder \
        .appName("Spark NLP") \
        .master("yarn")
    builder.config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.1")
    return builder.getOrCreate()

# grab command line args and store them as variables
bucket = sys.argv[1]
data_path= 'unprocessed-data/openaddr-collected-global/de/hh'


# preprocessing: Read data and union for all CSVs
def unionAll(dfs):
    print("the totall number of dataframes to merge:", len(dfs))
    df=dfs[0]
    for i in range(1,len(dfs)):
            print("We are currently merging df",i-1,"and dataframe", i)
            df=df.union(dfs[i])
    return df

def read_data(data_path):
    dfs=[]
    client=storage.Client()
    mybucket=client.bucket('postal-parser-28')
    for blob in mybucket.list_blobs(prefix=data_path):
        print("we are currently here:",blob.name)
        if ".csv" in blob.name:
            splits=blob.name.split('/')
            df=spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').option("encoding", "utf-8").load('gs://'+ sys.argv[1] +'/'+blob.name)
            dfs=dfs+[df]
    return unionAll(dfs)

    

if __name__ == "__main__":
    # create a SparkContext while checking if there is already SparkContext created
    try:
        spark = sparknlp.start()
        print('Created a SparkContext')
    except ValueError:
        warnings.warn('SparkContext already exists in this scope')

    print('retrieving data from {}'.format(data_path))
    df=read_data(data_path)
    print(df.count())
    print('Saving results dataframe...')
    df.write.mode("overwrite").parquet(output_dir)
    print('Results saved!! yay!')
    print('Done! Yay!!')
    spark.stop()
