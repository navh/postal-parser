###Import Packages ###

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml import Pipeline, PipelineModel

import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *

### set variable ###

model_path = 'Ner_model'
data_path = 'put in later'
output_path = 'output path'

###Define functions ###

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

def read_csv():
    ###Read csv to spark data frame###
    pass

def get_chunks(tokens, ner):
  chunks = []
  tags = []
  for i in range(len(tokens)):
    tag = ner[i][2:]
    position = ner[i][:2]
    if position == 'B-':
      chunks.append(tokens[i])
      tags.append(tag)
    if position == 'I-':
      chunks[-1] = chunks[-1] + ' ' + tokens[i]
  return (tags, chunks)

def get_colvals(text, tokens, ner):
  keys, vals = get_chunks(tokens, ner)
  keys = ['text'] + keys
  vals = [text] + vals
  return dict(zip(keys, vals))
  
  
def format_output():
    ###Start spark###
    
    print('Initializing spark session')
    gpu_access=False 
    spark = start(gpu=gpu_access)
    
    ###create pipeline##
    print('Creating prediction pipeline...')
    
    document = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

    sentence = SentenceDetector()\
        .setInputCols(['document'])\
        .setOutputCol('sentence')

    token = Tokenizer()\
        .setInputCols(['sentence'])\
        .setOutputCol('token')

    bert = BertEmbeddings.pretrained('bert_base_cased', 'en') \
        .setInputCols(["sentence",'token'])\
        .setOutputCol("embeddings")\
        .setCaseSensitive(False)

    loaded_ner_model = NerDLModel.load(path_to_model)\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

    converter = NerConverter()\
        .setInputCols(["document", "token", "ner"])\
        .setOutputCol("ner_span")

    ner_prediction_pipeline = Pipeline(
        stages = [
            document,
            sentence,
            token,
            bert,
            loaded_ner_model,
            converter])
    
    empty_data = spark.createDataFrame([['']]).toDF("text")
    prediction_model = ner_prediction_pipeline.fit(empty_data)
    
    ###Get test_Data###
    data = read_csv()
    
    ### Get predictions ###
    print('Getting predictions...')
    predictions = prediction_model.transform(data)
    predictions = predictions.selectExpr('text', 'token.result as tokens', 'ner.result as ner' )
    
    print('Formatting predictions...')
    tokens = predictions.rdd.map(lambda x: get_colvals(x[0], x[1], x[2]))
    final_df = tokens.map(lambda x: Row(**x)).toDF()
    final_df.show()
    
    ###Saving df as csv###
    print('Saving results dataframe...')
    final_df.write.format("csv").save(output_path)
    print('Results saved!! yay!')

if __name__ = "__main__":
    format_output()
    