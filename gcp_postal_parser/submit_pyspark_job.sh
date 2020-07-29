CLUSTER_NAME=postal-parser
REGION=europe-west1
BUCKET_NAME=postal-parser-27


echo "Submitting the job"

gcloud dataproc jobs submit pyspark \
        --cluster ${CLUSTER_NAME} \
        --properties=spark.jars.packages=com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.1 \
     --driver-log-levels root=FATAL \
     sparkNLP.py \
     -- ${BUCKET_NAME}
