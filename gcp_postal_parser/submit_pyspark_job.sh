CLUSTER_NAME=postal-parser
REGION=europe-west1
BUCKET_NAME=postal-parser-27
NUM_TRAIN=`ls [TRAIN DATA PATH] | wc -l`
NUM_TEST=`ls [TEST DATA PATH] | wc -l`


echo "Submitting the job"

gcloud dataproc jobs submit pyspark \
        --cluster ${CLUSTER_NAME} \
        --properties=spark.jars.packages=com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.1 \
     --driver-log-levels root=FATAL \
     sparkNLP.py \
     -- ${BUCKET_NAME} \
     -- ${NUM_TRAIN} \
     -- ${NUM_TEST}
