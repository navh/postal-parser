CLUSTER_NAME=postal-parser-new1
REGION=europe-west1
BUCKET_NAME=postal-parser-28


echo "Submitting the job"

gcloud dataproc jobs submit pyspark \
--cluster ${CLUSTER_NAME} \
--properties=spark.jars.packages=com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.4,spark.submit.deployMode=cluster,spark.executor.instances=129,spark.executor.cores=1,spark.executor.memory=21G \
--driver-log-levels root=FATAL \
postal_parse_pipeline.py \
-- ${BUCKET_NAME}

