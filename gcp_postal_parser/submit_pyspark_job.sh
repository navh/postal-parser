CLUSTER_NAME=postal-parser
REGION=us-central1
BUCKET_NAME=postal-parser


echo "Submitting the job"

gcloud dataproc jobs submit pyspark \ 
	--cluster ${CLUSTER_NAME} \ 
	--properties=spark.jars.packages=JohnSnowLabs:spark-nlp:2.5.1 \ 
        --driver-log-levels root=FATAL \ 
        sparkNLP.py \ 
	-- ${BUCKET_NAME}