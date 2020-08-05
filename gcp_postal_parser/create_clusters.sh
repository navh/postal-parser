CLUSTER_NAME=postal-parser-1
REGION=europe-west1
BUCKET_NAME=postal-parser-27


gcloud config set dataproc/region ${REGION}

echo "Creating a cluster"

gcloud beta dataproc clusters create ${CLUSTER_NAME} \
--region ${REGION} \
--master-machine-type e2-standard-4 \
--worker-machine-type e2-standard-4 \
--metadata 'PIP_PACKAGES=google-cloud-storage spark-nlp==2.5.1' \
--num-workers 2 \
--num-secondary-workers 20 \
--enable-component-gateway \
--max-idle 7200s \
--master-boot-disk-size=100GB \
--image-version 1.4-debian10 \
--initialization-actions gs://dataproc-initialization-actions/python/pip-install.sh \
--optional-components=JUPYTER,ANACONDA \
--enable-component-gateway \
--bucket=${BUCKET_NAME} \
--no-address

echo "Listing clusters that exist"

gcloud dataproc clusters list


echo "Submitting the job"

