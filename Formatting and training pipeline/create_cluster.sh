
CLUSTER_NAME=postal-parser-new1
REGION=europe-west1
BUCKET_NAME=postal-parser-28
NETWORK_NAME=for-cluster-2
ZONE=europe-west1-b
gcloud config set dataproc/region ${REGION}

echo "Creating a cluster"

gcloud beta dataproc clusters create ${CLUSTER_NAME} \
--master-machine-type e2-highmem-16 \
--worker-machine-type e2-highmem-16 \
--region ${REGION} \
--network ${NETWORK_NAME} \
--zone ${ZONE} \
--metadata 'PIP_PACKAGES=google-cloud-storage spark-nlp==2.5.1' \
--num-workers 12 \
--num-secondary-workers=100 \
--secondary-worker-type=preemptible \
--master-boot-disk-size=100GB \
--image-version 1.4-debian10 \
--initialization-actions gs://dataproc-initialization-actions/python/pip-install.sh \
--optional-components=JUPYTER,ANACONDA \
--enable-component-gateway \
--bucket=${BUCKET_NAME}

echo "Listing clusters that exist"

gcloud dataproc clusters list


echo "Submitting the job"

