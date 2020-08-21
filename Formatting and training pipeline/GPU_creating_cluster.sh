
CLUSTER_NAME=postal-parser-gpum
REGION=europe-west1
BUCKET_NAME=postal-parser-28
NETWORK_NAME=for-cluster-2
ZONE=europe-west1-b
gcloud config set dataproc/region ${REGION}

echo "Creating a cluster"

gcloud beta dataproc clusters create ${CLUSTER_NAME} \
--master-machine-type n1-standard-8 \
--region ${REGION} \
--master-accelerator type=nvidia-tesla-k80,count=1 \
--network ${NETWORK_NAME} \
--zone ${ZONE} \
--metadata 'PIP_PACKAGES=google-cloud-storage spark-nlp==2.5.1' \
--metadata 'gpu-driver-provider=NVIDIA' \
--master-boot-disk-size=100GB \
--image-version 1.4-debian10 \
--initialization-actions gs://dataproc-initialization-actions/python/pip-install.sh \
--initialization-actions gs://goog-dataproc-initialization-actions-europe-west1/gpu/install_gpu_driver.sh \
--optional-components=JUPYTER,ANACONDA \
--enable-component-gateway \
--bucket=${BUCKET_NAME}

echo "Listing clusters that exist"

gcloud dataproc clusters list
