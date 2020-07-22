CLUSTER_NAME=postal-parser
REGION=us-central1
BUCKET_NAME=postal-parser

gcloud config set dataproc/region ${REGION}

echo "Creating a cluster"

gcloud beta dataproc clusters create ${CLUSTER_NAME} \
        --region ${REGION} \
        --worker-machine-type n1-standard-2 \
        --metadata 'PIP_PACKAGES=google-cloud-storage spark-nlp==2.5.1' \
        --num-workers 2 \
	--master-boot-disk-size=40GB \
        --image-version 1.4-debian10 \
        --initialization-actions gs://dataproc-initialization-actions/python/pip-install.sh \
        --optional-components=JUPYTER,ANACONDA \
        --enable-component-gateway \
        --bucket=${BUCKET_NAME}

echo "Listing clusters that exist"

gcloud dataproc clusters list
