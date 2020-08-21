set -x -e

CLUSTER_NAME=postal-parser-mona3
REGION=europe-west1
BUCKET_NAME=postal-parser-28
NETWORK_NAME=for-cluster-2
ZONE=europe-west1-b
gcloud config set dataproc/region ${REGION}


cat << EOF > autoscale_tony.yaml
workerConfig:
    minInstances: 4
    maxInstances: 10
secondaryWorkerConfig:
    minInstances: 0
    maxInstances: 100
basicAlgorithm:
    cooldownPeriod: 2m
    yarnConfig:
        scaleUpFactor: 1.0
        scaleDownFactor: 1.0
        scaleUpMinWorkerFraction: 1.0
        scaleDownMinWorkerFraction: 1.0
        gracefulDecommissionTimeout: 15m
EOF
# Autoscaling policy creation
gcloud beta dataproc autoscaling-policies import autoscale_tony --source=autoscale_tony.yaml


echo "Creating a cluster"

gcloud beta dataproc clusters create ${CLUSTER_NAME} \
--master-machine-type n1-standard-8 \
--num-workers 120 --worker-machine-type n1-standard-4 \
--autoscaling-policy=autoscale_tony \
--region ${REGION} \
--network ${NETWORK_NAME} \
--zone ${ZONE} \
--metadata 'PIP_PACKAGES=google-cloud-storage spark-nlp==2.5.1' \
--master-boot-disk-size=100GB \
--image-version 1.4-debian10 \
--initialization-actions gs://dataproc-initialization-actions/python/pip-install.sh \
--optional-components=JUPYTER,ANACONDA \
--enable-component-gateway \
--bucket=${BUCKET_NAME}


echo "Listing clusters that exist"

gcloud dataproc clusters list


echo "Submitting the job"

