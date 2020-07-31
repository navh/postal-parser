from google.cloud import datastore
from google.cloud import storage

datastore_client = datastore.Client()
query = datastore_client.query(kind='CONLL')
query = query.add_filter('Training', '=', True)
result = query.fetch()


f = open("/tmp/tempfile", 'w+')
storage_client = storage.Client()
bkt = storage_client.bucket('postal-parser-27')
blob = bkt.blob('conll-data/training.txt')
for address in result:
	f.write(str(address)+"\n")
f.close()
blob.upload_from_filename("/tmp/tempfile")
