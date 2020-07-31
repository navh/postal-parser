from google.cloud import storage

f = open("/tmp/tempfile", 'w+')
storage_client = storage.Client()
bkt = storage_client.bucket('postal-parser-27')
blob = bkt.blob('conll-data/training.txt')
blob2 = bkt.blob('conll-data/training_formatted.txt')
downloaded_blob = blob.download_as_string()
downloaded_blob = downloaded_blob.decode("utf-8")
downloaded_blob = downloaded_blob.split("<Entity")
counter = 0
for line in downloaded_blob:
    if counter > 0:
        bparens_pos = line.find('{')
        eparens_pos = line.find('}')
        new_line = line[bparens_pos+1:eparens_pos]
        new_line = new_line.split(',')
        if len(new_line[0]) > len(new_line[1]):
            new_line = new_line[0]
        elif len(new_line[0]) < len(new_line[1]):
            new_line = new_line[1]
        else:
            new_line = new_line[0]
        new_line = new_line.split(':')
        new_line = new_line[1]
        new_line = new_line.split('\\n\\n')
        f.write(str(new_line[0]).lstrip(" '")+'\n\n')
        new_line = new_line[1]
        new_line = new_line.split('\\n')
        for item in new_line:
            f.write(str(item).rstrip("'")+'\n')
        f.write('\n')
    counter += 1
f.close()
blob2.upload_from_filename("/tmp/tempfile")

