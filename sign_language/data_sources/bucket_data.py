import glob
from google.cloud import storage
import os

def upload_processed_images():

    ## to do delete

    files = glob.glob('processed_images/**/**/*.*', recursive=True)

    client = storage.Client()
    bucket = client.bucket(os.environ.get('BUCKET'))
    counter = 0
    for file in files:
        blob = bucket.blob(file)
        blob.upload_from_filename(file)
        counter +=1
        if counter % 1000 == 0:
            print(f'Uploaded {counter} images')

def get_bucket_images():

    pass
