import glob
from google.cloud import storage
import os
import logging
from google.api_core import page_iterator


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

    '''Get images from GCP Cloud Storage and save in the Local File
    '''
    client = storage.Client()

    file = os.environ.get("CLOUD_DATA")
    bucket = client.bucket(os.environ.get("BUCKET"))
    blobs = bucket.list_blobs()
    dir_list  = list_directories(os.environ.get('BUCKET'),os.environ.get('SAVE_DIR'))
    items_per_dir = len(list_blobs_with_prefix(bucket,
                                        list_directories(os.environ.get('BUCKET'),
                                            os.environ.get('SAVE_DIR'))[0][:],'/'))
    dir_counter = 0
    items_read = 0


    for blob in blobs:
        logging.info('Blobs: {}'.format(blob.name))
        destination_file = f"""{file}/{dir_list[dir_counter][61:-1]}"""
        try:
            os.mkdir(destination_file)
        except:
            pass
        blob.download_to_filename(f"{destination_file[:57]}{blob.name[61:]}")
        print(blob.name)
        logging.info('Exported {} to {}'.format(
        blob.name, destination_file))
        items_read += 1
        if items_read == items_per_dir:
            dir_counter += 1
            items_read = 0
            try:
                items_per_dir = len(list_blobs_with_prefix(bucket,
                                            list_directories(os.environ.get('BUCKET'),
                                            os.environ.get('SAVE_DIR'))[dir_counter][:],'/'))
            except:
                break


def _item_to_value(iterator, item):
    return item

def list_directories(bucket_name, prefix):
    """Function to list all directories within the bucket
    """

    if prefix and not prefix.endswith('/'):
        prefix += '/'

    extra_params = {
        "projection": "noAcl",
        "prefix": prefix,
        "delimiter": '/'
    }

    gcs = storage.Client()

    path = "/b/" + bucket_name + "/o"

    iterator = page_iterator.HTTPIterator(
        client=gcs,
        api_request=gcs._connection.api_request,
        path=path,
        items_key='prefixes',
        item_to_value=_item_to_value,
        extra_params=extra_params,
    )

    return [x for x in iterator]

def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    """Creates a List of all the blobs within a specific folder inside the bucket
    this information is needed to let the get_bucket_images function know when to create
    a new folder with the next class eg: after finishing all A images now creates the B folder
    and starts downloading them.
    """

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
    blobs_ =[]
    # Note: The call returns a response only when the iterator is consumed.
    for blob in blobs:
        blobs_.append(str(blob.name))

    return blobs_
