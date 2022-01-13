import os
import requests
import tarfile

###### TODO: This does not work
#####def download_model(destination_folder: str):
#####    """
#####    Download a file from dropbox and put it in the destination folder.
#####
#####    Args:
#####        destination_folder: string, the local path where the files will be downloaded
#####    """
#####    # Create the destination folder
#####    os.makedirs(destination_folder, exist_ok=True)
#####
#####    # Download the tar.gz file from dropbox
#####    data_fname = os.path.join(destination_folder, "pretrained_model_simclr.pt")
#####    data_url = 'https://www.dropbox.com/s/pxwuali76e7rroe/pretrained_model_simclr.pt?dl=0'
#####    with open(data_fname, 'wb') as f:
#####        f.write(requests.get(data_url).content)
#####
#####    # Print the files in the destination_folder
#####    print("Files in the destination_folder")
#####    files = os.listdir(destination_folder)
#####    for f in files:
#####        print(os.path.join(destination_folder, f))


def download_data(destination_folder: str):
    """
    Download a compressed file from dropbox and expand it in the destination folder.

    Args:
        destination_folder: string, the local path where the files will be downloaded
    """
    # Create the destination folder
    os.makedirs(destination_folder, exist_ok=True)

    # Download the tar.gz file from dropbox
    data_fname = os.path.join(destination_folder, "slide_seq_data.tar.gz")
    data_url = 'https://www.dropbox.com/s/b2x5dn9k856opjh/slide_seq_data.tar.gz?dl=1'
    with open(data_fname, 'wb') as f:
        f.write(requests.get(data_url).content)

    # untar the tar.gz file
    tar = tarfile.open(data_fname)
    tar.extractall(path=destination_folder)
    tar.close()

    # Print the files in the destination_folder
    print("Files in the destination_folder")
    files = os.listdir(destination_folder)
    for f in files:
        print(os.path.join(destination_folder, f))

