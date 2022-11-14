import os
import gdown
import zipfile

def check_dir(dir_name: str) -> bool:
    return os.path.isdir(dir_name)

def download_data(dir_name: str = "data_dir") -> None:
    if not check_dir(dir_name):
        os.mkdir(dir_name)

    os.chdir(dir_name)
    print("[INFO] Downloading data....")

    # Annotation
    gdown.download(
        "https://drive.google.com/uc?id=11Uj_qlbqNfB6tMlb9oA9dwmsgrauV1C-", quiet=False
    )
    gdown.download(
        "https://drive.google.com/uc?id=1VNqa2iJcVvqRG6s3QFIbvNTjApb2xdDX", quiet=False
    )
    print("[INFO] Extracting zip file....")
    with zipfile.ZipFile("pa100k.zip", 'r') as zip_ref:
        zip_ref.extractall("pa100k/images")
    with zipfile.ZipFile("annotation.zip", 'r') as zip_ref:
        zip_ref.extractall("")
    os.remove("pa100k.zip")
    os.remove("annotation.zip")
    os.chdir("..")

if __name__=="__main__":
    download_data()