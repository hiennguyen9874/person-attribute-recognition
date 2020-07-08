# Copy from here: https://gist.github.com/grimpy/7dd579059d7c4c42d0528e4676edffaf
import requests
import sys
import os
# from tqdm import tqdm

def download_file_from_google_drive(id, destination=None):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    viewresp = session.get(URL, params={"id": id}, stream=True)

    token = None
    print(viewresp.cookies.get_dict())
    for key, value in viewresp.cookies.get_dict().items():
        if key.startswith("download_warning"):
            token = value
            break
    else:
        print("Could not find token for url")
        # sys.exit(1)
        raise SystemExit

    params = {"id": id, "confirm": token}
    response = session.get(URL, params=params, stream=True, headers={'Range': 'bytes=0-'})

    CHUNK_SIZE = 32 * 1024
    total_size = int(response.headers.get("content-length", 0))
    if total_size == 0:
        # attempt to get it out of Content-Range
        range = response.headers.get('Content-Range', None)
        if range:
            total_size = int(range.split('/')[-1])
    filename = None
    if destination is None or os.path.isdir(destination):
        for segment in response.headers.get('Content-Disposition').split(';'):
            if segment.startswith('filename='):
                filename = segment.split('=', 1)[-1].strip('"')
                break
        else:
            print("Could not find destination file")
            sys.exit(1)
        if destination is not None:
            destination = os.path.join(destination, filename)
        else:
            destination = filename
    
    if os.path.exists(destination):
        if os.path.getsize(destination) >= total_size:
            return filename
            
    partfile = destination + ".part"
    stat = None
    initial = 0
    if os.path.exists(partfile):
        stat = os.stat(partfile)
        response.close()
        response = session.get(URL, params=params, stream=True, headers={'Range': 'bytes={}-'.format(stat.st_size)})
        initial = stat.st_size 
        range = response.headers.get('Content-Range', None)

    # with tqdm(desc=destination, total=total_size, initial=initial, unit="B", unit_scale=True) as pbar:
    with open(partfile, "ab") as f:
        f.seek(initial)
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                # pbar.update(CHUNK_SIZE)
                f.write(chunk)
        os.rename(partfile, destination)
    return filename

def download_with_url(api, file_id, destination, name_file):
    session = requests.Session()
    url = "https://www.googleapis.com/drive/v3/files/" + file_id + "?alt=media&key=" + api
    response = session.get(url, stream=True, headers={'Range': 'bytes=0-'})

    CHUNK_SIZE = 32 * 1024
    total_size = int(response.headers.get("content-length", 0))
    if total_size == 0:
        # attempt to get it out of Content-Range
        range = response.headers.get('Content-Range', None)
        if range:
            total_size = int(range.split('/')[-1])
    
    filename = name_file
    destination = os.path.join(destination, filename)

    if os.path.exists(destination):
        if os.path.getsize(destination) >= total_size:
            return filename
            
    partfile = destination + ".part"
    stat = None
    initial = 0
    if os.path.exists(partfile):
        stat = os.stat(partfile)
        response.close()
        response = session.get(url, stream=True, headers={'Range': 'bytes={}-'.format(stat.st_size)})
        initial = stat.st_size 
        range = response.headers.get('Content-Range', None)

    # with tqdm(desc=destination, total=total_size, initial=initial, unit="B", unit_scale=True) as pbar:
    with open(partfile, "ab") as f:
        f.seek(initial)
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                # pbar.update(CHUNK_SIZE)
                f.write(chunk)
        os.rename(partfile, destination)
    return filename