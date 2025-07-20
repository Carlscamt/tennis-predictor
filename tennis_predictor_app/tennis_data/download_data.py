import requests

def download_file(url, local_filename):
    """Downloads a file from a given URL and saves it locally."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

if __name__ == '__main__':
    BASE_URL = "http://www.tennis-data.co.uk"
    for year in range(2000, 2026):
        if year >= 2012:
            file_ext = ".xlsx"
        else:
            file_ext = ".xls"
        
        url = f"{BASE_URL}/{year}/{year}{file_ext}"
        local_filename = f"C:/Users/Carlos/Documents/ODST/tennis_data/{year}{file_ext}"
        
        try:
            print(f"Downloading {url} to {local_filename}")
            download_file(url, local_filename)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")