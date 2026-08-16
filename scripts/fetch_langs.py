import pyarrow.parquet as pq
import urllib.request
import io
import pandas as pd

try:
    url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    print(f"Downloading slice from {url}...")
    req = urllib.request.Request(url, headers={'Range': 'bytes=0-1000000'}) # 1MB slice
    with urllib.request.urlopen(req) as response:
        content = response.read()
    
    # We can't read a partial parquet file easily, but we can look for strings
    import re
    langs = re.findall(b'target_lang\x00\x00\x00(.*?)\x00', content)
    # Actually, let's just use requests to get the exact language list from ai4bharat
    # The language codes are standard: hi, ta, te, ml, bn, mr, gu, kn, or, pa, as
    print("Downloaded bytes:", len(content))
except Exception as e:
    print(e)
