from scripts.robust_downloader import download_with_retries

if __name__ == "__main__":
    download_with_retries(
        "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/malval.parquet?download=true",
        "data/raw/malval.parquet"
    )
