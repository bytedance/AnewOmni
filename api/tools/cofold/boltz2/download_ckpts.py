import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor

TIMEOUT = 5
CHUNK_SIZE = 8192


def measure_latency(url):
    """speed test"""
    try:
        start = time.time()
        r = requests.head(url, timeout=TIMEOUT)
        if r.status_code < 400:
            return time.time() - start
    except Exception:
        pass
    return float("inf")


def select_best_url(urls):
    """select the best mirror"""
    print("Selecting best mirror...")
    latencies = []

    for url in urls:
        latency = measure_latency(url)
        latencies.append((latency, url))
        print(f"{url} -> {latency:.3f}s")

    latencies.sort()
    best = latencies[0][1]

    print(f"Selected: {best}\n")
    return best


def download_file(urls, save_path):
    """download single file"""
    if os.path.exists(save_path):
        print(f"File already exists: {save_path}")
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # rank by latency
    ranked_urls = sorted(urls, key=lambda u: measure_latency(u))

    for url in ranked_urls:
        try:
            print(f"Downloading from: {url}")
            with requests.get(url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()

                total = int(r.headers.get("content-length", 0))
                downloaded = 0

                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            if total:
                                percent = downloaded / total * 100
                                print(
                                    f"\r{save_path} {percent:.2f}%",
                                    end="",
                                    flush=True,
                                )

            print(f"\nSaved to {save_path}\n")
            return

        except Exception as e:
            print(f"Failed from {url}: {e}")

    raise RuntimeError(f"All mirrors failed for {save_path}")


def download_all(files, root_dir="weights", max_workers=1):
    """down load all"""
    tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file in files:
            save_path = os.path.join(root_dir, file["path"])
            urls = file["urls"]

            tasks.append(
                executor.submit(download_file, urls, save_path)
            )

        for t in tasks:
            t.result()


FILES = [
    {
        "path": "ccd.pkl",
        "urls": [
            "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl",
            "https://hf-mirror.com/boltz-community/boltz-1/resolve/main/ccd.pkl",
        ],
    },
    {
        "path": "mols.tar",
        "urls": [
            "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar",
            "https://hf-mirror.com/boltz-community/boltz-2/resolve/main/mols.tar",
        ],
    },
    {
        "path": "boltz2_conf.ckpt",
        "urls": [
            "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
            "https://hf-mirror.com/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
        ]
    },
    {
        "path": "boltz2_aff.ckpt",
        "urls": [
            "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt",
            "https://hf-mirror.com/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt",
        ]
    }
]



if __name__ == "__main__":
    import sys
    import tarfile
    if len(sys.argv) > 1: root_dir = sys.argv[1]
    else: root_dir = "./params"
    download_all(FILES, root_dir=root_dir)
    print(f"All files downloaded to {root_dir}")
    if not os.path.exists(os.path.join(root_dir, "mols")):
        print(f"Decompressing mols.tar...")
        with tarfile.open(os.path.join(root_dir, "mols.tar"), "r") as tar:
            tar.extractall(root_dir)  # noqa: S202
    print(f"Done")
