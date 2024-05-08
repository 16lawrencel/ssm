from torchvision.datasets.utils import download_and_extract_archive

import os
from urllib.error import URLError
from pathlib import Path


class LRAUtils:
    mirrors = ["https://storage.googleapis.com/long-range-arena/"]

    resources = [("lra_release.gz", "e153bba13388d140f886dcef1a1b6c1d")]

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.listops_dir = Path(data_dir) / "lra_release"

    def _check_exists(self) -> bool:
        return self.listops_dir.exists()

    def download(self) -> None:
        """Download LRA (which includes listops) dataset if doesn't already exist.
        Code is adapted from torchvision's MNIST download code.

        NOTE: for some reason the file doesn't get properly extracted into a directory.
        TODO: fix this
        """
        if self._check_exists():
            return

        print(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
                    download_and_extract_archive(
                        url, download_root=self.data_dir, filename=filename, md5=md5
                    )
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")
