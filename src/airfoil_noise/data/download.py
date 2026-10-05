from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

DATASET_URL = "https://archive.ics.uci.edu/static/public/291/airfoil%2Bself%2Bnoise.zip"
ARCHIVE_SHA256 = "5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c"
RAW_SHA256 = "74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3"

ARCHIVE_NAME = "airfoil_self_noise.zip"
RAW_NAME = "airfoil_self_noise.dat"
DEFAULT_OUTPUT_DIR = Path("data/raw")


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256(usedforsecurity=False)

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_checksum(file_path: Path, expected_checksum: str) -> None:
    actual_checksum = calculate_sha256(file_path)

    if actual_checksum != expected_checksum:
        message = (
            f"Контрольная сумма файла {file_path} не совпадает. "
            f"Ожидалось: {expected_checksum}. "
            f"Получено: {actual_checksum}."
        )
        raise ValueError(message)


def download_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / ARCHIVE_NAME
    raw_path = output_dir / RAW_NAME

    if raw_path.exists():
        verify_checksum(raw_path, RAW_SHA256)
        print(f"Датасет уже загружен и проверен: {raw_path}")
        return raw_path

    print(f"Загрузка датасета из {DATASET_URL}")

    with (
        urlopen(DATASET_URL, timeout=60) as response,
        archive_path.open("wb") as archive_file,
    ):
        shutil.copyfileobj(response, archive_file)

    verify_checksum(archive_path, ARCHIVE_SHA256)

    with (
        ZipFile(archive_path) as archive,
        archive.open(RAW_NAME) as source_file,
        raw_path.open("wb") as target_file,
    ):
        shutil.copyfileobj(source_file, target_file)

    verify_checksum(raw_path, RAW_SHA256)
    archive_path.unlink(missing_ok=True)

    print(f"Датасет загружен и проверен: {raw_path}")
    return raw_path


def main() -> None:
    download_dataset()


if __name__ == "__main__":
    main()
