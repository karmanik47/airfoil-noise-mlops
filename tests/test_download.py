from pathlib import Path

import pytest

from airfoil_noise.data.download import (
    calculate_sha256,
    verify_checksum,
)


def test_calculate_sha256(tmp_path: Path) -> None:
    test_file = tmp_path / "example.txt"
    test_file.write_bytes(b"airfoil")

    checksum = calculate_sha256(test_file)

    assert checksum == (
        "621306704b3c1b426474fab455c51bcb97a3cf86a3de48b223133cd94013f77a"
    )


def test_verify_checksum_rejects_modified_file(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "example.txt"
    test_file.write_bytes(b"modified")

    with pytest.raises(
        ValueError,
        match="Контрольная сумма файла",
    ):
        verify_checksum(test_file, "incorrect-checksum")
