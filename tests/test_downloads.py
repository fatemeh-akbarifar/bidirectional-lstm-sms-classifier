import zipfile
import pytest
from data_utils import extract_zip


def test_archive_cannot_escape_destination(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as f: f.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError): extract_zip(archive, tmp_path / "data")
    assert not (tmp_path / "escape.txt").exists()
