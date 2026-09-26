import zipfile
import pytest
from data_utils import extract_zip


def test_archive_cannot_escape_destination(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as f: f.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError): extract_zip(archive, tmp_path / "data")
    assert not (tmp_path / "escape.txt").exists()


def test_download_identifies_client_and_reuses_cache(tmp_path, monkeypatch):
    import io
    import data_utils
    requests = []
    def response(request, timeout):
        requests.append(request)
        assert request.get_header("User-agent")
        return io.BytesIO(b"example data")
    monkeypatch.setattr(data_utils, "urlopen", response)
    destination = tmp_path / "data" / "example.csv"
    assert data_utils.download("https://example.test/data.csv", destination).read_bytes() == b"example data"
    assert data_utils.download("https://example.test/data.csv", destination) == destination
    assert len(requests) == 1
    assert not destination.with_suffix(".csv.part").exists()
