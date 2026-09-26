"""Small, explicit dataset download helpers."""
from pathlib import Path
from urllib.request import Request, urlopen
import shutil
import zipfile


def download(url, destination):
    destination = Path(destination)
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; FCCPortfolio/1.0)"})
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def extract_zip(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError("Archive contains an unsafe path")
        source.extractall(destination)
