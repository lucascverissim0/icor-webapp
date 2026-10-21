from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path

import pytest

from icor.evidence.dgt_fleet_acquisition import (
    ARCHIVE_URL,
    SOURCE_HEADER,
    acquire_fleet,
    download_archive,
)
from icor.evidence.sources.dgt import SCHEMA

MEMBER = "parque_test.txt"


def _row(**values: str) -> str:
    fields = dict.fromkeys(SOURCE_HEADER, "")
    fields.update(
        {
            "MUNICIPIO": "28079",
            "FECHA_MATR": "01/02/2015",
            "TIPO_TITULAR": "PERSONA FISICA",
            "DISTANCIA_EJES": "2600",
        }
    )
    fields.update(values)
    return "|".join(fields[name] for name in SOURCE_HEADER) + "\r\n"


def _archive(tmp_path: Path, rows: list[str], *, header: list[str] = SOURCE_HEADER) -> Path:
    path = tmp_path / "parque.zip"
    text = "|".join(header) + "\r\n" + "".join(rows)
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(MEMBER, text.encode("latin-1"))
    return path


def _acquire(tmp_path: Path, archive: Path):
    return acquire_fleet(
        tmp_path / "out" / "dgt.csv",
        archive,
        expected_bytes=archive.stat().st_size,
        expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        member=MEMBER,
    )


def _rows(path: Path) -> list[tuple[str, ...]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [tuple(row) for row in csv.reader(stream)]


def test_keeps_passenger_cars_by_make_model_and_first_registration_year(
    tmp_path: Path,
) -> None:
    archive = _archive(
        tmp_path,
        [
            _row(
                MARCA="MARCA B",
                MODELO="MODEL  X",
                CAT_EURO="M1",
                TIPO_DGT="TURISMOS",
                FEC_PRIM_MATR="03/04/2004",
                FECHA_MATR="05/06/2012",
                NUEVO_USADO="U",
            ),
            _row(
                MARCA="MARCA B",
                MODELO="MODEL X",
                CAT_EURO="M1",
                TIPO_DGT="TURISMOS",
                FEC_PRIM_MATR="09/09/2004",
            ),
            _row(
                MARCA="MARCA A",
                MODELO="VAN PEOPLE",
                CAT_EURO="M1G",
                TIPO_DGT="FURGONETAS",
                FEC_PRIM_MATR="01/01/2008",
            ),
            _row(
                MARCA="MARCA Ñ",
                MODELO="OLD ONE",
                CAT_EURO="",
                TIPO_DGT="TURISMOS",
                FEC_PRIM_MATR="01/01/1988",
            ),
            _row(
                MARCA="MARCA A",
                MODELO="¡",
                CAT_EURO="M1",
                TIPO_DGT="TURISMOS",
                FEC_PRIM_MATR="01/01/2001",
            ),
            _row(
                MARCA="MARCA A",
                MODELO="TRUCK",
                CAT_EURO="N1",
                TIPO_DGT="FURGONETAS",
                FEC_PRIM_MATR="01/01/2001",
            ),
            _row(
                MARCA="MARCA A",
                MODELO="BIKE",
                CAT_EURO="",
                TIPO_DGT="MOTOCICLETAS",
                FEC_PRIM_MATR="01/01/2001",
            ),
            _row(
                MARCA="MARCA A",
                MODELO="NEW",
                CAT_EURO="M1",
                TIPO_DGT="TURISMOS",
                FEC_PRIM_MATR="01/01/2020",
            ),
        ],
    )

    result = _acquire(tmp_path, archive)

    assert _rows(result.path) == [
        SCHEMA,
        ("MARCA A", "NEW", "2020", "1"),
        ("MARCA A", "VAN PEOPLE", "2008", "1"),
        ("MARCA A", "¡", "2001", "1"),
        ("MARCA B", "MODEL X", "2004", "2"),
        ("MARCA Ñ", "OLD ONE", "1988", "1"),
    ]
    assert result.source_rows == 8
    assert result.passenger_cars == 6
    assert result.group_count == 5
    assert result.accepted_group_count == 3
    assert result.rejected_group_count == 2
    assert result.accepted_vehicle_count == 4
    assert result.masked_vehicle_count == 1
    assert result.artifact_bytes == result.path.stat().st_size
    assert result.sha256 == hashlib.sha256(result.path.read_bytes()).hexdigest()


def test_no_row_level_field_leaves_the_archive(tmp_path: Path) -> None:
    archive = _archive(
        tmp_path,
        [
            _row(
                MARCA="MARCA A",
                MODELO="MODEL X",
                CAT_EURO="M1",
                TIPO_DGT="TURISMOS",
                FEC_PRIM_MATR="01/01/2001",
                MUNICIPIO="28079",
                PROVINCIA="28",
            )
        ],
    )

    result = _acquire(tmp_path, archive)

    text = result.path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == ",".join(SCHEMA)
    assert "28079" not in text
    assert "PERSONA" not in text
    assert "2600" not in text


def test_output_is_deterministic_regardless_of_row_order(tmp_path: Path) -> None:
    rows = [
        _row(MARCA="B", MODELO="X", CAT_EURO="M1", FEC_PRIM_MATR="01/01/2001"),
        _row(MARCA="A", MODELO="Y", CAT_EURO="M1", FEC_PRIM_MATR="01/01/2002"),
    ]
    first = _acquire(tmp_path / "a", _archive(_dir(tmp_path / "a"), rows))
    second = _acquire(tmp_path / "b", _archive(_dir(tmp_path / "b"), rows[::-1]))

    assert first.path.read_bytes() == second.path.read_bytes()


def test_refuses_an_archive_that_is_not_the_pinned_release(tmp_path: Path) -> None:
    archive = _archive(tmp_path, [_row(MARCA="A", MODELO="B", CAT_EURO="M1")])

    with pytest.raises(ValueError, match="pinned release"):
        acquire_fleet(
            tmp_path / "dgt.csv",
            archive,
            expected_bytes=archive.stat().st_size,
            expected_sha256="0" * 64,
            member=MEMBER,
        )
    assert not (tmp_path / "dgt.csv").exists()


def test_refuses_an_unexpected_schema_or_malformed_row(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema"):
        _acquire(tmp_path / "s", _archive(_dir(tmp_path / "s"), [], header=SOURCE_HEADER[:-1]))
    malformed = _row(MARCA="A", MODELO="B", CAT_EURO="M1").replace("|", "", 1)
    with pytest.raises(ValueError, match="malformed"):
        _acquire(tmp_path / "m", _archive(_dir(tmp_path / "m"), [malformed]))


def _dir(path: Path) -> Path:
    path.mkdir()
    return path


class _Response:
    def __init__(self, payload: bytes, url: str = ARCHIVE_URL) -> None:
        self._payload = payload
        self._url = url

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        chunk, self._payload = self._payload[:size], self._payload[size:]
        return chunk

    def geturl(self) -> str:
        return self._url


def test_download_stays_on_the_allowlisted_origin_and_byte_ceiling(tmp_path: Path) -> None:
    def opener(payload: bytes, url: str = ARCHIVE_URL):
        return lambda request, timeout: _Response(payload, url)

    path = download_archive(tmp_path / "a.zip", opener=opener(b"abc"), expected_bytes=3)
    assert path.read_bytes() == b"abc"

    with pytest.raises(ValueError, match="byte limit"):
        download_archive(tmp_path / "b.zip", opener=opener(b"abcd"), expected_bytes=3)
    with pytest.raises(ValueError, match="allowlisted"):
        download_archive(
            tmp_path / "c.zip",
            opener=opener(b"abc", "https://example.org/x.zip"),
            expected_bytes=3,
        )
    with pytest.raises(ValueError, match="allowlisted"):
        download_archive(tmp_path / "d.zip", url="http://www.dgt.es/x.zip", expected_bytes=3)
    assert sorted(item.name for item in tmp_path.iterdir()) == ["a.zip"]
