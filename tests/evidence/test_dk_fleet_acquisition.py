from __future__ import annotations

import ftplib
import hashlib
import io
import json
import struct
import zipfile
import zlib
from pathlib import Path

import pytest

from icor.evidence import dk_fleet_acquisition as dk
from icor.evidence.dk_fleet_acquisition import (
    ENTRY_NAME,
    FTP_HOST,
    NAMESPACE,
    DeflatedEntry,
    FtpFile,
    acquire_fleet,
    aggregate_zip,
    connect_ftp,
    newest_snapshot,
    prefetched,
)

NEWEST = "ESStatistikListeModtag-20261020-122157.zip"
OLDER = "ESStatistikListeModtag-20261011-215909.zip"
# Fictional values that must never leave the source.
VIN = "FICTIONALVIN000001"
PLATE = "ZZ99999"
IDENT = "9000000000000001"
APPROVAL = "FICTIONAL-APPROVAL-77"
COMMENT = "FICTIONAL COMMENT ABOUT THE OWNER NORTHSTAR-PRIVATE"


def vehicle(
    ident: str = IDENT,
    *,
    art: str = "1",
    art_name: str = "Personbil",
    origin: str = "Typeattest",
    status: str = "Registreret",
    registration_status: str = "Registreret",
    first: str | None = "2012-05-02+02:00",
    make: str | None = "AURORA",
    model: str | None = "NOVA",
    model_year: str | None = None,
    inspection: tuple[str, str] | None = ("PeriodiskSyn", "2020-01-15+01:00"),
    status_date: str = "2019-03-01T10:00:00.000+01:00",
    registration_status_date: str = "2019-03-01T10:00:00.000+01:00",
    leasing: str | None = None,
) -> str:
    """One fictional ``<Statistik>`` record laid out like the DMR extract."""

    parts = [
        f"<ns:KoeretoejIdent>{ident}</ns:KoeretoejIdent>",
        f"<ns:KoeretoejArtNummer>{art}</ns:KoeretoejArtNummer>",
        f"<ns:KoeretoejArtNavn>{art_name}</ns:KoeretoejArtNavn>",
        "<ns:KoeretoejAnvendelseStruktur><ns:KoeretoejAnvendelseNummer>2"
        "</ns:KoeretoejAnvendelseNummer></ns:KoeretoejAnvendelseStruktur>",
        f"<ns:RegistreringNummerNummer>{PLATE}</ns:RegistreringNummerNummer>",
    ]
    grund = [
        f"<ns:KoeretoejOplysningOprettetUdFra>{origin}</ns:KoeretoejOplysningOprettetUdFra>",
        f"<ns:KoeretoejOplysningStatus>{status}</ns:KoeretoejOplysningStatus>",
        f"<ns:KoeretoejOplysningStatusDato>{status_date}</ns:KoeretoejOplysningStatusDato>",
    ]
    if first is not None:
        grund.append(
            "<ns:KoeretoejOplysningFoersteRegistreringDato>"
            f"{first}</ns:KoeretoejOplysningFoersteRegistreringDato>"
        )
    grund += [
        f"<ns:KoeretoejOplysningStelNummer>{VIN}</ns:KoeretoejOplysningStelNummer>",
        "<ns:KoeretoejOplysningTypeAnmeldelseNummer>"
        f"{APPROVAL}</ns:KoeretoejOplysningTypeAnmeldelseNummer>",
        f"<ns:KoeretoejOplysningKommentar>{COMMENT}</ns:KoeretoejOplysningKommentar>",
    ]
    if model_year is not None:
        grund.append(f"<ns:KoeretoejOplysningModelAar>{model_year}</ns:KoeretoejOplysningModelAar>")
    betegnelse = []
    if make is not None:
        betegnelse.append(f"<ns:KoeretoejMaerkeTypeNavn>{make}</ns:KoeretoejMaerkeTypeNavn>")
    if model is not None:
        betegnelse.append(
            f"<ns:Model><ns:KoeretoejModelTypeNavn>{model}</ns:KoeretoejModelTypeNavn></ns:Model>"
        )
    grund.append(
        "<ns:KoeretoejBetegnelseStruktur>"
        + "".join(betegnelse)
        + "</ns:KoeretoejBetegnelseStruktur>"
    )
    parts.append(
        "<ns:KoeretoejOplysningGrundStruktur>"
        + "".join(grund)
        + "</ns:KoeretoejOplysningGrundStruktur>"
    )
    if leasing is not None:
        parts.append(f"<ns:LeasingGyldigFra>{leasing}</ns:LeasingGyldigFra>")
    if inspection is not None:
        parts.append(
            f"<ns:SynResultatStruktur><ns:SynResultatSynsType>{inspection[0]}"
            f"</ns:SynResultatSynsType><ns:SynResultatSynsDato>{inspection[1]}"
            "</ns:SynResultatSynsDato></ns:SynResultatStruktur>"
        )
    parts.append(
        f"<ns:KoeretoejRegistreringStatus>{registration_status}</ns:KoeretoejRegistreringStatus>"
    )
    parts.append(
        "<ns:KoeretoejRegistreringStatusDato>"
        f"{registration_status_date}</ns:KoeretoejRegistreringStatusDato>"
    )
    return "<ns:Statistik>" + "".join(parts) + "</ns:Statistik>"


def document(
    *records: str, namespace: str = NAMESPACE, root: str = "ESStatistikListeModtag_I"
) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<ns:{root} xmlns:ns="{namespace}"><ns:StatistikSamling>'
        + "".join(records)
        + f"</ns:StatistikSamling></ns:{root}>"
    ).encode()


def server_zip(xml: bytes, *, name: str = ENTRY_NAME, crc_delta: int = 0) -> bytes:
    """A ZIP laid out like the real extract: streamed deflate, ZIP64 descriptor and directory."""

    compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
    data = compressor.compress(xml) + compressor.flush()
    crc = (zlib.crc32(xml) + crc_delta) & 0xFFFFFFFF
    encoded = name.encode()
    local = struct.pack(
        "<4sHHHHHIIIHH", b"PK\x03\x04", 20, 0x0808, 8, 0, 0, 0, 0, 0, len(encoded), 0
    )
    body = local + encoded + data
    descriptor = struct.pack("<4sIQQ", b"PK\x07\x08", crc, len(data), len(xml))
    extra = struct.pack("<HHQQ", 1, 16, len(xml), len(data))
    central = (
        struct.pack(
            "<4sHHHHHHIIIHHHHHII",
            b"PK\x01\x02",
            45,
            45,
            0x0808,
            8,
            0,
            0,
            crc,
            0xFFFFFFFF,
            0xFFFFFFFF,
            len(encoded),
            len(extra),
            0,
            0,
            0,
            0,
            0,
        )
        + encoded
        + extra
    )
    offset = len(body) + len(descriptor)
    zip64_end = struct.pack(
        "<4sQHHIIQQQQ", b"PK\x06\x06", 44, 45, 45, 0, 0, 1, 1, len(central), offset
    )
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, offset + len(central), 1)
    end = struct.pack("<4sHHHHIIH", b"PK\x05\x06", 0, 0, 1, 1, len(central), 0xFFFFFFFF, 0)
    return body + descriptor + central + zip64_end + locator + end


def python_zip(xml: bytes, *, seekable: bool, entries: tuple[str, ...] = (ENTRY_NAME,)) -> bytes:
    class Unseekable(io.RawIOBase):
        def __init__(self) -> None:
            self.buffer = io.BytesIO()

        def writable(self) -> bool:
            return True

        def write(self, data: bytes) -> int:  # type: ignore[override]
            return self.buffer.write(data)

        def tell(self) -> int:
            return self.buffer.tell()

    target: io.IOBase = io.BytesIO() if seekable else Unseekable()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:  # type: ignore[arg-type]
        for entry in entries:
            bundle.writestr(entry, xml)
    return target.getvalue() if seekable else target.buffer.getvalue()  # type: ignore[union-attr]


class Server:
    def __init__(self, files: dict[str, bytes], *, drops: int = 0, drop_after: int = 1000) -> None:
        self.files = files
        self.drops = drops
        self.drop_after = drop_after
        self.sessions = 0
        self.rests: list[int | None] = []
        self.sizes: dict[str, int] = {}


class FakeData:
    def __init__(self, server: Server, payload: bytes, chunk: int = 64) -> None:
        self.server = server
        self.payload = payload
        self.chunk = chunk
        self.sent = 0

    def recv(self, size: int) -> bytes:
        if self.server.drops and self.sent >= self.server.drop_after:
            self.server.drops -= 1
            raise ConnectionResetError("connection reset by peer")
        count = min(size, self.chunk, len(self.payload) - self.sent)
        piece = self.payload[self.sent : self.sent + count]
        self.sent += count
        return piece

    def close(self) -> None:
        return None


class FakeFtp:
    def __init__(self, server: Server, host: str = FTP_HOST) -> None:
        self.server = server
        self.host = host
        self.type = "I"
        server.sessions += 1

    def nlst(self) -> list[str]:
        self.type = "A"  # a listing switches the session to ASCII, as ftplib's does
        return sorted(["README.txt", *self.server.files])

    def voidcmd(self, command: str) -> str:
        if command.startswith("TYPE "):
            self.type = command.split()[1]
        return "200 ok"

    def size(self, name: str) -> int:
        return self.server.sizes.get(name, len(self.server.files[name]))

    def sendcmd(self, command: str) -> str:
        assert command == f"MDTM {NEWEST}"
        return "213 20261020102216"

    def transfercmd(self, command: str, rest: int | None = None) -> FakeData:
        if rest is not None and self.type != "I":
            raise ftplib.error_perm("550 No support for resume of ASCII transfer.")
        self.server.rests.append(rest)
        name = command.removeprefix("RETR ")
        return FakeData(self.server, self.server.files[name][rest or 0 :])

    def voidresp(self) -> str:
        return "226 done"

    def quit(self) -> str:
        return "221 bye"

    def close(self) -> None:
        return None


def serve(xml: bytes, **options: int) -> tuple[Server, bytes]:
    archive = server_zip(xml)
    server = Server({OLDER: b"older bytes", NEWEST: archive}, **options)
    return server, archive


def run(
    tmp_path: Path, xml: bytes, **options: int
) -> tuple[dk.AcquisitionResult, dict[str, object], Server, bytes]:
    server, archive = serve(xml, **options)
    result = acquire_fleet(
        tmp_path / "dk-registered-cars-{date}.csv",
        connect=lambda: FakeFtp(server),
        sleep=lambda _seconds: None,
    )
    profile = json.loads(result.profile_path.read_text(encoding="utf-8"))
    return result, profile, server, archive


SAMPLE = document(
    vehicle("9000000000000001", first="2012-05-02+02:00"),
    vehicle("9000000000000002", first="2012-11-30+01:00", make="AURORA", model="NOVA"),
    vehicle("9000000000000003", first="2019-06-01+02:00", make="NORTHSTAR", model="COMET"),
    vehicle("9000000000000004", status="Afmeldt", registration_status="Afmeldt"),
    vehicle("9000000000000005", status="Skrottet", registration_status="Afmeldt"),
    vehicle("9000000000000006", status="Eksporteret", registration_status="Afmeldt"),
    vehicle("9000000000000007", art="3", art_name="Varebil"),
    vehicle("9000000000000008", first="2021-01-05+01:00", make="NORTHSTAR", model=None),
    vehicle("9000000000000009", first=None, make="AURORA", model="NOVA"),
)


def test_filters_funnel_and_cohort_come_from_the_first_registration(tmp_path: Path) -> None:
    result, profile, _, _ = run(tmp_path, SAMPLE)

    assert result.path.name == "dk-registered-cars-20261020.csv"
    assert result.path.read_text(encoding="utf-8") == (
        "make,model,cohort,vehicles\n"
        "AURORA,NOVA,2012,2\n"
        "AURORA,NOVA,,1\n"
        "NORTHSTAR,,2021,1\n"
        "NORTHSTAR,COMET,2019,1\n"
    )
    assert (result.source_rows, result.category_rows, result.registered_rows) == (9, 8, 5)
    assert (result.group_count, result.vehicles) == (4, 5)
    assert (result.blank_model_vehicles, result.blank_cohort_vehicles) == (1, 1)
    assert result.masked_vehicles == 0
    assert profile["filters"]["funnel"] == {  # type: ignore[index]
        "source_rows": 9,
        "after_category_filter": 8,
        "repeated_vehicle_rows_dropped": 0,
        "after_category_filter_unique_vehicles": 8,
        "after_status_filter": 5,
    }
    assert profile["passenger_cars_by_status"] == {
        "Afmeldt": 1,
        "Eksporteret": 1,
        "Registreret": 5,
        "Skrottet": 1,
    }
    assert profile["categories_in_source"] == {"1 Personbil": 8, "3 Varebil": 1}
    assert profile["registered_by_cohort"] == {"(missing)": 1, "2012": 2, "2019": 1, "2021": 1}
    assert profile["passenger_cars_by_cohort_and_status_from_2000"]["2012"] == {  # type: ignore[index]
        "Afmeldt": 1,
        "Eksporteret": 1,
        "Registreret": 2,
        "Skrottet": 1,
    }
    assert [item["make"] for item in profile["top_makes_registered"]] == [  # type: ignore[union-attr]
        "AURORA",
        "NORTHSTAR",
    ]


def test_identifying_values_never_reach_the_outputs(tmp_path: Path) -> None:
    messages: list[str] = []
    server, _ = serve(SAMPLE)
    result = acquire_fleet(
        tmp_path / "dk-{date}.csv",
        connect=lambda: FakeFtp(server),
        log=messages.append,
    )

    outputs = [result.path.read_bytes(), result.profile_path.read_bytes()]
    outputs.append("\n".join(messages).encode())
    for secret in (VIN, PLATE, IDENT, "9000000000000001", APPROVAL, COMMENT, "NORTHSTAR-PRIVATE"):
        assert all(secret.encode() not in blob for blob in outputs), secret
    profile = json.loads(result.profile_path.read_text(encoding="utf-8"))
    assert profile["owner_like_element_names"] == []
    assert "KoeretoejOplysningStelNummer" in profile["element_names_seen"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "dk-20261020.csv",
        "dk-20261020.csv.profile.json",
    ]


def test_the_newest_snapshot_is_streamed_hashed_and_dated(tmp_path: Path) -> None:
    messages: list[str] = []
    server, archive = serve(SAMPLE)
    result = acquire_fleet(
        tmp_path / "x-{date}.csv", connect=lambda: FakeFtp(server), log=messages.append
    )

    assert result.source_name == NEWEST
    assert result.source_date == "2026-10-20"
    assert result.source_bytes == len(archive)
    assert result.source_sha256 == hashlib.sha256(archive).hexdigest()
    assert result.sha256 == hashlib.sha256(result.path.read_bytes()).hexdigest()
    profile = json.loads(result.profile_path.read_text(encoding="utf-8"))
    source = profile["source"]
    assert source["url"] == f"ftp://{FTP_HOST}/ESStatistikListeModtag/{NEWEST}"
    assert source["last_modified_utc"] == "2026-10-20T10:22:16+00:00"
    assert source["sha256"] == result.source_sha256 and source["bytes"] == len(archive)
    assert source["entry_inflated_bytes"] == len(SAMPLE)
    assert "no explicit reuse licence" in source["terms"]
    assert any(message.startswith(f"snapshot {NEWEST}") for message in messages)
    assert any("stream complete" in message for message in messages)
    assert server.rests == [None]
    # The pinned name is honoured, and an unknown one is refused.
    with pytest.raises(ValueError, match="not in the directory"):
        newest_snapshot(FakeFtp(server), name="ESStatistikListeModtag-20200101-000000.zip")


def test_a_vehicle_repeated_per_leasing_period_is_counted_once(tmp_path: Path) -> None:
    xml = document(
        vehicle(IDENT, leasing="2020-01-01"),
        vehicle(IDENT, leasing="2022-01-01"),
        vehicle(IDENT, leasing="2024-01-01", status="Afmeldt"),
        vehicle("9000000000000002"),
        vehicle("9000000000000003", art="3", art_name="Varebil"),
        vehicle("9000000000000003", art="3", art_name="Varebil"),
    )

    result, profile, _, _ = run(tmp_path, xml)

    assert (result.source_rows, result.category_rows) == (6, 4)
    assert (result.duplicate_category_rows, result.registered_rows) == (2, 2)
    repeated = profile["filters"]["repeated_vehicles"]
    assert repeated["rows_dropped"] == 2  # type: ignore[index]
    assert repeated["dropped_rows_that_differ_in_status_origin_labels_or_date"] == 1  # type: ignore[index]
    assert repeated["passenger_car_rows_without_ident"] == 0  # type: ignore[index]
    assert result.path.read_text(encoding="utf-8").splitlines()[1:] == ["AURORA,NOVA,2012,2"]


def test_used_imports_are_compared_with_their_danish_entry_dates(tmp_path: Path) -> None:
    xml = document(
        # Imported in 2019 from a car first registered abroad in 2012: the customs inspection
        # is years after the first registration, and the model year equals it.
        vehicle(
            "9000000000000001",
            origin="Registreringssyn",
            first="2012-03-01+01:00",
            model_year="2012",
            inspection=("RegistreringssynToldsyn", "2019-06-20+02:00"),
            registration_status_date="2019-07-01T09:00:00.000+02:00",
        ),
        # A new car registered in Denmark: inspection the same month.
        vehicle(
            "9000000000000002",
            origin="CocDokument",
            first="2024-05-02+02:00",
            model_year="2023",
            inspection=("RegistreringsSyn", "2024-05-10+02:00"),
            registration_status_date="2024-05-02T09:00:00.000+02:00",
        ),
        vehicle(
            "9000000000000003",
            origin="Registreringssyn",
            first="2016-05-02+02:00",
            model_year="2013",
            inspection=None,
            status="Afmeldt",
            registration_status="Afmeldt",
        ),
    )

    _, profile, _, _ = run(tmp_path, xml)

    evidence = profile["used_import_evidence"]["scopes"]  # type: ignore[index]
    imported = evidence["registered"]["Registreringssyn"]
    assert imported["vehicles"] == 1
    customs = imported["first_registration_vs:inspection:RegistreringssynToldsyn"]
    assert customs["usable"] == 1
    assert customs["first_registration_more_than_180_days_earlier"] == 1
    assert customs["buckets"] == {">1825": 1}
    status_date = imported["first_registration_vs:registration_status_date"]
    assert status_date["buckets"] == {">1825": 1}
    assert imported["model_year_summary"]["cohort_same_year"] == 1
    new_car = evidence["registered"]["CocDokument"]
    assert new_car["first_registration_vs:inspection:RegistreringsSyn"]["buckets"] == {"0..30": 1}
    assert new_car["model_year_summary"]["cohort_1_year_after"] == 1
    # The unregistered import only appears in the all-statuses scope.
    assert evidence["registered"]["Registreringssyn"]["vehicles"] == 1
    everything = evidence["all_statuses"]["Registreringssyn"]
    assert everything["vehicles"] == 2
    assert everything["model_year_summary"]["cohort_2_or_more_years_after"] == 1
    assert everything["first_registration_vs:inspection:none"]["buckets"] == {"missing": 1}
    assert profile["registered_by_origin_and_cohort"] == {
        "CocDokument": {"2024": 1},
        "Registreringssyn": {"2012": 1},
    }


def test_the_gap_buckets_split_at_one_hundred_and_eighty_days() -> None:
    assert [dk._gap_bucket(days) for days in (-200, -181, -180, -1, 0, 30, 31, 180)] == [
        "<=-181",
        "<=-181",
        "-180..-1",
        "-180..-1",
        "0..30",
        "0..30",
        "31..180",
        "31..180",
    ]
    assert [dk._gap_bucket(days) for days in (181, 365, 366, 730, 731, 1825, 1826)] == [
        "181..365",
        "181..365",
        "366..730",
        "366..730",
        "731..1825",
        "731..1825",
        ">1825",
    ]
    assert [dk._model_year_bucket(gap) for gap in (-9, -5, -4, 0, 14, 15, 40)] == [
        "<=-5",
        "<=-5",
        "-4",
        "0",
        "14",
        ">=15",
        ">=15",
    ]


def test_hosts_outside_the_allowlist_are_refused() -> None:
    with pytest.raises(ValueError, match="allowlisted"):
        connect_ftp("example.org")
    for url in (
        "http://5.44.137.84/ESStatistikListeModtag/",
        "https://5.44.137.84/ESStatistikListeModtag/",
        "ftp://5.44.137.85/ESStatistikListeModtag/",
        "ftp://user@5.44.137.84/ESStatistikListeModtag/",
        "ftp://5.44.137.84.evil.example/ESStatistikListeModtag/",
    ):
        with pytest.raises(ValueError, match="allowlisted"):
            dk._require_allowlisted(url)
    dk._require_allowlisted(dk.DIRECTORY_URL)
    server, _ = serve(SAMPLE)
    with pytest.raises(ValueError, match="allowlisted"):
        newest_snapshot(FakeFtp(server, host="203.0.113.9"))
    remote = FtpFile(lambda: FakeFtp(server, host="203.0.113.9"), NEWEST, 10)
    with pytest.raises(ValueError, match="allowlisted"):
        remote.read(5)


def test_a_dropped_connection_resumes_with_rest_and_keeps_the_hash(tmp_path: Path) -> None:
    messages: list[str] = []
    server, archive = serve(SAMPLE, drops=2, drop_after=150)
    result = acquire_fleet(
        tmp_path / "d-{date}.csv",
        connect=lambda: FakeFtp(server),
        log=messages.append,
        sleep=lambda _seconds: None,
    )

    assert result.source_sha256 == hashlib.sha256(archive).hexdigest()
    assert result.vehicles == 5
    assert server.rests[0] is None and all(rest for rest in server.rests[1:])
    assert len(server.rests) == 3
    assert sum("interrupted" in message for message in messages) == 2


def test_a_download_longer_than_listed_is_refused(tmp_path: Path) -> None:
    server, archive = serve(SAMPLE)
    server.sizes[NEWEST] = len(archive) - 10

    with pytest.raises(ValueError, match="byte limit"):
        acquire_fleet(tmp_path / "d.csv", connect=lambda: FakeFtp(server))
    assert list(tmp_path.iterdir()) == []


def test_a_download_shorter_than_listed_gives_up(tmp_path: Path) -> None:
    server, archive = serve(SAMPLE)
    server.sizes[NEWEST] = len(archive) + 500

    with pytest.raises((RuntimeError, ValueError)):
        acquire_fleet(
            tmp_path / "d.csv", connect=lambda: FakeFtp(server), sleep=lambda _seconds: None
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("xml", "message"),
    [
        (document(vehicle(), namespace="http://example.org/other"), "namespace"),
        (document(vehicle(), root="SomethingElse"), "document root"),
        (document(vehicle(art="3", art_name="Varebil")), "no Personbil"),
        (document(vehicle(art_name="Lastbil")), "no longer named Personbil"),
        (document(vehicle(make=None)), "make"),
        (document(vehicle(model=None)), "model"),
        (document(vehicle(first=None)), "first_registration"),
    ],
)
def test_schema_drift_raises(tmp_path: Path, xml: bytes, message: str) -> None:
    server, _ = serve(xml)

    with pytest.raises(ValueError, match=message):
        acquire_fleet(tmp_path / "d.csv", connect=lambda: FakeFtp(server))
    assert list(tmp_path.iterdir()) == []


def test_a_moved_make_element_is_drift_not_a_blank_label(tmp_path: Path) -> None:
    moved = vehicle().replace(
        "<ns:KoeretoejBetegnelseStruktur><ns:KoeretoejMaerkeTypeNavn>AURORA"
        "</ns:KoeretoejMaerkeTypeNavn>",
        "<ns:KoeretoejBetegnelseStruktur><ns:Wrapper><ns:KoeretoejMaerkeTypeNavn>AURORA"
        "</ns:KoeretoejMaerkeTypeNavn></ns:Wrapper>",
    )
    server, _ = serve(document(moved))

    with pytest.raises(ValueError, match="no passenger car holds make"):
        acquire_fleet(tmp_path / "d.csv", connect=lambda: FakeFtp(server))


def test_no_vehicles_and_malformed_xml_are_refused(tmp_path: Path) -> None:
    for xml, message in (
        (document(), "no Personbil"),
        (SAMPLE[: len(SAMPLE) // 2], "not well formed"),
        (b"", "not well formed"),
    ):
        server, _ = serve(xml)
        with pytest.raises(ValueError, match=message):
            acquire_fleet(tmp_path / "d.csv", connect=lambda s=server: FakeFtp(s))


def test_the_zip_is_checked_against_its_directory() -> None:
    assert zlib.crc32(SAMPLE) != 0

    def read(archive: bytes) -> int:
        counter, reader, entry = aggregate_zip(io.BytesIO(archive), limit=len(archive))
        assert reader.bytes_read == len(archive)
        assert entry.inflated_bytes == len(SAMPLE)
        return counter.registered_rows

    assert read(server_zip(SAMPLE)) == 5
    with pytest.raises(ValueError, match="CRC-32"):
        read(server_zip(SAMPLE, crc_delta=1))
    wrong_name = server_zip(SAMPLE, name="Other.xml")
    with pytest.raises(ValueError, match="unexpected name"):
        read(wrong_name)
    with pytest.raises(ValueError, match="not a ZIP"):
        read(b"<?xml version='1.0'?>" + b" " * 64)
    with pytest.raises(ValueError, match="ends"):
        read(server_zip(SAMPLE)[:200])
    with pytest.raises(ValueError, match="central directory"):
        read(server_zip(SAMPLE).replace(b"PK\x01\x02", b"PK\x01\x03"))


def test_standard_zip_writers_are_read_as_well() -> None:
    for seekable in (True, False):
        archive = python_zip(SAMPLE, seekable=seekable)
        counter, reader, _ = aggregate_zip(io.BytesIO(archive), limit=len(archive))
        assert counter.registered_rows == 5
        assert reader.sha256 == hashlib.sha256(archive).hexdigest()


def test_an_archive_with_a_second_entry_or_stored_data_is_refused() -> None:
    two = python_zip(SAMPLE, seekable=True, entries=(ENTRY_NAME, "Second.xml"))
    with pytest.raises(ValueError, match="more than its one entry"):
        aggregate_zip(io.BytesIO(two), limit=len(two))
    stored = io.BytesIO()
    with zipfile.ZipFile(stored, "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr(ENTRY_NAME, SAMPLE)
    with pytest.raises(ValueError, match="not deflate"):
        aggregate_zip(io.BytesIO(stored.getvalue()))
    encrypted = bytearray(server_zip(SAMPLE))
    encrypted[6] |= 0x01
    with pytest.raises(ValueError, match="encrypted"):
        aggregate_zip(io.BytesIO(bytes(encrypted)))


def test_the_inflater_stops_at_the_end_of_the_deflate_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deflater = zlib.compressobj(6, zlib.DEFLATED, -15)
    deflated = len(deflater.compress(SAMPLE) + deflater.flush())
    entry = DeflatedEntry(io.BytesIO(server_zip(SAMPLE)))
    assert b"".join(entry.chunks()) == SAMPLE
    assert (entry.inflated_bytes, entry.compressed_bytes) == (len(SAMPLE), deflated)
    # Small reads and a small output cap give the same bytes: nothing is lost between them.
    monkeypatch.setattr(dk, "_READ_SIZE", 37)
    monkeypatch.setattr(dk, "_MAX_OUTPUT", 101)
    capped = DeflatedEntry(io.BytesIO(server_zip(SAMPLE)))
    assert b"".join(capped.chunks()) == SAMPLE
    assert capped.crc32 == zlib.crc32(SAMPLE)


def test_progress_reports_bytes_and_records(tmp_path: Path) -> None:
    messages: list[str] = []
    server, archive = serve(SAMPLE)
    ticks = iter(range(0, 10_000, 100))

    acquire_fleet(
        tmp_path / "d.csv",
        connect=lambda: FakeFtp(server),
        log=messages.append,
        interval=50.0,
        clock=lambda: float(next(ticks)),
    )

    reports = [message for message in messages if message.startswith("bytes=")]
    assert reports
    assert all("records=" in message and "inflated=" in message for message in reports)
    assert f"bytes={len(archive)}" in "\n".join(messages)


def test_the_reader_thread_hands_over_chunks_and_errors() -> None:
    assert list(prefetched(iter([b"a", b"b", b"c"]), depth=1)) == [b"a", b"b", b"c"]

    def failing():  # type: ignore[no-untyped-def]
        yield b"a"
        raise OSError("link down")

    with pytest.raises(OSError, match="link down"):
        list(prefetched(failing(), depth=1))
    # Stopping early releases the thread instead of leaving it blocked on a full queue.
    stream = prefetched(iter([b"x"] * 1000), depth=1)
    assert next(stream) == b"x"
    stream.close()
