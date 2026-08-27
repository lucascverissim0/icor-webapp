from icor.evidence.normalization import normalize_vehicle_label, stable_evidence_id


def test_normalization_removes_layout_noise_without_guessing_aliases() -> None:
    assert normalize_vehicle_label("  ŠKODA\t OCTAVIA  ") == "škoda octavia"
    assert normalize_vehicle_label("VW") == "vw"
    assert normalize_vehicle_label("Volkswagen") == "volkswagen"


def test_normalization_rejects_publisher_missing_value_markers() -> None:
    for marker in ("", " ", "[x]", "[z]", "[c]", "n/a", "N/A", "-"):
        assert normalize_vehicle_label(marker) is None


def test_stable_evidence_id_is_repeatable_order_sensitive_and_identifier_safe() -> None:
    first = stable_evidence_id("obs", "DE", "VW", "GOLF")

    assert first == stable_evidence_id("obs", "DE", "VW", "GOLF")
    assert first != stable_evidence_id("obs", "DE", "GOLF", "VW")
    assert first.startswith("obs-")
    assert len(first) <= 80
    assert first.replace("-", "").isalnum()


def test_stable_evidence_id_separates_embedded_delimiters() -> None:
    assert stable_evidence_id("obs", "a|b", "c") != stable_evidence_id("obs", "a", "b|c")
