from __future__ import annotations

import base64
from collections.abc import Iterator

import pytest
from argon2 import PasswordHasher


def test_hash_user_uses_hidden_double_entry_and_emits_argon2id_only() -> None:
    from scripts.generate_preview_credentials import hash_user

    answers: Iterator[str] = iter(("private-password", "private-password"))
    output = hash_user("Lucas", password_reader=lambda prompt: next(answers))

    assert output.startswith("$argon2id$")
    assert PasswordHasher().verify(output, "private-password")
    assert "private-password" not in output


def test_hash_user_rejects_mismatch_and_invalid_name_without_echo() -> None:
    from scripts.generate_preview_credentials import CredentialError, hash_user

    answers: Iterator[str] = iter(("first-secret", "second-secret"))
    with pytest.raises(CredentialError, match="confirmation") as mismatch:
        hash_user("Lucas", password_reader=lambda prompt: next(answers))
    assert "first-secret" not in str(mismatch.value)
    assert "second-secret" not in str(mismatch.value)

    with pytest.raises(CredentialError, match="username") as invalid:
        hash_user("invalid\nsecret", password_reader=lambda prompt: "unused")
    assert "invalid\nsecret" not in str(invalid.value)


def test_session_secret_is_exactly_32_random_bytes() -> None:
    from scripts.generate_preview_credentials import session_secret

    value = session_secret(random_bytes=lambda size: bytes(range(size)))
    assert base64.urlsafe_b64decode(value + "==") == bytes(range(32))
    assert "=" not in value


def test_cli_has_no_plaintext_password_argument() -> None:
    from scripts.generate_preview_credentials import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["hash-user", "--username", "Lucas", "--password", "secret"])
