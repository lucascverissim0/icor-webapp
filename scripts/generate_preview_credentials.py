#!/usr/bin/env python3
"""Generate copyable Codespaces preview verifiers without persisting secrets."""

from __future__ import annotations

import argparse
import base64
import getpass
import re
import secrets
from collections.abc import Callable, Sequence

from argon2 import PasswordHasher

_USERNAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}\Z")


class CredentialError(ValueError):
    """Credential input is unsafe or inconsistent."""


def hash_user(
    username: str,
    *,
    password_reader: Callable[[str], str] = getpass.getpass,
) -> str:
    if _USERNAME.fullmatch(username) is None:
        raise CredentialError("preview username is invalid")
    password = password_reader("Password: ")
    confirmation = password_reader("Confirm password: ")
    if not password or password != confirmation:
        raise CredentialError("password confirmation failed")
    return PasswordHasher().hash(password)


def session_secret(
    *, random_bytes: Callable[[int], bytes] = secrets.token_bytes
) -> str:
    return base64.urlsafe_b64encode(random_bytes(32)).decode("ascii").rstrip("=")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ICOR Codespaces preview secrets.")
    commands = parser.add_subparsers(dest="command", required=True)
    user = commands.add_parser("hash-user", help="Create one Argon2id password verifier")
    user.add_argument("--username", required=True)
    commands.add_parser("session-secret", help="Create a random session signing secret")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "hash-user":
            value = hash_user(args.username)
            print(f"ICOR_PREVIEW_USERS value for {args.username}: {value}")
        else:
            print(f"ICOR_PREVIEW_SESSION_SECRET: {session_secret()}")
    except CredentialError:
        print("Credential generation failed.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
