"""Narrow durable-filesystem boundary for evidence snapshot publication."""

from __future__ import annotations

import ctypes
import os
import shutil
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


class SnapshotPathError(RuntimeError):
    """A snapshot filesystem path is unsafe or escaped its declared root."""


class SnapshotFilesystem:
    """No-follow path checks and durability operations used by snapshot storage."""

    def prepare_root(self, root: Path) -> Path:
        absolute = self._absolute(root)
        self._reject_existing_reparse_components(absolute)
        absolute.mkdir(parents=True, exist_ok=True)
        self.require_directory(absolute, absolute)
        return absolute

    def require_root(self, root: Path) -> Path:
        absolute = self._absolute(root)
        self.require_directory(absolute, absolute)
        return absolute

    def prepare_directory(self, path: Path, root: Path) -> Path:
        absolute_root = self.require_root(root)
        absolute = self._contained_absolute(path, absolute_root)
        self._reject_existing_reparse_components(absolute.parent)
        absolute.mkdir(parents=True, exist_ok=True)
        self.require_directory(absolute, absolute_root)
        return absolute

    def require_directory(self, path: Path, root: Path) -> Path:
        absolute_root = self._absolute(root)
        absolute = self._contained_absolute(path, absolute_root)
        self._reject_existing_reparse_components(absolute)
        if not absolute.is_dir():
            raise SnapshotPathError("snapshot directory is unavailable or unsafe")
        self._require_resolved_containment(absolute, absolute_root)
        return absolute

    def require_file(self, path: Path, root: Path) -> Path:
        absolute_root = self.require_root(root)
        absolute = self._contained_absolute(path, absolute_root)
        self._reject_existing_reparse_components(absolute)
        try:
            mode = absolute.lstat().st_mode
        except OSError as error:
            raise SnapshotPathError("snapshot file is unavailable or unsafe") from error
        if not stat.S_ISREG(mode):
            raise SnapshotPathError("snapshot file is unavailable or unsafe")
        self._require_resolved_containment(absolute, absolute_root)
        return absolute

    def require_absent(self, path: Path, root: Path) -> Path:
        absolute_root = self.require_root(root)
        absolute = self._contained_absolute(path, absolute_root)
        self._reject_existing_reparse_components(absolute.parent)
        if os.path.lexists(absolute):
            raise FileExistsError(f"snapshot path already exists: {absolute.name}")
        return absolute

    def copy_file(
        self,
        source: Path,
        destination: Path,
        *,
        source_root: Path,
        destination_root: Path,
    ) -> None:
        safe_source = self.require_file(source, source_root)
        safe_destination = self.require_absent(destination, destination_root)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        source_fd = os.open(safe_source, flags)
        try:
            if not stat.S_ISREG(os.fstat(source_fd).st_mode):
                raise SnapshotPathError("snapshot source file changed during copy")
            destination_fd = os.open(
                safe_destination,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                while chunk := os.read(source_fd, 1024 * 1024):
                    view = memoryview(chunk)
                    while view:
                        written = os.write(destination_fd, view)
                        view = view[written:]
            finally:
                os.close(destination_fd)
        finally:
            os.close(source_fd)
        self.require_file(safe_destination, destination_root)

    def fsync_file(self, path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def fsync_directory(self, path: Path) -> None:
        if os.name != "nt":
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return
        self._flush_windows_directory(path)

    def publish_directory(self, source: Path, destination: Path) -> None:
        safe_root = self.require_root(source.parent)
        deadline = time.monotonic() + 1.0
        while True:
            safe_source = self.require_directory(source, safe_root)
            safe_destination = self.require_absent(destination, safe_root)
            try:
                os.rename(safe_source, safe_destination)
                return
            except PermissionError:
                if os.name != "nt" or time.monotonic() >= deadline:
                    raise
                # Closed SQLite/flush handles and filesystem scanners can retain a
                # transient Windows sharing reservation. Recheck every path before
                # retrying; a real ACL denial still fails after the bounded interval.
                time.sleep(0.01)

    def replace_verified_file(
        self,
        source: Path,
        destination: Path,
        verify: Callable[[], None],
    ) -> None:
        safe_root = self.require_root(source.parent)
        safe_source = self.require_file(source, safe_root)
        safe_destination = self._contained_absolute(destination, safe_root)
        self._reject_existing_reparse_components(safe_destination.parent)
        if os.path.lexists(safe_destination):
            self.require_file(safe_destination, safe_root)
        verify()
        self.require_file(safe_source, safe_root)
        self._reject_existing_reparse_components(safe_destination.parent)
        os.replace(safe_source, safe_destination)

    def make_immutable(self, directory: Path, root: Path) -> None:
        safe_directory = self.require_directory(directory, root)
        for entry in safe_directory.iterdir():
            safe_file = self.require_file(entry, root)
            safe_file.chmod(stat.S_IREAD)
        safe_directory.chmod(stat.S_IREAD | stat.S_IEXEC)

    def cleanup_directory(self, directory: Path, root: Path) -> None:
        if not os.path.lexists(directory):
            return
        safe_directory = self.require_directory(directory, root)
        for current_root, directories, files in os.walk(safe_directory, topdown=False):
            current = Path(current_root)
            self.require_directory(current, root)
            for filename in files:
                file_path = self.require_file(current / filename, root)
                file_path.chmod(stat.S_IREAD | stat.S_IWRITE)
            for dirname in directories:
                child = self.require_directory(current / dirname, root)
                child.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        safe_directory.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        shutil.rmtree(safe_directory)

    @contextmanager
    def promotion_lock(self, root: Path, *, timeout: float = 10.0) -> Iterator[None]:
        safe_root = self.require_root(root)
        lock = self._contained_absolute(safe_root / ".promotion.lock", safe_root)
        deadline = time.monotonic() + timeout
        while True:
            try:
                lock.mkdir()
                break
            except FileExistsError:
                self.require_directory(lock, safe_root)
                if time.monotonic() >= deadline:
                    raise SnapshotPathError(
                        "snapshot promotion lock is unavailable"
                    ) from None
                time.sleep(0.01)
        try:
            self.require_directory(lock, safe_root)
            yield
        finally:
            self.require_directory(lock, safe_root).rmdir()

    @classmethod
    def _reject_existing_reparse_components(cls, path: Path) -> None:
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            if not os.path.lexists(current):
                continue
            try:
                metadata = current.lstat()
            except OSError as error:
                raise SnapshotPathError("snapshot path cannot be inspected safely") from error
            attributes = getattr(metadata, "st_file_attributes", 0)
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            if stat.S_ISLNK(metadata.st_mode) or attributes & reparse_flag:
                raise SnapshotPathError("snapshot path contains an unsafe reparse component")

    @staticmethod
    def _absolute(path: Path) -> Path:
        return Path(os.path.abspath(Path(path)))

    @classmethod
    def _contained_absolute(cls, path: Path, root: Path) -> Path:
        absolute = cls._absolute(path)
        try:
            absolute.relative_to(cls._absolute(root))
        except ValueError as error:
            raise SnapshotPathError("snapshot path escapes its storage root") from error
        return absolute

    @staticmethod
    def _require_resolved_containment(path: Path, root: Path) -> None:
        try:
            path.resolve(strict=True).relative_to(root.resolve(strict=True))
        except (OSError, ValueError) as error:
            raise SnapshotPathError("snapshot path escapes its storage root") from error

    @staticmethod
    def _flush_windows_directory(path: Path) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        create_file.restype = ctypes.c_void_p
        flush_file_buffers = kernel32.FlushFileBuffers
        flush_file_buffers.argtypes = [ctypes.c_void_p]
        flush_file_buffers.restype = ctypes.c_int
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        handle = create_file(
            str(path),
            0x80000000,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,
            0x02000000,
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle == invalid_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not flush_file_buffers(handle):
                error_code = ctypes.get_last_error()
                if error_code != 5:
                    raise ctypes.WinError(error_code)
                # Windows opens directory handles for metadata ordering but commonly
                # rejects FlushFileBuffers on them. Publication still uses same-volume
                # atomic renames; the seam preserves an explicit directory flush point.
        finally:
            close_handle(handle)
