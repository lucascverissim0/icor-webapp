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
from uuid import uuid4


class _WindowsFileInformation(ctypes.Structure):
    _fields_ = [
        ("file_attributes", ctypes.c_uint32),
        ("creation_time_low", ctypes.c_uint32),
        ("creation_time_high", ctypes.c_uint32),
        ("last_access_time_low", ctypes.c_uint32),
        ("last_access_time_high", ctypes.c_uint32),
        ("last_write_time_low", ctypes.c_uint32),
        ("last_write_time_high", ctypes.c_uint32),
        ("volume_serial_number", ctypes.c_uint32),
        ("file_size_high", ctypes.c_uint32),
        ("file_size_low", ctypes.c_uint32),
        ("number_of_links", ctypes.c_uint32),
        ("file_index_high", ctypes.c_uint32),
        ("file_index_low", ctypes.c_uint32),
    ]


class _StableSnapshot:
    def __init__(self, filesystem: SnapshotFilesystem, paths: tuple[Path, ...]) -> None:
        self.filesystem = filesystem
        self.paths = paths
        self.handles: list[int] = []
        self.identities: list[tuple[int, ...]] = []
        self.digests: tuple[str, ...] | None = None

    def __enter__(self) -> _StableSnapshot:
        try:
            for index, path in enumerate(self.paths):
                handle, identity = self.filesystem._open_stable_path(
                    path, directory=index == 0
                )
                self.handles.append(handle)
                self.identities.append(identity)
        except BaseException:
            self.close()
            raise
        return self

    def seal(self) -> None:
        self.assert_paths_unchanged()
        self.digests = tuple(
            self.filesystem._digest_open_file(handle) for handle in self.handles[1:]
        )

    def assert_unchanged(self) -> None:
        if self.digests is None:
            raise SnapshotPathError("stable snapshot was not sealed")
        self.assert_paths_unchanged()
        if tuple(
            self.filesystem._digest_open_file(handle) for handle in self.handles[1:]
        ) != self.digests:
            raise SnapshotPathError("stable snapshot file content changed")

    def assert_paths_unchanged(self) -> None:
        for index, (path, expected) in enumerate(
            zip(self.paths, self.identities, strict=True)
        ):
            handle, actual = self.filesystem._open_stable_path(
                path, directory=index == 0
            )
            self.filesystem._close_stable_handle(handle)
            if actual != expected:
                raise SnapshotPathError("stable snapshot path identity changed")

    def close(self) -> None:
        while self.handles:
            self.filesystem._close_stable_handle(self.handles.pop())

    def __exit__(self, *_: object) -> None:
        self.close()


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
        *,
        stable_directory: Path,
        stable_files: tuple[Path, ...],
    ) -> None:
        safe_root = self.require_root(source.parent)
        safe_source = self.require_file(source, safe_root)
        safe_destination = self._contained_absolute(destination, safe_root)
        self._reject_existing_reparse_components(safe_destination.parent)
        if os.path.lexists(safe_destination):
            self.require_file(safe_destination, safe_root)
        safe_stable_directory = self.require_directory(stable_directory, safe_root)
        safe_stable_files = tuple(
            self.require_file(path, safe_root) for path in stable_files
        )
        previous = safe_root / f".active.previous.{uuid4().hex}"
        had_previous = os.path.lexists(safe_destination)
        if had_previous:
            self.copy_file(
                safe_destination,
                previous,
                source_root=safe_root,
                destination_root=safe_root,
            )
            self.fsync_file(previous)
        replaced = False
        try:
            with _StableSnapshot(
                self, (safe_stable_directory, *safe_stable_files)
            ) as stable:
                verify()
                stable.seal()
                stable.assert_unchanged()
                self.require_file(safe_source, safe_root)
                self._reject_existing_reparse_components(safe_destination.parent)
                self.replace_atomic_file(safe_source, safe_destination)
                replaced = True
                stable.assert_unchanged()
        except BaseException:
            if had_previous and os.path.lexists(previous):
                self._replace_atomic_file_raw(previous, safe_destination)
                self.fsync_directory(safe_root)
            elif replaced and os.path.lexists(safe_destination):
                self.require_file(safe_destination, safe_root).unlink()
                self.fsync_directory(safe_root)
            raise
        finally:
            if os.path.lexists(previous):
                self.require_file(previous, safe_root).unlink()

    def replace_atomic_file(self, source: Path, destination: Path) -> None:
        """Replace one file atomically; separated for filesystem fault injection."""
        self._replace_atomic_file_raw(source, destination)

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
        self._reject_existing_reparse_components(lock)
        descriptor = os.open(
            lock,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise SnapshotPathError("snapshot promotion lock is unsafe")
        deadline = time.monotonic() + timeout
        acquired = False
        try:
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            self._acquire_file_lock(descriptor, deadline)
            acquired = True
            yield
        finally:
            if acquired:
                self._release_file_lock(descriptor)
            os.close(descriptor)

    @staticmethod
    def _acquire_file_lock(descriptor: int, deadline: float) -> None:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise SnapshotPathError(
                        "snapshot promotion lock is unavailable"
                    ) from None
                time.sleep(0.01)

    @staticmethod
    def _release_file_lock(descriptor: int) -> None:
        if os.name == "nt":
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)

    def _open_stable_path(
        self, path: Path, *, directory: bool
    ) -> tuple[int, tuple[int, ...]]:
        if os.name == "nt":
            return self._open_windows_stable_path(path, directory=directory)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        if directory:
            flags |= getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise SnapshotPathError("stable snapshot path cannot be opened") from error
        metadata = os.fstat(descriptor)
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected_type(metadata.st_mode):
            os.close(descriptor)
            raise SnapshotPathError("stable snapshot path has an unsafe type")
        return descriptor, (metadata.st_dev, metadata.st_ino)

    @staticmethod
    def _digest_open_file(handle: int) -> str:
        from hashlib import sha256

        if os.name == "nt":
            digest = sha256()
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            set_pointer = kernel32.SetFilePointerEx
            set_pointer.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_uint32,
            ]
            set_pointer.restype = ctypes.c_int
            read_file = kernel32.ReadFile
            read_file.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_void_p,
            ]
            read_file.restype = ctypes.c_int
            if not set_pointer(handle, 0, None, 0):
                raise ctypes.WinError(ctypes.get_last_error())
            buffer = ctypes.create_string_buffer(1024 * 1024)
            bytes_read = ctypes.c_uint32()
            while True:
                if not read_file(
                    handle,
                    buffer,
                    len(buffer),
                    ctypes.byref(bytes_read),
                    None,
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                if bytes_read.value == 0:
                    break
                digest.update(buffer.raw[: bytes_read.value])
            return digest.hexdigest()
        os.lseek(handle, 0, os.SEEK_SET)
        digest = sha256()
        while chunk := os.read(handle, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _open_windows_stable_path(
        path: Path, *, directory: bool
    ) -> tuple[int, tuple[int, ...]]:
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
        flags = 0x00200000
        if directory:
            flags |= 0x02000000
        handle = create_file(
            str(path),
            0x80000000,
            0x00000001,
            None,
            3,
            flags,
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle == invalid_handle:
            raise SnapshotPathError("stable snapshot path cannot be opened") from ctypes.WinError(
                ctypes.get_last_error()
            )
        information = _WindowsFileInformation()
        get_information = kernel32.GetFileInformationByHandle
        get_information.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_WindowsFileInformation),
        ]
        get_information.restype = ctypes.c_int
        if not get_information(handle, ctypes.byref(information)):
            SnapshotFilesystem._close_stable_handle(handle)
            raise ctypes.WinError(ctypes.get_last_error())
        is_directory = bool(information.file_attributes & 0x00000010)
        is_reparse_point = bool(information.file_attributes & 0x00000400)
        if is_directory != directory or is_reparse_point:
            SnapshotFilesystem._close_stable_handle(handle)
            raise SnapshotPathError("stable snapshot path has an unsafe type")
        identity = (
            information.volume_serial_number,
            information.file_index_high,
            information.file_index_low,
        )
        return handle, identity

    @staticmethod
    def _close_stable_handle(handle: int) -> None:
        if os.name != "nt":
            os.close(handle)
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        close_handle(handle)

    @staticmethod
    def _replace_atomic_file_raw(source: Path, destination: Path) -> None:
        os.replace(source, destination)

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
