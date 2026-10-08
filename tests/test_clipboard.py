"""Windows clipboard ownership and text-boundary checks."""

import ctypes

from clipspeak.clipboard import ClipboardStatus, read_clipboard_text


class FakeClipboard:
    def __init__(self, text: str | None = "hello", *, raw: bytes | None = None):
        self.raw = raw if raw is not None else (text.encode("utf-16-le") + b"\0\0" if text is not None else b"")
        self.buffer = ctypes.create_string_buffer(self.raw)
        self.busy = 0
        self.available = text is not None or raw is not None
        self.formats = 1
        self.lock_ok = True
        self.data_ok = True
        self.opens = 0
        self.closes = 0
        self.locks = 0
        self.unlocks = 0

    def OpenClipboard(self, owner):
        self.opens += 1
        return self.opens > self.busy

    def CloseClipboard(self):
        self.closes += 1
        return 1

    def IsClipboardFormatAvailable(self, format_id):
        assert format_id == 13
        return self.available

    def GetClipboardData(self, format_id):
        return 1 if self.data_ok else 0

    def CountClipboardFormats(self):
        return self.formats

    def GlobalLock(self, handle):
        self.locks += 1
        return ctypes.addressof(self.buffer) if self.lock_ok else 0

    def GlobalSize(self, handle):
        return len(self.raw)

    def GlobalUnlock(self, handle):
        self.unlocks += 1
        return 1


def read(fake, sleep=lambda _: None):
    return read_clipboard_text(_functions=(fake, fake), _sleep=sleep)


def test_busy_clipboard_retries_then_copies_and_releases():
    fake = FakeClipboard("A😀\n punctuation, preserved! ")
    fake.busy = 2
    delays = []
    result = read(fake, delays.append)
    assert result.status is ClipboardStatus.OK
    assert result.text == "A😀\n punctuation, preserved! "
    assert (fake.opens, fake.closes, fake.locks, fake.unlocks) == (3, 1, 1, 1)
    assert delays == [0.05, 0.05]
    assert "punctuation" not in repr(result)
    fake.buffer.raw = b"x" * len(fake.buffer)
    assert result.text == "A😀\n punctuation, preserved! "


def test_busy_clipboard_exhausts_five_attempts():
    fake = FakeClipboard()
    fake.busy = 5
    delays = []
    assert read(fake, delays.append).status is ClipboardStatus.UNAVAILABLE
    assert (fake.opens, fake.closes, fake.locks) == (5, 0, 0)
    assert delays == [0.05] * 4


def test_non_text_and_whitespace_are_distinct():
    empty = FakeClipboard(None)
    empty.formats = 0
    assert read(empty).status is ClipboardStatus.EMPTY
    assert empty.closes == 1
    non_text = FakeClipboard(None)
    assert read(non_text).status is ClipboardStatus.NON_TEXT
    assert non_text.closes == 1
    whitespace = FakeClipboard(" \t\r\n ")
    assert read(whitespace).status is ClipboardStatus.EMPTY
    assert whitespace.unlocks == whitespace.closes == 1


def test_get_data_and_lock_failures_release_owned_resources():
    data_failure = FakeClipboard()
    data_failure.data_ok = False
    assert read(data_failure).status is ClipboardStatus.UNAVAILABLE
    assert (data_failure.closes, data_failure.unlocks) == (1, 0)
    lock_failure = FakeClipboard()
    lock_failure.lock_ok = False
    assert read(lock_failure).status is ClipboardStatus.UNAVAILABLE
    assert (lock_failure.closes, lock_failure.unlocks) == (1, 0)


def test_bounded_read_handles_padding_and_python_character_limit():
    padded = FakeClipboard(raw="😀".encode("utf-16-le") * 100_000 + b"\0\0" + b"x" * 100)
    assert read(padded).status is ClipboardStatus.OK
    assert padded.unlocks == padded.closes == 1
    too_long = FakeClipboard("x" * 100_001)
    assert read(too_long).status is ClipboardStatus.OVERSIZED
    assert too_long.unlocks == too_long.closes == 1


def test_malformed_utf16_and_missing_terminator_fail_safely():
    malformed = FakeClipboard(raw=b"\x00\xd8\0\0")
    assert read(malformed).status is ClipboardStatus.UNAVAILABLE
    unterminated = FakeClipboard(raw=b"h\0i\0")
    assert read(unterminated).status is ClipboardStatus.UNAVAILABLE
    assert malformed.unlocks == malformed.closes == 1
    assert unterminated.unlocks == unterminated.closes == 1


def test_copy_error_unlocks_and_closes():
    fake = FakeClipboard()
    def invalid_size(handle):
        raise ValueError("native read failed")
    fake.GlobalSize = invalid_size
    assert read(fake).status is ClipboardStatus.UNAVAILABLE
    assert fake.unlocks == fake.closes == 1
