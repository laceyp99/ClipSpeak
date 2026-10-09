"""Turn common Markdown presentation syntax into text suitable for speech."""

from __future__ import annotations

import re


_HEADING = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+|$)")
_TRAILING_HEADING = re.compile(r"[ \t]+#+[ \t]*$")
_QUOTE = re.compile(r"^(?: {0,3}>[ \t]?)+")
_LIST = re.compile(r"^ {0,3}(?:[-+*]|\d{1,9}[.)])[ \t]+")
_TASK = re.compile(r"^\[[ xX]\][ \t]+")
_RULE = re.compile(r"^ {0,3}(?:\*[ \t]*){3,}$|^ {0,3}(?:-[ \t]*){3,}$|^ {0,3}(?:_[ \t]*){3,}$")
_SETEXT = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)$")


_MARKDOWN_ESCAPES = frozenset(r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~")


def _escaped(text: str, index: int) -> bool:
    return (
        text[index] == "\\"
        and index + 1 < len(text)
        and text[index + 1] in _MARKDOWN_ESCAPES
    )


def _code_spans(text: str) -> dict[int, tuple[int, int]]:
    """Pair runs of equal-length backticks in one pass."""
    pending: dict[int, int] = {}
    spans: dict[int, tuple[int, int]] = {}
    for run in re.finditer(r"`+", text):
        width = run.end() - run.start()
        opening = pending.pop(width, None)
        if opening is None:
            pending[width] = run.start()
        else:
            spans[opening] = (run.start(), width)
    return spans


def _tick_width(text: str, start: int) -> int:
    end = start + 1
    while end < len(text) and text[end] == "`":
        end += 1
    return end - start


def _link_pairs(
    text: str, code_spans: dict[int, tuple[int, int]]
) -> tuple[dict[int, int], dict[int, int]]:
    bracket_stack: list[int] = []
    parenthesis_stack: list[int] = []
    brackets: dict[int, int] = {}
    parentheses: dict[int, int] = {}
    i = 0
    while i < len(text):
        if _escaped(text, i):
            i += 2
            continue
        if text[i] == "`":
            code = code_spans.get(i)
            if code is not None:
                end, width = code
                i = end + width
                continue
            i += _tick_width(text, i)
            continue
        if text[i] == "[":
            bracket_stack.append(i)
        elif text[i] == "]" and bracket_stack:
            brackets[bracket_stack.pop()] = i
        elif text[i] == "(":
            parenthesis_stack.append(i)
        elif text[i] == ")" and parenthesis_stack:
            parentheses[parenthesis_stack.pop()] = i
        i += 1
    return brackets, parentheses


def _spoken_label(label: str) -> str:
    code_spans = _code_spans(label)
    result: list[str] = []
    i = 0
    while i < len(label):
        if _escaped(label, i):
            result.append(label[i : i + 2])
            i += 2
            continue
        if label[i] == "`":
            code = code_spans.get(i)
            if code is not None:
                end, width = code
                result.append(label[i : end + width])
                i = end + width
                continue
            width = _tick_width(label, i)
            result.append(label[i : i + width])
            i += width
            continue
        if label[i] not in "[]":
            result.append(label[i])
        i += 1
    return "".join(result)


def _strip_links(text: str) -> str:
    code_spans = _code_spans(text)
    brackets, parentheses = _link_pairs(text, code_spans)
    result: list[str] = []
    i = 0
    while i < len(text):
        if _escaped(text, i):
            result.append(text[i : i + 2])
            i += 2
            continue
        if text[i] == "`":
            code = code_spans.get(i)
            if code is not None:
                end, width = code
                result.append(text[i : end + width])
                i = end + width
                continue
            width = _tick_width(text, i)
            result.append(text[i : i + width])
            i += width
            continue
        image = text[i] == "!" and i + 1 < len(text) and text[i + 1] == "["
        label_start = i + 1 if image else i
        label_end = brackets.get(label_start) if text[label_start] == "[" else None
        if label_end is not None and label_end + 1 < len(text):
            suffix = text[label_end + 1]
            dest_end = None
            if suffix == "(":
                dest_end = parentheses.get(label_end + 1)
            elif suffix == "[":
                dest_end = brackets.get(label_end + 1)
            if dest_end is not None:
                label = text[label_start + 1 : label_end]
                result.append(_spoken_label(label))
                i = dest_end + 1
                continue
        result.append(text[i])
        i += 1
    return "".join(result)


def _strip_inline(text: str) -> str:
    remove = bytearray(len(text))
    code_spans = _code_spans(text)
    stack: list[tuple[str, int]] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if _escaped(text, i):
            remove[i] = 1
            i += 2
            continue
        if ch == "`":
            code = code_spans.get(i)
            if code is not None:
                end, width = code
                remove[i : i + width] = b"\x01" * width
                remove[end : end + width] = b"\x01" * width
                i = end + width
                continue
            i += _tick_width(text, i)
            continue
        marker = None
        if text.startswith(("***", "___"), i):
            marker = text[i : i + 3]
        elif text.startswith(("**", "__", "~~"), i):
            marker = text[i : i + 2]
        elif ch in "*_":
            marker = ch
        if marker is None:
            i += 1
            continue
        width = len(marker)
        before = text[i - 1] if i else ""
        after = text[i + width] if i + width < len(text) else ""
        can_open = bool(after and not after.isspace() and (not before or before.isspace() or before in "([{>\"'*_~"))
        can_close = bool(before and not before.isspace() and (not after or after.isspace() or after in ").,;:!?]}\"'*_~"))
        if marker.startswith("_") and before.isalnum() and after.isalnum():
            can_open = can_close = False
        if stack and stack[-1][0] == marker and can_close:
            _, opening = stack.pop()
            remove[opening : opening + width] = b"\x01" * width
            remove[i : i + width] = b"\x01" * width
        elif can_open:
            stack.append((marker, i))
        i += width
    return "".join(ch for pos, ch in enumerate(text) if not remove[pos])


def markdown_to_speech(text: str) -> str:
    """Remove common Markdown markers while retaining spoken content and code."""
    lines = text.splitlines()
    output: list[str] = []
    fence_char = ""
    fence_width = 0
    for index, original in enumerate(lines):
        line = original
        if fence_char:
            if re.fullmatch(rf" {{0,3}}{re.escape(fence_char)}{{{fence_width},}}[ \t]*", line):
                fence_char = ""
                fence_width = 0
            else:
                output.append(line)
            continue
        fence = _FENCE.match(line)
        if fence and not (fence.group(1).startswith("`") and "`" in fence.group(2)):
            fence_char = fence.group(1)[0]
            fence_width = len(fence.group(1))
            continue
        if _RULE.fullmatch(line):
            continue
        if _SETEXT.fullmatch(line) and index > 0 and lines[index - 1].strip():
            continue
        quote = _QUOTE.match(line)
        if quote:
            line = line[quote.end() :]
        heading = _HEADING.match(line)
        if heading:
            line = _TRAILING_HEADING.sub("", line[heading.end() :])
        listing = _LIST.match(line)
        if listing:
            line = line[listing.end() :]
            task = _TASK.match(line)
            if task:
                line = line[task.end() :]
        output.append(_strip_inline(_strip_links(line)))
    return "\n".join(output).strip()
