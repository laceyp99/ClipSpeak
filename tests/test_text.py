from clipspeak.text import markdown_to_speech


def test_headings_quotes_lists_tasks_and_rules() -> None:
    source = """# Main title #
Short introduction
-------------
> Quoted **idea**
> - [x] First task
1. Second task
---
End"""
    assert markdown_to_speech(source) == (
        "Main title\nShort introduction\nQuoted idea\nFirst task\nSecond task\nEnd"
    )


def test_inline_markup_and_links_keep_spoken_words() -> None:
    source = (
        "**bold _inner_**, *italic*, ~~old~~. "
        "Read [the docs](https://example.com/a_(b)) and "
        "![a red bird](bird.png). See [guide][ref]."
    )
    assert markdown_to_speech(source) == (
        "bold inner, italic, old. Read the docs and a red bird. See guide."
    )


def test_fenced_and_inline_code_preserve_literal_content() -> None:
    source = """~~~python
x * y
snake_case = "**literal**"
~~~
Use `a_b * c` and **bold**."""
    assert markdown_to_speech(source) == (
        'x * y\nsnake_case = "**literal**"\nUse a_b * c and bold.'
    )


def test_single_line_triple_backticks_keep_their_words() -> None:
    assert markdown_to_speech("```items[0]```") == "items[0]"


def test_long_quote_prefix_finishes_without_repeated_slicing() -> None:
    assert markdown_to_speech(">" * 99_995 + "Hello") == "Hello"


def test_plain_punctuation_and_unmatched_markers_survive() -> None:
    source = "C#, x * y, snake_case, https://example.com/a_b, an *unmatched marker"
    assert markdown_to_speech(source) == source


def test_escaped_markers_stay_literal_and_bare_image_syntax_is_conservative() -> None:
    assert markdown_to_speech(r"Literal \*stars\* and ![unfinished]") == (
        "Literal *stars* and ![unfinished]"
    )


def test_nested_brackets_and_parentheses_in_link() -> None:
    assert markdown_to_speech("[an [inner] label](https://example.com/a_(b))") == (
        "an inner label"
    )


def test_link_label_preserves_brackets_inside_inline_code() -> None:
    assert markdown_to_speech("[Use `items[0]`](https://example.com)") == "Use items[0]"


def test_windows_paths_keep_backslashes_and_triple_emphasis_reads_cleanly() -> None:
    source = r"Open C:\Users\Pat\Notes then read ***bold italic*** and \*literal\*."
    assert markdown_to_speech(source) == (
        r"Open C:\Users\Pat\Notes then read bold italic and *literal*."
    )


def test_malformed_markup_is_preserved_at_queue_limit() -> None:
    unmatched_code = "before " + "".join("`" * width + "x " for width in range(1, 445))
    unmatched_links = "[a]( " * 20_000
    assert markdown_to_speech(unmatched_code) == unmatched_code.strip()
    assert markdown_to_speech(unmatched_links) == unmatched_links.strip()
