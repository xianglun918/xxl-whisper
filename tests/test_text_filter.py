"""Unit tests for the hesitation-filler text filter."""

from app.text_filter import filter_fillers

DEFAULTS = "嗯,呃,啊,哦,噢,诶,唉,那个"

EMPTY_FILLERS = ""


def test_removes_leading_filler_and_separator() -> None:
    assert filter_fillers("嗯嗯，我们走吧", DEFAULTS) == "我们走吧"


def test_keeps_semantic_particle_mid_phrase() -> None:
    assert filter_fillers("好啊", DEFAULTS) == "好啊"


def test_removes_filler_yes_is_preserved_content() -> None:
    assert filter_fillers("嗯，是的", DEFAULTS) == "是的"


def test_removes_multiple_fillers_across_clauses() -> None:
    assert filter_fillers("呃那个啊，然后我们做什么", DEFAULTS) == "然后我们做什么"


def test_removes_filler_but_keeps_connective_meaning() -> None:
    assert filter_fillers("嗯，然后我们做什么", DEFAULTS) == "然后我们做什么"


def test_keeps_meaning_that_clause_initial_but_real_word() -> None:
    assert filter_fillers("他是那个部门的经理", DEFAULTS) == "他是那个部门的经理"


def test_sentence_with_no_fillers_unchanged() -> None:
    text = "今天天气怎么样？我们一起去吃火锅吧"
    assert filter_fillers(text, DEFAULTS) == text


def test_all_fillers_removes_to_empty() -> None:
    assert filter_fillers("嗯啊呃", DEFAULTS) == ""


def test_empty_input() -> None:
    assert filter_fillers("", DEFAULTS) == ""
    assert filter_fillers("  ", DEFAULTS) == ""


def test_empty_filler_list_is_noop() -> None:
    assert filter_fillers("嗯，是的", EMPTY_FILLERS) == "嗯，是的"


def test_multichar_filler_matches_before_matches_single() -> None:
    words = "那,那个"
    assert filter_fillers("那个，我们走", words) == "我们走"


def test_not_removed_when_not_at_clause_start() -> None:
    assert filter_fillers("我们那个部门", DEFAULTS) == "我们那个部门"
