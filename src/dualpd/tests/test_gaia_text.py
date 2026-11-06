import json

import pytest

from dualpd.data.config import DatasetSpec
from dualpd.data.gaia_text.harness import parse_action
from dualpd.data.gaia_text.loader import load_samples
from dualpd.data.gaia_text.tools import validate_call


def test_loader_keeps_answer_out_of_prompt(tmp_path):
    path = tmp_path / "gaia.jsonl"
    path.write_text(
        json.dumps({"query_id": 7, "question": "What year?", "answer": "1999", "Level": 2})
        + "\n"
    )
    samples = load_samples(None, DatasetSpec("gaia", "gaia_text", str(path)))
    assert len(samples) == 1
    assert samples[0].prompt[-1]["content"] == "What year?"
    assert "1999" not in str(samples[0].prompt)
    assert samples[0].metadata["gold_answer"] == "1999"


def test_parse_tools_and_final():
    kind, calls = parse_action(
        '<think>x</think><tool_call>{"name":"search_engine_query","arguments":{"keywords":"x"}}</tool_call>'
    )
    assert kind == "tools" and calls[0]["name"] == "search_engine_query"
    assert parse_action("<final_answer>34689</final_answer>") == ("final", "34689")


@pytest.mark.parametrize(
    "call",
    [
        {"name": "code_interpreter", "arguments": {"code": "print(2+2)"}},
        {"name": "fetch_url_content", "arguments": {"url": "https://example.com"}},
    ],
)
def test_validate_supported_calls(call):
    assert validate_call(call)[0] == call["name"]


def test_reject_mixed_terminal_and_tool():
    with pytest.raises(ValueError, match="both"):
        parse_action(
            '<final_answer>x</final_answer><tool_call>{"name":"search_engine_query","arguments":{"keywords":"x"}}</tool_call>'
        )
