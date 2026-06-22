import re
from dataclasses import dataclass, field

TURN_HW_MULTI_PATTERN = re.compile(r"^turn_(\d+)_homework_(\d+)\.(.+)$")
TURN_HW_LEGACY_PATTERN = re.compile(r"^turn_(\d+)_homework\.(.+)$")
TURN_RESP_PATTERN = re.compile(r"^turn_(\d+)_response\.json$")


@dataclass
class HomeworkFile:
    filename: str
    key: str
    index: int


@dataclass
class TurnData:
    homework_files: list[HomeworkFile] = field(default_factory=list)
    response_key: str | None = None


def _get_or_create_turn(turns: dict[int, TurnData], turn_num: int) -> TurnData:
    if turn_num not in turns:
        turns[turn_num] = TurnData()
    return turns[turn_num]


def _add_homework_file(
    turn: TurnData, index: int, key: str, filename: str
) -> None:
    turn.homework_files.append(
        HomeworkFile(filename=filename, key=key, index=index)
    )


def parse_turn_files(keys: list[str], prefix: str) -> dict[int, TurnData]:
    """Groups S3 object keys under a conversation prefix by turn number."""
    turns: dict[int, TurnData] = {}

    for key in keys:
        filename = key[len(prefix) :]
        if filename == "meta.json":
            continue

        multi_match = TURN_HW_MULTI_PATTERN.match(filename)
        if multi_match:
            turn_num = int(multi_match.group(1))
            index = int(multi_match.group(2))
            turn = _get_or_create_turn(turns, turn_num)
            _add_homework_file(turn, index, key, filename)
            continue

        legacy_match = TURN_HW_LEGACY_PATTERN.match(filename)
        if legacy_match:
            turn_num = int(legacy_match.group(1))
            turn = _get_or_create_turn(turns, turn_num)
            _add_homework_file(turn, 0, key, filename)
            continue

        resp_match = TURN_RESP_PATTERN.match(filename)
        if resp_match:
            turn = _get_or_create_turn(turns, int(resp_match.group(1)))
            turn.response_key = key

    return turns
