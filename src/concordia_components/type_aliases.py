import dataclasses

@dataclasses.dataclass
class Thread:
    id: int
    content: list[dict[str, str]]