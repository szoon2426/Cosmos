from dataclasses import dataclass, field


@dataclass
class SessionState:
    is_active: bool = False
    tracked_person_id: str | None = None
    metadata: dict[str, float | str] = field(default_factory=dict)
