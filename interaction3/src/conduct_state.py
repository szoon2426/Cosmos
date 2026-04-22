from __future__ import annotations

from dataclasses import dataclass

if __package__ in (None, ""):
    from conduct_features import ConductFeatures
else:
    from .conduct_features import ConductFeatures


READY_HOLD_SECONDS = 0.7
ACTIVE_INDEX_LOSS_GRACE_SECONDS = 0.6
ACTIVE_TRACKING_LOSS_SECONDS = 1.0


@dataclass(slots=True)
class ConductSession:
    mode: str = "idle"
    ready_started_at: float | None = None
    index_missing_started_at: float | None = None
    tracking_missing_started_at: float | None = None
    active_started_at: float | None = None
    base_right_index_x: float = 0.0
    base_right_index_y: float = 0.0
    base_left_hand_open: float = 0.0

    def is_active(self) -> bool:
        return self.mode == "conduct_active"

    def _enter_active(self, features: ConductFeatures, now: float) -> list[str]:
        self.mode = "conduct_active"
        self.active_started_at = now
        self.index_missing_started_at = None
        self.tracking_missing_started_at = None
        self.base_right_index_x = features.right_index_x
        self.base_right_index_y = features.right_index_y
        self.base_left_hand_open = features.left_hand_open if features.left_hand_open_valid else 0.0
        return ["ready_enabled", "conduct_started"]

    def _exit_active(self) -> list[str]:
        self.mode = "idle"
        self.ready_started_at = None
        self.index_missing_started_at = None
        self.tracking_missing_started_at = None
        self.active_started_at = None
        return ["ready_disabled", "conduct_stopped"]

    def update(self, features: ConductFeatures, now: float) -> list[str]:
        events: list[str] = []

        if self.mode == "idle":
            if features.right_index_raised:
                self.mode = "conduct_ready"
                self.ready_started_at = now
            return events

        if self.mode == "conduct_ready":
            if not features.right_index_raised:
                self.mode = "idle"
                self.ready_started_at = None
                return events

            if self.ready_started_at is None:
                self.ready_started_at = now

            if now - self.ready_started_at >= READY_HOLD_SECONDS:
                return self._enter_active(features, now)
            return events

        if self.mode == "conduct_active":
            if not features.right_hand_visible:
                if self.tracking_missing_started_at is None:
                    self.tracking_missing_started_at = now
            else:
                self.tracking_missing_started_at = None

            if not features.right_index_raised:
                if self.index_missing_started_at is None:
                    self.index_missing_started_at = now
            else:
                self.index_missing_started_at = None

            lost_tracking = (
                self.tracking_missing_started_at is not None
                and (now - self.tracking_missing_started_at) >= ACTIVE_TRACKING_LOSS_SECONDS
            )
            lowered_index = (
                self.index_missing_started_at is not None
                and (now - self.index_missing_started_at) >= ACTIVE_INDEX_LOSS_GRACE_SECONDS
            )

            if lost_tracking or lowered_index:
                return self._exit_active()

        return events
