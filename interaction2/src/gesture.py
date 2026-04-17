from dataclasses import dataclass, field

from .eeg_profile import EmotionProfile


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def move_toward_target(current: float, target: float, rate: float) -> float:
    return current + (target - current) * rate


@dataclass
class OpenState:
    phase: str = "idle"
    hold_elapsed: float = 0.0
    amount: float = 0.0
    previous_amount: float = 0.0
    hand_branch: str = "missing"
    active_time: float = 0.0


@dataclass
class RiseState:
    phase: str = "idle"
    hold_elapsed: float = 0.0
    amount: float = 0.0
    previous_amount: float = 0.0


@dataclass
class GatherState:
    phase: str = "idle"
    amount: float = 0.0
    active_time: float = 0.0


@dataclass
class DeepBreathState:
    phase: str = "idle"
    amount: float = 0.0
    active_time: float = 0.0


@dataclass
class InteractionState:
    open: OpenState = field(default_factory=OpenState)
    rise: RiseState = field(default_factory=RiseState)
    gather: GatherState = field(default_factory=GatherState)
    deep_breath: DeepBreathState = field(default_factory=DeepBreathState)


@dataclass
class EmotionState:
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0


@dataclass
class InteractionSignals:
    dt: float
    open_ready: bool = False
    open_branch: str = "missing"
    open_spread: float = 0.0
    open_release: bool = False
    rise_ready: bool = False
    rise_height: float = 0.0
    rise_release: bool = False
    gather_ready: bool = False
    gather_release: bool = False
    breath_ready: bool = False
    breath_release: bool = False


@dataclass
class EngineEvents:
    triggered: list[str] = field(default_factory=list)

    def fire(self, event: str) -> None:
        self.triggered.append(event)


@dataclass
class InteractionEngine:
    interaction: InteractionState = field(default_factory=InteractionState)
    emotion: EmotionState = field(default_factory=EmotionState)
    profile: EmotionProfile = field(default_factory=EmotionProfile)
    open_hold_duration: float = 1.0
    rise_hold_duration: float = 0.5
    stable_valence: float = 0.6
    stable_arousal: float = 0.35
    stable_dominance: float = 0.5
    open_v_rate: float = 0.45
    open_d_rate: float = 0.35
    rise_a_rate: float = 0.55
    rise_v_rate: float = 0.12
    rise_d_rate: float = 0.12
    open_v_max: float = 1.0
    open_d_max: float = 1.0
    rise_a_max: float = 1.0
    rise_v_max: float = 0.4
    rise_d_max: float = 0.4

    def reset(self) -> None:
        self.interaction = InteractionState()
        self.emotion = EmotionState(
            valence=self.profile.baseline_v,
            arousal=self.profile.baseline_a,
            dominance=self.profile.baseline_d,
        )

    def set_profile(self, profile: EmotionProfile) -> None:
        self.profile = profile
        self.reset()

    def update(self, signals: InteractionSignals) -> EngineEvents:
        events = EngineEvents()
        self._update_open(signals, events)
        self._update_rise(signals, events)
        self._update_gather(signals, events)
        self._update_deep_breath(signals, events)
        self._update_emotion(signals.dt)
        self._clamp_emotion()
        return events

    @property
    def is_interacting(self) -> bool:
        return any(
            (
                self.interaction.open.phase == "active",
                self.interaction.rise.phase == "active",
                self.interaction.gather.phase == "active",
                self.interaction.deep_breath.phase == "active",
            )
        )

    def _update_open(self, signals: InteractionSignals, events: EngineEvents) -> None:
        state = self.interaction.open

        if state.phase == "idle":
            state.previous_amount = state.amount
            state.amount = 0.0
            state.hand_branch = "missing"
            state.active_time = 0.0
            if signals.open_ready:
                state.phase = "arming"
                state.hold_elapsed = signals.dt
            else:
                state.hold_elapsed = 0.0
            return

        if state.phase == "arming":
            if not signals.open_ready:
                state.phase = "idle"
                state.hold_elapsed = 0.0
                state.hand_branch = "missing"
                return

            state.hold_elapsed += signals.dt
            state.hand_branch = signals.open_branch
            if state.hold_elapsed >= self.open_hold_duration:
                if signals.open_branch == "open":
                    state.phase = "active"
                    state.active_time = 0.0
                    events.fire("open_started")
                elif signals.open_branch == "gather":
                    state.phase = "idle"
                    events.fire("open_redirected_to_gather")
                else:
                    state.phase = "idle"
                    events.fire("open_missing")
                state.hold_elapsed = 0.0
            return

        if state.phase == "active":
            state.active_time += signals.dt
            state.previous_amount = state.amount
            state.amount = clamp(signals.open_spread, -1.0, 1.0)
            if signals.open_release:
                state.phase = "idle"
                state.previous_amount = state.amount
                state.amount = 0.0
                state.active_time = 0.0
                events.fire("open_ended")

    def _update_rise(self, signals: InteractionSignals, events: EngineEvents) -> None:
        state = self.interaction.rise

        if state.phase == "idle":
            state.previous_amount = state.amount
            state.amount = 0.0
            if signals.rise_ready:
                state.phase = "arming"
                state.hold_elapsed = signals.dt
            else:
                state.hold_elapsed = 0.0
            return

        if state.phase == "arming":
            if not signals.rise_ready:
                state.phase = "idle"
                state.hold_elapsed = 0.0
                return

            state.hold_elapsed += signals.dt
            if state.hold_elapsed >= self.rise_hold_duration:
                state.phase = "active"
                state.hold_elapsed = 0.0
                events.fire("rise_started")
            return

        if state.phase == "active":
            state.previous_amount = state.amount
            state.amount = clamp(signals.rise_height, -1.0, 1.0)
            if signals.rise_release:
                state.phase = "idle"
                state.previous_amount = state.amount
                state.amount = 0.0
                events.fire("rise_ended")

    def _update_gather(self, signals: InteractionSignals, events: EngineEvents) -> None:
        state = self.interaction.gather

        if state.phase == "idle":
            state.amount = 0.0
            state.active_time = 0.0
            if signals.gather_ready:
                state.phase = "active"
                state.amount = 1.0
                events.fire("gather_started")
            return

        if state.phase == "active":
            state.active_time += signals.dt
            state.amount = 1.0
            if signals.gather_release:
                state.phase = "idle"
                state.amount = 0.0
                state.active_time = 0.0
                events.fire("gather_ended")

    def _update_deep_breath(self, signals: InteractionSignals, events: EngineEvents) -> None:
        state = self.interaction.deep_breath

        if state.phase == "idle":
            state.amount = 0.0
            state.active_time = 0.0
            if signals.breath_ready:
                state.phase = "active"
                state.amount = 1.0
                events.fire("deep_breath_started")
            return

        if state.phase == "active":
            state.active_time += signals.dt
            state.amount = 1.0
            if signals.breath_release:
                state.phase = "idle"
                state.amount = 0.0
                state.active_time = 0.0
                events.fire("deep_breath_ended")

    def _update_emotion(self, dt: float) -> None:
        open_amount = self.interaction.open.amount
        rise_amount = self.interaction.rise.amount
        gather_active = self.interaction.gather.phase == "active"
        breath_active = self.interaction.deep_breath.phase == "active"

        # Open/rise are now interpreted as "current amount -> current target"
        # instead of accumulating forever over time. That means reducing amount
        # immediately pulls the related emotion axes back down.
        open_target_v = open_amount * self.open_v_max * self.profile.v_gain
        open_target_d = open_amount * self.open_d_max * self.profile.d_gain
        rise_target_a = rise_amount * self.rise_a_max * self.profile.a_gain
        rise_target_v = rise_amount * self.rise_v_max * self.profile.v_gain
        rise_target_d = rise_amount * self.rise_d_max * self.profile.d_gain

        target_v = clamp(
            self.profile.baseline_v + open_target_v + rise_target_v,
            self.profile.v_min,
            self.profile.v_max,
        )
        target_a = clamp(
            self.profile.baseline_a + rise_target_a,
            self.profile.a_min,
            self.profile.a_max,
        )
        target_d = clamp(
            self.profile.baseline_d + open_target_d + rise_target_d,
            self.profile.d_min,
            self.profile.d_max,
        )

        self.emotion.valence = move_toward_target(
            self.emotion.valence, target_v, self.profile.v_return_rate * dt
        )
        self.emotion.arousal = move_toward_target(
            self.emotion.arousal, target_a, self.profile.a_return_rate * dt
        )
        self.emotion.dominance = move_toward_target(
            self.emotion.dominance, target_d, self.profile.d_return_rate * dt
        )

        if gather_active:
            self.emotion.valence = move_toward_target(
                self.emotion.valence, self.profile.baseline_v, self.profile.v_return_rate * dt
            )
            self.emotion.dominance = move_toward_target(
                self.emotion.dominance, self.profile.baseline_d, self.profile.d_return_rate * dt
            )

        if breath_active:
            self.emotion.arousal = move_toward_target(
                self.emotion.arousal, self.profile.baseline_a, self.profile.a_return_rate * dt
            )

    def _clamp_emotion(self) -> None:
        self.emotion.valence = clamp(self.emotion.valence, self.profile.v_min, self.profile.v_max)
        self.emotion.arousal = clamp(self.emotion.arousal, self.profile.a_min, self.profile.a_max)
        self.emotion.dominance = clamp(self.emotion.dominance, self.profile.d_min, self.profile.d_max)
