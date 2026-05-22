from __future__ import annotations

import math

try:
    from .final_hud_state import HudFrameState
    from .final_windows_overlay import apply_click_through
except ImportError:
    from final_hud_state import HudFrameState
    from final_windows_overlay import apply_click_through

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:  # pragma: no cover - exercised when HUD dependency is absent.
    QtCore = None
    QtGui = None
    QtWidgets = None


class FinalHudController:
    def __init__(
        self,
        *,
        enabled: bool,
        monitor_index: int = 0,
        click_through: bool = True,
        topmost: bool = True,
        opacity: float = 1.0,
        scale: float = 1.0,
    ) -> None:
        self.enabled = enabled
        self.monitor_index = monitor_index
        self.click_through = click_through
        self.topmost = topmost
        self.opacity = max(0.05, min(opacity, 1.0))
        self.scale = max(0.25, scale)
        self.app = None
        self.window = None

    def start(self) -> None:
        if not self.enabled:
            return
        if QtWidgets is None or QtGui is None or QtCore is None:
            print("[final_hud] PySide6 is not installed; HUD disabled.")
            self.enabled = False
            return

        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        screens = QtGui.QGuiApplication.screens()
        if not screens:
            print("[final_hud] no screens available; HUD disabled.")
            self.enabled = False
            return
        if 0 <= self.monitor_index < len(screens):
            screen = screens[self.monitor_index]
        else:
            screen = QtGui.QGuiApplication.primaryScreen()
            print(f"[final_hud] monitor index {self.monitor_index} is unavailable; using primary screen.")

        self.window = FinalHudWindow(opacity=self.opacity, scale=self.scale)
        flags = QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Tool
        if self.topmost:
            flags |= QtCore.Qt.WindowType.WindowStaysOnTopHint
        self.window.setWindowFlags(flags)
        self.window.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.window.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.window.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, self.click_through)
        self.window.setGeometry(screen.geometry())
        self.window.show()
        self.window.raise_()

        if self.click_through:
            apply_click_through(int(self.window.winId()), topmost=self.topmost)

    def update(self, state: HudFrameState) -> None:
        if self.enabled and self.window is not None:
            self.window.set_state(state)

    def process_events(self) -> None:
        if self.enabled and self.app is not None and QtCore is not None:
            self.app.processEvents()

    def close(self) -> None:
        if self.window is not None:
            self.window.close()
            self.window = None
        if self.app is not None and QtCore is not None:
            self.app.processEvents()


if QtWidgets is not None:
    class FinalHudWindow(QtWidgets.QWidget):
        def __init__(self, *, opacity: float, scale: float) -> None:
            super().__init__()
            self._state: HudFrameState | None = None
            self._opacity = opacity
            self._scale = scale
            self.setWindowTitle("Cosmos Interaction HUD")

        def set_state(self, state: HudFrameState) -> None:
            self._state = state
            self.update()

        def paintEvent(self, event) -> None:  # noqa: N802
            if self._state is None:
                return

            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.setOpacity(self._opacity)
            state = self._state
            width = self.width()
            height = self.height()
            scale = self._scale

            self._draw_world_pulse(painter, state, width, height, scale)
            self._draw_vad_bars(painter, state, width, height, scale)
            self._draw_mode_ring(painter, state, width, height, scale)
            self._draw_hand_orb(painter, state.left, width * 0.16, height * 0.58, scale, "L")
            self._draw_hand_orb(painter, state.right, width * 0.84, height * 0.58, scale, "R")
            self._draw_pointer(painter, state, width, height, scale)
            if state.switch_to_camera:
                self._draw_flash(painter, width, height)

        def _pen(self, color: QtGui.QColor, width: float) -> QtGui.QPen:
            pen = QtGui.QPen(color)
            pen.setWidthF(width)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            return pen

        def _draw_world_pulse(self, painter, state: HudFrameState, width: int, height: int, scale: float) -> None:
            cx = width * 0.5
            cy = height * 0.08
            pulse = 0.5 + 0.5 * math.sin(state.timestamp * 3.0)
            radius = (10.0 + pulse * 8.0) * scale
            color = QtGui.QColor(120, 240, 255, 170 if state.world_active else 65)
            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(self._pen(QtGui.QColor(210, 255, 255, 210 if state.world_active else 80), 1.5 * scale))
            painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)
            if state.world_number is not None:
                painter.setPen(self._pen(QtGui.QColor(230, 255, 255, 190), 1.0))
                painter.setFont(QtGui.QFont("Arial", max(9, int(11 * scale))))
                painter.drawText(QtCore.QRectF(cx - 40, cy + radius + 6, 80, 18), QtCore.Qt.AlignmentFlag.AlignCenter, str(state.world_number))

        def _draw_mode_ring(self, painter, state: HudFrameState, width: int, height: int, scale: float) -> None:
            cx = width * 0.5
            cy = height * 0.72
            grab = max(0.0, min(state.grab_strength, 1.0))
            open_strength = max(0.0, min(state.open_strength, 1.0))
            base_radius = (58.0 + open_strength * 48.0 - grab * 22.0) * scale
            color = QtGui.QColor(80, 230, 255, 150)
            if state.mode == "grab":
                color = QtGui.QColor(120, 255, 150, 185)
            elif not state.interaction_active:
                color = QtGui.QColor(180, 180, 180, 70)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.setPen(self._pen(color, (4.0 + grab * 7.0) * scale))
            painter.drawEllipse(QtCore.QPointF(cx, cy), base_radius, base_radius)
            painter.setPen(self._pen(QtGui.QColor(color.red(), color.green(), color.blue(), 65), 1.5 * scale))
            painter.drawEllipse(QtCore.QPointF(cx, cy), base_radius + 18 * scale, base_radius + 18 * scale)
            painter.setPen(self._pen(QtGui.QColor(235, 255, 255, 150), 1.0))
            painter.setFont(QtGui.QFont("Arial", max(10, int(13 * scale)), QtGui.QFont.Weight.Bold))
            painter.drawText(QtCore.QRectF(cx - 70, cy - 10, 140, 24), QtCore.Qt.AlignmentFlag.AlignCenter, state.mode.upper())

        def _draw_hand_orb(self, painter, hand, cx: float, cy: float, scale: float, label: str) -> None:
            radius = 30.0 * scale
            alpha = 160 if hand.visible else 45
            base = QtGui.QColor(220, 220, 230, alpha)
            if hand.grab_active:
                base = QtGui.QColor(120, 255, 150, alpha)
            elif hand.open_strength >= 0.62:
                base = QtGui.QColor(80, 230, 255, alpha)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(base.red(), base.green(), base.blue(), max(25, alpha // 3))))
            painter.setPen(self._pen(base, 2.2 * scale))
            painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

            rect = QtCore.QRectF(cx - radius - 7 * scale, cy - radius - 7 * scale, 2 * (radius + 7 * scale), 2 * (radius + 7 * scale))
            painter.setPen(self._pen(QtGui.QColor(80, 230, 255, alpha), 3.0 * scale))
            painter.drawArc(rect, 90 * 16, int(-360 * hand.open_strength * 16))
            painter.setPen(self._pen(QtGui.QColor(120, 255, 150, alpha), 5.0 * scale))
            painter.drawArc(rect.adjusted(-8 * scale, -8 * scale, 8 * scale, 8 * scale), 90 * 16, int(360 * hand.grab_strength * 16))

            painter.setPen(self._pen(QtGui.QColor(235, 255, 255, alpha), 1.0))
            painter.setFont(QtGui.QFont("Arial", max(9, int(12 * scale)), QtGui.QFont.Weight.Bold))
            painter.drawText(QtCore.QRectF(cx - 20, cy - 10, 40, 20), QtCore.Qt.AlignmentFlag.AlignCenter, label)

        def _draw_pointer(self, painter, state: HudFrameState, width: int, height: int, scale: float) -> None:
            if not state.interaction_active or state.pointer_x < 0.0 or state.pointer_y < 0.0:
                return
            cx = max(0.0, min(state.pointer_x, 1.0)) * width
            cy = max(0.0, min(state.pointer_y, 1.0)) * height
            color = QtGui.QColor(120, 255, 150, 165) if state.mode == "grab" else QtGui.QColor(80, 230, 255, 145)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(color.red(), color.green(), color.blue(), 55)))
            painter.setPen(self._pen(color, 2.0 * scale))
            painter.drawEllipse(QtCore.QPointF(cx, cy), 18 * scale, 18 * scale)
            painter.setBrush(QtGui.QBrush(color))
            painter.drawEllipse(QtCore.QPointF(cx, cy), 4 * scale, 4 * scale)

        def _draw_vad_bars(self, painter, state: HudFrameState, width: int, height: int, scale: float) -> None:
            x = width * 0.5 - 95 * scale
            y = height * 0.86
            for idx, (label, value, color) in enumerate((
                ("V", state.target_v, QtGui.QColor(255, 120, 190, 150)),
                ("A", state.target_a, QtGui.QColor(80, 220, 255, 150)),
                ("D", state.target_d, QtGui.QColor(130, 255, 130, 150)),
            )):
                top = y + idx * 20 * scale
                painter.setPen(self._pen(QtGui.QColor(220, 220, 220, 90), 1.0 * scale))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawRect(QtCore.QRectF(x, top, 190 * scale, 8 * scale))
                center = x + 95 * scale
                fill = 95 * scale * max(0.0, min(abs(value), 1.0))
                painter.setBrush(QtGui.QBrush(color))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                if value >= 0:
                    painter.drawRect(QtCore.QRectF(center, top + scale, fill, 6 * scale))
                else:
                    painter.drawRect(QtCore.QRectF(center - fill, top + scale, fill, 6 * scale))
                painter.setPen(self._pen(QtGui.QColor(235, 255, 255, 120), 1.0))
                painter.setFont(QtGui.QFont("Arial", max(8, int(10 * scale))))
                painter.drawText(QtCore.QRectF(x - 22 * scale, top - 5 * scale, 18 * scale, 18 * scale), QtCore.Qt.AlignmentFlag.AlignRight, label)

        def _draw_flash(self, painter, width: int, height: int) -> None:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 55)))
            painter.drawRect(QtCore.QRectF(0, 0, width, height))
else:
    class FinalHudWindow:  # type: ignore[no-redef]
        pass
