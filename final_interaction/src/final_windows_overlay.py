from __future__ import annotations

import platform


def is_windows() -> bool:
    return platform.system() == "Windows"


def apply_click_through(hwnd: int, *, topmost: bool = True) -> bool:
    if not is_windows():
        print(f"[final_hud] Windows click-through unavailable on {platform.system()}; using Qt mouse transparency only.")
        return False

    try:
        import win32con
        import win32gui
    except ImportError:
        print("[final_hud] pywin32 is not installed; HUD click-through was not applied.")
        return False

    ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    ex_style |= win32con.WS_EX_LAYERED
    ex_style |= win32con.WS_EX_TRANSPARENT
    ex_style |= win32con.WS_EX_TOOLWINDOW
    noactivate = getattr(win32con, "WS_EX_NOACTIVATE", 0x08000000)
    ex_style |= noactivate
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style)

    if topmost:
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_TOPMOST,
            0,
            0,
            0,
            0,
            win32con.SWP_NOMOVE
            | win32con.SWP_NOSIZE
            | win32con.SWP_NOACTIVATE
            | win32con.SWP_SHOWWINDOW,
        )
    return True
