import pytest

from racecar.track import Track
from racecar.track_gen import SMALL_LIMITS, corridor_image


@pytest.fixture
def corridor_track(tmp_path) -> Track:
    """Straight corridor: track x in [25, 125), y in [10, 30); start patch x in [20, 25)."""
    path = tmp_path / "corridor.bmp"
    corridor_image().save(path)
    return Track.load(path, **SMALL_LIMITS)


@pytest.fixture(scope="session")
def tk_master():
    """One hidden Tk root for the whole session; windows are Toplevels of it.

    Creating many Tk roots in one process occasionally fails on Windows with
    "Can't find a usable init.tcl", so tests share one.
    """
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display available: {e}")
    root.withdraw()
    yield root
    root.destroy()
