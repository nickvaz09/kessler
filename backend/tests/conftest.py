"""Suite-wide defaults: no live NOAA space-weather fetches (tests that need
the fetch path switch it back on with monkeypatch and fake the feeds)."""
import os

os.environ.setdefault("KESSLER_SPACEWEATHER", "0")
