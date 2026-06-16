# Companion eyes visual cue

The companion eyes are a tiny Tkinter overlay for seeing what MoK is paying attention to without tailing logs.

The overlay reads this local JSON file:

```text
~/.mok/companion/attention.json
```

The companion watch loop updates that file after each tick. When MoK sees an active window, the eyes bias toward a window target. When it is scanning allowlisted files, the eyes bias down-left. When no fresh update arrives, the eyes relax into a quiet/stale state.

## Run the overlay

```powershell
python -m mok.companion.eyes
```

Keep the eyes above other windows:

```powershell
python -m mok.companion.eyes --topmost
```

Borderless testing window:

```powershell
python -m mok.companion.eyes --topmost --frameless
```

Demo animation without the watcher:

```powershell
python -m mok.companion.eyes --demo --topmost
```

The overlay can be dragged with the mouse. Press `T` to toggle always-on-top and `Esc` to close it.

## Feed real activity

Run the normal companion watcher in a separate terminal or through the lifecycle commands:

```powershell
python run_mok.py companion watch
```

or:

```powershell
python run_mok.py companion wakeup
```

The watcher publishes attention state automatically.

Example state:

```json
{
  "state": "watching",
  "target_kind": "window",
  "target_label": "Visual Studio Code - service.py",
  "gaze_x": 0.34,
  "gaze_y": -0.11,
  "detail": {
    "scan": {
      "scanned": 42,
      "stored": 42,
      "skipped": 0,
      "errors": []
    },
    "window": "Visual Studio Code - service.py"
  },
  "updated_at": "2026-06-16T05:00:00Z"
}
```

## Design notes

The first version is deliberately approximate. The watcher currently knows the active window title and whether files were scanned; it does not know real screen coordinates yet. If a future observer can provide coordinates, it can write explicit `gaze_x` and `gaze_y` values between `-1.0` and `1.0`, and the overlay will use those directly.
