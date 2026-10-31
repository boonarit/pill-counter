# pill-counter

Counting pills in phone photos with classical image processing (OpenCV), and measuring how wrong each method gets
as pills touch more and as the background color gets closer to the pill color. Course project for Computer Vision.

Three methods, one pipeline (color channel → blur → threshold → morphology → size filter):

| Method | Idea |
|---|---|
| CC | connected components (8-connectivity) |
| Area | Σ round(blob area / one-pill area); one-pill area = median watershed region |
| Watershed | distance transform → per-blob peaks (≥ 0.60 × blob max) → `cv2.watershed` |

## Parts

- `web/index.html` — the lab app (v1). Single HTML file, vanilla JS, runs fully in the browser, shows every step,
  makes synthetic test images with a known count, logs results to CSV.
- `pillcount.py` — the same pipeline and defaults in Python + `cv2`, for batch runs on a folder of photos.
- `tests/` — 7 synthetic images exported from the app (seed 42) with the app's counts.

## Batch run (M1)

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python pillcount.py images/ images/truth.csv -o results.csv
.venv/bin/python pillcount.py images/ images/truth.csv -o results-sat.csv --chan s
.venv/bin/python -m unittest
```

`truth.csv`: `file,truth,density,background` (people-counted truth; density/background group the MAPE table).
iPhone photos must be JPG/PNG (Settings → Camera → Formats → Most Compatible), not HEIC.
`images/` and `results*.csv` are git-ignored.

### Python vs app

On the synthetic set, CC / Area / Watershed counts match the app exactly in 5 of 7 cases. The two that differ are the
"background close to pill color" cases, where the threshold is fragile and small differences in blur rounding and
image-edge handling change the mask. `cv2.watershed` floods by image color difference, while the app floods by
distance, so the watershed count is the same but region areas (the Area method's reference) can differ.
Report synthetic and real-photo results separately.

## Deploy (M4, Tailscale only)

The web app is static, so production is `python3 -m http.server` on `127.0.0.1:8792` under launchd,
published to the tailnet with HTTPS (needed for the iPhone camera) by `tailscale serve` on port 8445.
Code reaches the M4 by `git push m4 main` from the M1 (the M4 checkout uses `receive.denyCurrentBranch=updateInstead`),
so the M4 needs no GitHub credentials.

One-time setup on the M4:

```sh
git init ~/services/pill-counter && git -C ~/services/pill-counter config receive.denyCurrentBranch updateInstead
# from the M1: git remote add m4 <user>@<m4-tailscale-ip>:services/pill-counter && git push m4 main
sed "s#__HOME__#$HOME#g" ~/services/pill-counter/deploy/com.pillcounter.web.plist > ~/Library/LaunchAgents/com.pillcounter.web.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pillcounter.web.plist
/Applications/Tailscale.app/Contents/MacOS/Tailscale serve --bg --https=8445 http://127.0.0.1:8792
```

Update: `git push m4 main` (static files are served as soon as they land).
Open: `https://<m4-name>.<tailnet>.ts.net:8445/` from any device on the tailnet.
