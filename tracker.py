#!/usr/bin/env python3
"""
YouTube View Count Tracker & Projector

Polls a YouTube video's view count every 5 minutes,
logs data to CSV, and projects the view count by a target date
using exponential decay modeled from historical video performance.

Usage:
    python tracker.py <VIDEO_URL_OR_ID>

Environment:
    YOUTUBE_API_KEY - Your YouTube Data API v3 key
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing 'requests' package. Install with: pip install requests")

try:
    import yaml
except ImportError:
    yaml = None


# ── Config ──────────────────────────────────────────────────────────────────

POLL_INTERVAL = 5 * 60  # 5 minutes in seconds
API_URL = "https://www.googleapis.com/youtube/v3/videos"
CSV_FILE = "view_log.csv"
REPORT_FILE = "BBHT Forecast.html"
CONFIG_FILE = "config.yaml"
DECAY_SCENARIOS = [0.82, 0.60, 0.40]  # Daily retention rates for projection scenarios


def load_config() -> dict:
    """Load settings from config.yaml if it exists."""
    config_path = Path(__file__).parent / CONFIG_FILE
    if not config_path.exists():
        return {}
    if yaml is None:
        print("Warning: pyyaml not installed, skipping config.yaml. Install with: pip install pyyaml")
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}

# ── Helpers ─────────────────────────────────────────────────────────────────


def extract_video_id(url_or_id: str) -> str:
    """Extract the video ID from a YouTube URL or return as-is if already an ID."""
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    sys.exit(f"Could not extract video ID from: {url_or_id}")


def fetch_video_info(video_id: str, api_key: str) -> tuple[int, str, datetime]:
    """Fetch current view count, title, and publish date from YouTube API."""
    params = {
        "part": "statistics,snippet",
        "id": video_id,
        "key": api_key,
    }
    resp = requests.get(API_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    items = data.get("items", [])
    if not items:
        sys.exit(f"Video not found: {video_id}")

    snippet = items[0]["snippet"]
    title = snippet["title"]
    published = datetime.fromisoformat(snippet["publishedAt"].replace("Z", "+00:00"))
    views = int(items[0]["statistics"]["viewCount"])
    return views, title, published


def estimate_current_views_per_day(csv_path: Path) -> float | None:
    """Estimate current daily view rate from the most recent polled data.

    Uses the last two data points to compute an instantaneous rate.
    Returns views per day, or None if < 2 data points.
    """
    timestamps: list[float] = []
    views: list[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            views.append(float(row["views"]))

    if len(timestamps) < 2:
        return None

    # Use last two points for most recent rate
    dt = timestamps[-1] - timestamps[-2]
    dv = views[-1] - views[-2]

    if dt <= 0:
        return None

    views_per_second = dv / dt
    return views_per_second * 86400  # convert to views per day


def compute_hourly_stats(csv_path: Path) -> dict:
    """Compute view rate stats over different time windows from CSV data.

    Returns dict with keys: views_last_hour, vph_12h, vph_24h, vph_overall.
    Values are None if insufficient data for that window.
    """
    timestamps: list[float] = []
    views: list[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            views.append(float(row["views"]))

    if len(timestamps) < 2:
        return {"views_last_hour": None, "vph_12h": None, "vph_24h": None, "vph_overall": None}

    now_ts = timestamps[-1]
    now_views = views[-1]

    def _find_closest(target_ts: float) -> tuple[float, float] | None:
        """Find the data point closest to target_ts that is <= target_ts."""
        best_i = None
        for i, ts in enumerate(timestamps):
            if ts <= target_ts:
                best_i = i
        if best_i is None:
            return None
        return timestamps[best_i], views[best_i]

    def _rate_for_window(hours: float) -> float | None:
        target_ts = now_ts - hours * 3600
        point = _find_closest(target_ts)
        if point is None:
            return None
        ts, v = point
        elapsed_hours = (now_ts - ts) / 3600
        if elapsed_hours < hours * 0.5:
            # Not enough data spanning this window
            return None
        if elapsed_hours <= 0:
            return None
        return (now_views - v) / elapsed_hours

    # Views gained in the last hour (absolute, not rate)
    views_last_hour = None
    point_1h = _find_closest(now_ts - 3600)
    if point_1h is not None:
        ts_1h, v_1h = point_1h
        elapsed = (now_ts - ts_1h) / 3600
        if elapsed > 0:
            # Scale to a full hour
            views_last_hour = int((now_views - v_1h) / elapsed)

    # Overall views per hour since first data point
    vph_overall = None
    total_hours = (now_ts - timestamps[0]) / 3600
    if total_hours > 0:
        vph_overall = (now_views - views[0]) / total_hours

    return {
        "views_last_hour": views_last_hour,
        "vph_12h": _rate_for_window(12),
        "vph_24h": _rate_for_window(24),
        "vph_overall": vph_overall,
    }


def compute_observed_decay(csv_path: Path) -> dict:
    """Compute observed hourly decay rates from the CSV data.

    Buckets data into 1-hour intervals, computes the ratio of views gained
    in consecutive hours, averages those ratios, and converts to a daily
    equivalent (hourly_ratio^24).

    Returns dict with keys: decay_1h, decay_6h, decay_12h (daily equivalents),
    or None if insufficient data for a window.
    """
    timestamps: list[float] = []
    views: list[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            views.append(float(row["views"]))

    if len(timestamps) < 3:
        return {"decay_1h": None, "decay_6h": None, "decay_12h": None}

    now_ts = timestamps[-1]
    first_ts = timestamps[0]

    # Build hourly buckets: views gained in each 1-hour window
    # Bucket boundaries work backwards from now_ts
    hourly_gains = []
    bucket_end = now_ts
    while bucket_end - 3600 >= first_ts:
        bucket_start = bucket_end - 3600
        # Find views closest to bucket_start and bucket_end
        v_start = None
        v_end = None
        for i, ts in enumerate(timestamps):
            if ts <= bucket_start:
                v_start = views[i]
            if ts <= bucket_end:
                v_end = views[i]
        if v_start is not None and v_end is not None:
            gain = v_end - v_start
            if gain > 0:
                hourly_gains.insert(0, gain)  # oldest first
            else:
                break  # stop if we hit non-positive gains
        bucket_end = bucket_start

    if len(hourly_gains) < 2:
        return {"decay_1h": None, "decay_6h": None, "decay_12h": None}

    # Compute hourly ratios between consecutive buckets
    hourly_ratios = []
    for i in range(1, len(hourly_gains)):
        if hourly_gains[i - 1] > 0:
            hourly_ratios.append(hourly_gains[i] / hourly_gains[i - 1])

    if not hourly_ratios:
        return {"decay_1h": None, "decay_6h": None, "decay_12h": None}

    def _avg_decay_daily(ratios: list[float]) -> float | None:
        if not ratios:
            return None
        avg_hourly = sum(ratios) / len(ratios)
        return avg_hourly ** 24  # convert hourly retention to daily

    # Most recent ratio (last 1h pair)
    decay_1h = _avg_decay_daily(hourly_ratios[-1:])

    # Last 6 hours worth of ratios
    decay_6h = _avg_decay_daily(hourly_ratios[-6:]) if len(hourly_ratios) >= 1 else None

    # Last 12 hours worth of ratios
    decay_12h = _avg_decay_daily(hourly_ratios[-12:]) if len(hourly_ratios) >= 1 else None

    return {
        "decay_1h": decay_1h,
        "decay_6h": decay_6h,
        "decay_12h": decay_12h,
    }


def project_views_decay(
    current_views: int,
    current_daily_rate: float,
    days_since_publish: float,
    target: datetime,
    publish_date: datetime,
    decay_rate: float,
) -> tuple[int, float]:
    """Project total views by target date using exponential decay.

    The model assumes daily views decay by a constant factor each day:
        views_on_day(d) = views_on_day(current) * decay_rate^(d - current_day)

    Returns (projected_total_views, current_views_per_hour).
    """
    now = datetime.now(timezone.utc)
    days_remaining = (target - now).total_seconds() / 86400

    if days_remaining <= 0:
        return current_views, current_daily_rate / 24

    # Sum projected daily views for each future day
    # Day 0 = rest of today (fractional), then full days after
    total_new_views = 0.0

    # Fractional first day
    fraction_of_today = 1.0 - (days_remaining - int(days_remaining))
    if fraction_of_today < 1.0:
        # partial day remaining today
        pass

    # Sum over each future day
    for day_offset in range(int(math.ceil(days_remaining))):
        daily_views = current_daily_rate * (decay_rate ** day_offset)
        # For the last partial day, scale proportionally
        if day_offset == int(math.ceil(days_remaining)) - 1:
            fractional = days_remaining - int(days_remaining)
            if fractional > 0:
                daily_views *= fractional
        total_new_views += daily_views

    projected_total = current_views + max(0, int(total_new_views))
    views_per_hour = current_daily_rate / 24

    return projected_total, views_per_hour


def generate_projection_series(
    current_daily_rate: float,
    decay_rate: float,
    now: datetime,
    target: datetime,
) -> list[dict]:
    """Generate day-by-day projected daily views from now to target date."""
    points = []
    days_remaining = (target - now).total_seconds() / 86400
    for day_offset in range(int(math.ceil(days_remaining)) + 1):
        day_date = now + timedelta(days=day_offset)
        daily_views = current_daily_rate * (decay_rate ** day_offset)
        points.append({
            "date": day_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "views": int(daily_views),
        })
    return points


def generate_cumulative_projection(
    current_views: int,
    current_daily_rate: float,
    decay_rate: float,
    now: datetime,
    target: datetime,
) -> list[dict]:
    """Generate cumulative projected total views from now to target date."""
    points = [{"date": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "views": current_views}]
    cumulative = current_views
    days_remaining = (target - now).total_seconds() / 86400
    for day_offset in range(1, int(math.ceil(days_remaining)) + 1):
        day_date = now + timedelta(days=day_offset)
        daily_views = current_daily_rate * (decay_rate ** (day_offset - 1))
        if day_offset == int(math.ceil(days_remaining)):
            fractional = days_remaining - int(days_remaining)
            if fractional > 0:
                daily_views *= fractional
        cumulative += daily_views
        points.append({
            "date": day_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "views": int(cumulative),
        })
    return points


def generate_html_report(
    title: str,
    video_id: str,
    current_views: int,
    publish_date: datetime,
    target: datetime,
    decay_scenarios: list[float],
    current_daily_rate: float | None,
    scenario_projections: list[tuple[float, int]],
    hourly_stats: dict,
    observed_decay: dict,
    csv_path: Path,
    report_path: Path,
):
    """Generate an HTML report with chart and stats.

    scenario_projections is a list of (decay_rate, projected_total) tuples.
    """
    now = datetime.now(timezone.utc)
    days_since_publish = (now - publish_date).total_seconds() / 86400
    days_remaining = (target - now).total_seconds() / 86400

    # Read historical data from CSV
    historical = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            historical.append({
                "date": datetime.fromtimestamp(float(row["timestamp"]), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "views": int(row["views"]),
            })

    # Generate projection series for each scenario
    all_cumulative = {}
    all_daily = {}
    if current_daily_rate and current_daily_rate > 0 and days_remaining > 0:
        for rate in decay_scenarios:
            all_cumulative[rate] = generate_cumulative_projection(
                current_views, current_daily_rate, rate, now, target,
            )
            all_daily[rate] = generate_projection_series(
                current_daily_rate, rate, now, target,
            )

    # Build scenario JSON outside the f-string to avoid brace escaping issues
    scenarios_dict = {
        str(r): {"cumulative": all_cumulative.get(r, []), "daily": all_daily.get(r, [])}
        for r in decay_scenarios
    }
    scenarios_json = json.dumps(scenarios_dict)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="300">
<title>BBHT Forecast</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f0f; color: #e1e1e1; padding: 24px; }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 20px; font-weight: 600; margin-bottom: 4px; }}
  .subtitle {{ color: #888; font-size: 13px; margin-bottom: 24px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .stat {{ background: #1a1a1a; border-radius: 12px; padding: 20px; }}
  .stat-label {{ font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
  .stat-value {{ font-size: 28px; font-weight: 700; }}
  .stat-value.accent {{ color: #f0b90b; }}
  .stat-value.green {{ color: #00c853; }}
  .chart-container {{ background: #1a1a1a; border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
  .chart-title {{ font-size: 14px; font-weight: 600; margin-bottom: 16px; }}
  .updated {{ color: #555; font-size: 11px; text-align: center; margin-top: 16px; }}
</style>
</head>
<body>
<div class="container">
  <h1>BBHT Forecast</h1>
  <div class="subtitle">Last updated: {now.strftime('%b %d, %Y %H:%M UTC')}</div>
  <div class="subtitle">Published {publish_date.strftime('%b %d, %Y')} &middot; Day {days_since_publish:.1f} &middot; Target: {target.strftime('%b %d, %Y')}</div>

  <div class="stats">
    <div class="stat">
      <div class="stat-label">Current Views</div>
      <div class="stat-value">{current_views:,}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Views Last Hour</div>
      <div class="stat-value">{_fmt_stat(hourly_stats.get('views_last_hour'))}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Views/Hr (12h avg)</div>
      <div class="stat-value">{_fmt_stat(hourly_stats.get('vph_12h'))}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Views/Hr (24h avg)</div>
      <div class="stat-value">{_fmt_stat(hourly_stats.get('vph_24h'))}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Views/Hr (Overall)</div>
      <div class="stat-value">{_fmt_stat(hourly_stats.get('vph_overall'))}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Days Remaining</div>
      <div class="stat-value">{days_remaining:.1f}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Observed Decay (1h)</div>
      <div class="stat-value">{_fmt_decay(observed_decay.get('decay_1h'))}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Observed Decay (6h)</div>
      <div class="stat-value">{_fmt_decay(observed_decay.get('decay_6h'))}</div>
    </div>
    <div class="stat">
      <div class="stat-label">Observed Decay (12h)</div>
      <div class="stat-value">{_fmt_decay(observed_decay.get('decay_12h'))}</div>
    </div>
{''.join(f"""    <div class="stat">
      <div class="stat-label">Projected ({rate}) by {target.strftime('%b %d')}</div>
      <div class="stat-value accent">{proj:,} <span style="font-size:14px;color:#00c853">(+{proj - current_views:,})</span></div>
    </div>""" for rate, proj in scenario_projections)}
  </div>

  <div class="chart-container">
    <div class="chart-title">Total Views — Actual + Projected Scenarios</div>
    <canvas id="cumulativeChart"></canvas>
  </div>

  <div class="chart-container">
    <div class="chart-title">Projected Daily Views — Decay Scenarios</div>
    <canvas id="dailyChart"></canvas>
  </div>

  <div class="updated">Last updated: {now.strftime('%Y-%m-%d %H:%M:%S UTC')} &middot; Auto-refreshes every 5 min</div>
</div>

<script>
const historical = {json.dumps(historical)};
const scenarios = {scenarios_json};
const scenarioColors = ['#f0b90b', '#00c853', '#e040fb'];

Chart.defaults.color = '#888';
Chart.defaults.borderColor = '#333';

const scenarioKeys = Object.keys(scenarios);

// Plugin to draw end-of-line labels on the cumulative chart
const endLabelPlugin = {{
  id: 'endLabels',
  afterDatasetsDraw(chart) {{
    const ctx = chart.ctx;
    chart.data.datasets.forEach((dataset, i) => {{
      const meta = chart.getDatasetMeta(i);
      if (!meta.visible || meta.data.length === 0) return;
      const last = meta.data[meta.data.length - 1];
      if (!last) return;
      const value = dataset.data[dataset.data.length - 1]?.y;
      if (value == null) return;
      const label = (value / 1000000).toFixed(1) + 'M';
      ctx.save();
      ctx.font = 'bold 11px -apple-system, sans-serif';
      ctx.fillStyle = dataset.borderColor;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, last.x + 6, last.y);
      ctx.restore();
    }});
  }},
}};

// Cumulative chart
new Chart(document.getElementById('cumulativeChart'), {{
  type: 'line',
  plugins: [endLabelPlugin],
  data: {{
    datasets: [
      {{
        label: 'Actual Views',
        data: historical.map(p => ({{ x: p.date, y: p.views }})),
        borderColor: '#e1e1e1',
        backgroundColor: 'rgba(225,225,225,0.1)',
        borderWidth: 2,
        pointRadius: historical.length > 50 ? 0 : 2,
        fill: true,
      }},
      ...scenarioKeys.map((key, i) => ({{
        label: 'Decay ' + key,
        data: scenarios[key].cumulative.map(p => ({{ x: p.date, y: p.views }})),
        borderColor: scenarioColors[i % scenarioColors.length],
        borderDash: [6, 3],
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
      }})),
    ],
  }},
  options: {{
    responsive: true,
    layout: {{ padding: {{ right: 50 }} }},
    interaction: {{ intersect: false, mode: 'index' }},
    scales: {{
      x: {{ type: 'time', time: {{ unit: 'day' }}, grid: {{ color: '#222' }} }},
      y: {{ grid: {{ color: '#222' }}, ticks: {{ callback: v => (v/1000000).toFixed(1) + 'M' }} }},
    }},
    plugins: {{ legend: {{ labels: {{ usePointStyle: true }} }} }},
  }},
}});

// Daily projected chart
new Chart(document.getElementById('dailyChart'), {{
  type: 'line',
  data: {{
    datasets: scenarioKeys.map((key, i) => ({{
      label: 'Decay ' + key,
      data: scenarios[key].daily.map(p => ({{ x: p.date, y: p.views }})),
      borderColor: scenarioColors[i % scenarioColors.length],
      backgroundColor: scenarioColors[i % scenarioColors.length] + '33',
      borderWidth: 2,
      pointRadius: 0,
      fill: true,
    }})),
  }},
  options: {{
    responsive: true,
    scales: {{
      x: {{ type: 'time', time: {{ unit: 'day' }}, grid: {{ color: '#222' }} }},
      y: {{ grid: {{ color: '#222' }}, ticks: {{ callback: v => (v/1000000).toFixed(1) + 'M' }} }},
    }},
    plugins: {{ legend: {{ labels: {{ usePointStyle: true }} }} }},
  }},
}});
</script>
</body>
</html>"""

    with open(report_path, "w") as f:
        f.write(html)


def _fmt_stat(value: float | None) -> str:
    """Format a stat value with commas, or show '—' if unavailable."""
    if value is None:
        return "—"
    return f"{int(value):,}"


def _fmt_decay(value: float | None) -> str:
    """Format a decay rate value, or show '—' if unavailable."""
    if value is None:
        return "—"
    return f"{value:.3f}"


def git_push_report(report_path: Path, csv_path: Path):
    """Auto-commit and push the report and CSV to GitHub for Pages hosting."""
    repo_dir = report_path.parent
    try:
        subprocess.run(
            ["git", "add", str(report_path.name), str(csv_path.name)],
            cwd=repo_dir, capture_output=True, timeout=10,
        )
        result = subprocess.run(
            ["git", "commit", "-m", f"Update report {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"],
            cwd=repo_dir, capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            subprocess.run(
                ["git", "push"],
                cwd=repo_dir, capture_output=True, timeout=30,
            )
            print("  Pushed to GitHub Pages")
        else:
            # Nothing to commit (no changes)
            pass
    except Exception as e:
        print(f"  Git push failed: {e}")


def format_number(n: int) -> str:
    """Format a number with commas."""
    return f"{n:,}"


def time_until(target: datetime) -> str:
    """Human-readable time remaining until target."""
    delta = target - datetime.now(timezone.utc)
    if delta.total_seconds() <= 0:
        return "already passed"
    days = delta.days
    hours, rem = divmod(delta.seconds, 3600)
    minutes = rem // 60
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    return " ".join(parts) if parts else "<1m"


# ── Main loop ───────────────────────────────────────────────────────────────


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Track YouTube video view counts")
    parser.add_argument("video", nargs="?", default=None, help="YouTube video URL or ID")
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Poll interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--target-date",
        default=None,
        help="Target date for projection in YYYY-MM-DD (default: 2026-02-15)",
    )
    args = parser.parse_args()

    # Resolve video: CLI arg > config file
    video = args.video or config.get("video")
    if not video:
        sys.exit("Provide a video URL/ID as an argument or set 'video' in config.yaml")

    # Resolve API key: env var > config file
    api_key = os.environ.get("YOUTUBE_API_KEY") or config.get("youtube_api_key")
    if not api_key:
        sys.exit(
            "Set youtube_api_key in config.yaml or the YOUTUBE_API_KEY environment variable.\n"
            "Get one at: https://console.cloud.google.com/apis/credentials"
        )

    # Resolve optional settings: CLI arg > config file > default
    args.interval = args.interval or config.get("poll_interval", POLL_INTERVAL)
    args.target_date = args.target_date or config.get("target_date", "2026-02-15")
    args.video = video
    decay_scenarios = config.get("decay_scenarios", DECAY_SCENARIOS)

    video_id = extract_video_id(args.video)
    target = datetime.strptime(args.target_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    csv_path = Path(__file__).parent / CSV_FILE
    report_path = Path(__file__).parent / REPORT_FILE
    csv_exists = csv_path.exists()

    # Ensure CSV has a header
    if not csv_exists:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "datetime", "views"])

    print(f"Tracking video: {video_id}")
    print(f"Target date:    {args.target_date}")
    print(f"Decay scenarios: {decay_scenarios}")
    print(f"Poll interval:  {args.interval}s")
    print(f"Log file:       {csv_path}")
    print("-" * 60)

    title = None
    publish_date = None
    check_num = 0

    while True:
        try:
            views, title_fetched, pub_date = fetch_video_info(video_id, api_key)
            if title is None:
                title = title_fetched
                publish_date = pub_date
                print(f"Video title:    {title}")
                print(f"Published:      {publish_date.strftime('%Y-%m-%d %H:%M UTC')}")
                print("-" * 60)

            now = datetime.now(timezone.utc)
            ts = now.timestamp()
            check_num += 1
            days_since_publish = (now - publish_date).total_seconds() / 86400

            # Log to CSV
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ts, now.isoformat(), views])

            # Display
            print(f"\n[#{check_num}] {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"  Current views: {format_number(views)}")
            print(f"  Day {days_since_publish:.1f} since publish")
            print(f"  Time until {args.target_date}: {time_until(target)}")

            # Project with decay model — run all scenarios
            daily_rate = estimate_current_views_per_day(csv_path)
            scenario_projections = []
            if daily_rate is not None and daily_rate > 0:
                print(f"  Views/day:     {format_number(int(daily_rate))}")
                for rate in decay_scenarios:
                    proj, vph = project_views_decay(
                        current_views=views,
                        current_daily_rate=daily_rate,
                        days_since_publish=days_since_publish,
                        target=target,
                        publish_date=publish_date,
                        decay_rate=rate,
                    )
                    scenario_projections.append((rate, proj))
                    gain = proj - views
                    print(f"  Decay {rate}: {format_number(proj)} (+{format_number(gain)})")
            else:
                print("  (Need at least 2 data points for projection)")

            # Generate HTML report
            hourly_stats = compute_hourly_stats(csv_path)
            observed_decay = compute_observed_decay(csv_path)
            if daily_rate is not None and daily_rate > 0 and scenario_projections:
                generate_html_report(
                    title=title,
                    video_id=video_id,
                    current_views=views,
                    publish_date=publish_date,
                    target=target,
                    decay_scenarios=decay_scenarios,
                    current_daily_rate=daily_rate,
                    scenario_projections=scenario_projections,
                    hourly_stats=hourly_stats,
                    observed_decay=observed_decay,
                    csv_path=csv_path,
                    report_path=report_path,
                )
                print(f"  Report:        {report_path}")
                git_push_report(report_path, csv_path)

        except requests.exceptions.RequestException as e:
            print(f"\n  API error: {e} — will retry next cycle")
        except KeyboardInterrupt:
            print("\n\nStopped by user.")
            if csv_path.exists():
                print(f"Data saved to: {csv_path}")
            break

        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\nStopped by user.")
            if csv_path.exists():
                print(f"Data saved to: {csv_path}")
            break


if __name__ == "__main__":
    main()
