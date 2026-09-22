"""Render the three-page project guide and PNG previews using local artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pet_health_ai.inference import predict_sensor_dataframe
from pet_health_ai.ui_data import prepare_sensor_data

OUT = Path(__file__).resolve().parent
INK = "#132B40"
MUTED = "#526779"
TEAL = "#087F83"
BLUE = "#346ED1"
LIGHT = "#EDF5F7"
ORANGE = "#BA642D"
BG = "#FAFCFD"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "pdf.fonttype": 42, "axes.unicode_minus": False})


def text(ax, x, y, value, size=10, color=INK, weight="normal", **kwargs):
    return ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                   va="top", linespacing=1.45, **kwargs)


def box(ax, x, y, width, height, color=LIGHT, border=None):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                 boxstyle="round,pad=0.4,rounding_size=1", facecolor=color,
                 edgecolor=border or color, linewidth=0.8))


def arrow(ax, start, end, color=TEAL):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.35,
                            "mutation_scale": 11})


def page(number, title, subtitle):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    ax.add_patch(Rectangle((0, 98.8), 100, 1.2, color=TEAL))
    text(ax, 6, 95.4, "KTINOSKARE  /  MODEL & TESTING GUIDE", 9, TEAL, "bold")
    text(ax, 6, 91.4, title, 25, weight="bold")
    text(ax, 6, 86.4, subtitle, 10, MUTED)
    ax.plot([6, 94], [5.4, 5.4], color="#D9E3E8", lw=0.8)
    text(ax, 6, 4.1, "PROJECT v0.2  •  Repository snapshot: 20 Sep 2026", 7.5, MUTED)
    text(ax, 94, 4.1, f"{number} / 3", 8, TEAL, "bold", ha="right")
    return fig, ax


def save(pdf, fig, index):
    # Catch elements accidentally positioned outside the exported page.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for ax in fig.axes:
        for item in ax.texts:
            extent = item.get_window_extent(renderer)
            assert extent.x0 >= 0 and extent.x1 <= bounds.width + 1, item.get_text()
            assert extent.y0 >= 0 and extent.y1 <= bounds.height + 1, item.get_text()
    pdf.savefig(fig, facecolor=BG)
    fig.savefig(OUT / f"prediction_guide_page_{index}.png", dpi=150, facecolor=BG)
    plt.close(fig)


def verify_samples():
    rows = []
    results = {}
    folder = ROOT / "data/test_samples"
    for path in sorted(folder.glob("*.csv")):
        frame, warnings = prepare_sensor_data(pd.read_csv(path, dtype={"pet_id": "string"}))
        result = predict_sensor_dataframe(frame, ROOT / "artifacts")
        expected = sum(1 + (len(g) - 15) // 5 for _, g in frame.groupby("pet_id"))
        assert len(result) == expected
        assert np.isfinite(result.select_dtypes(include="number")).all().all()
        assert result.health_score_0_100.between(0, 100).all()
        results[path.name] = result
        rows.append({"file": path.name, "accepted": True, "pets": frame.pet_id.nunique(),
                     "windows": len(result), "warnings": warnings})
    for path in sorted((folder / "invalid").glob("*.csv")):
        try:
            prepare_sensor_data(pd.read_csv(path))
        except ValueError as exc:
            rows.append({"file": "invalid/" + path.name, "accepted": False, "reason": str(exc)})
        else:
            raise AssertionError(f"Invalid input accepted: {path}")
    (OUT / "prediction_guide_verification.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return results


def architecture(pdf):
    fig, ax = page(1, "From readings to predictions", "A multi-model pipeline screens four patterns and flags unusual sensor windows.")
    box(ax, 6, 74.8, 88, 7.5, INK)
    text(ax, 9, 81, "01  SENSOR INPUT", 10, "#8BE0D4", "bold")
    text(ax, 9, 78.5, "Heart rate  •  SpO2  •  Temperature  •  Ambient light  •  GPS", 11, "white")
    text(ax, 9, 76.3, "CSV: pet_id + timestamp + six sensor columns; target labels are not required.", 8.5, "#D9E8F0")
    arrow(ax, (50, 74.4), (50, 72.1))
    box(ax, 6, 63.3, 88, 8.2)
    text(ax, 9, 70.2, "02  PREPARE EACH PET'S HISTORY", 10, TEAL, "bold")
    text(ax, 9, 67.9, "Validate → clean → GPS speed + baseline differences → overlapping windows", 9.5)
    text(ax, 9, 65.5, "15 readings per window; advance 5 readings. The UI expects one reading per minute.", 8.7, MUTED)
    arrow(ax, (28, 62.8), (28, 60.6))
    arrow(ax, (72, 62.8), (72, 60.6))
    box(ax, 6, 54, 42, 6)
    box(ax, 52, 54, 42, 6)
    count = len(json.loads((ROOT / "artifacts/feature_columns.json").read_text()))
    text(ax, 9, 58.9, f"{count} window summary features", 10, weight="bold")
    text(ax, 9, 56.5, "Means, slopes, extremes, activity fractions", 8.1, MUTED)
    text(ax, 55, 58.9, "15 × 10 sequence values", 10, weight="bold")
    text(ax, 55, 56.5, "Ordered sensor and derived measurements", 8.1, MUTED)
    models = [
        (6, "XGBOOST", "4 classifiers\n300 trees each\nOne per condition", TEAL),
        (29, "ISOLATION\nFOREST", "Healthy windows only\n23 selected features\nAnomaly percentile", ORANGE),
        (52, "LSTM", "Shared 2-layer encoder\nLearns time sequences\n4 output scores", BLUE),
        (75, "TCN", "Temporal convolutions\nDilated residual blocks\n4 output scores", BLUE),
    ]
    for x, title, body, color in models:
        arrow(ax, (x + 9.5, 53.6), (x + 9.5, 50.8))
        box(ax, x, 36.2, 19, 14, "#FFFFFF", "#CEDCE4")
        text(ax, x + 1.6, 48.8, title, 10.4, color, "bold")
        text(ax, x + 1.6, 42.3, body, 7.8, MUTED)
        arrow(ax, (x + 9.5, 35.7), (x + 9.5, 31.8))
    box(ax, 6, 23.4, 88, 7.8, INK)
    text(ax, 9, 29.9, "03  FOUR TASK-SPECIFIC ENSEMBLES", 10, "#8BE0D4", "bold")
    text(ax, 9, 27.3, "Each logistic-regression model combines XGBoost + LSTM + TCN + anomaly.", 9, "white")
    text(ax, 9, 25.1, "The anomaly signal also remains a separate output for alerts and risk scoring.", 8.4, "#D9E8F0")
    arrow(ax, (50, 23), (50, 20.8))
    box(ax, 6, 13.5, 88, 6.7, "#E3F3EC")
    text(ax, 9, 19, "04  RESULTS FOR EACH WINDOW", 10, TEAL, "bold")
    text(ax, 9, 16.6, "4 condition estimates + anomaly → alerts → smoothed health score (0–100)", 9)
    text(ax, 6, 11.1, "CODE MAP", 8, TEAL, "bold")
    text(ax, 6, 9.2, "streamlit_app.py → ui_data.py / features.py → inference.py + models.py\nhealth_score.py creates the score; pipeline.py trains and saves artifacts.", 8.3, MUTED)
    save(pdf, fig, 1)


def decisions(pdf, results):
    fig, ax = page(2, "How to read the results", "Condition alerts and overall health status use different calculations.")
    thresholds = json.loads((ROOT / "artifacts/task_thresholds.json").read_text())
    text(ax, 6, 82, "ALERT WHEN SCORE ≥ THRESHOLD", 9.5, TEAL, "bold")
    names = [("hypoxemia", "Low oxygen"), ("fever", "Fever pattern"),
             ("abnormal_resting_hr", "Resting heart rate"), ("activity_reduction", "Reduced activity"),
             ("general_anomaly", "General anomaly")]
    for i, (task, label) in enumerate(names):
        y = 78.7 - i * 3
        ax.add_patch(Rectangle((6, y - 2.5), 39, 2.7, color=LIGHT if i % 2 == 0 else BG))
        text(ax, 7, y - .2, label, 9)
        text(ax, 43, y - .2, f"{thresholds[task]:.2f}", 9, weight="bold", ha="right")
    text(ax, 6, 62.9, "Saved thresholds: artifacts/task_thresholds.json\nSelected for validation F1; not clinical cutoffs.", 7.8, MUTED)
    box(ax, 49, 61.1, 45, 20.2)
    text(ax, 52, 79.9, "HEALTH SCORE", 10, TEAL, "bold")
    text(ax, 52, 77.2, "W = weighted mean of the five signals\nInstant risk = 0.70 W + 0.30 max(signal)\nSmoothed risk = 0.25 current + 0.75 prior\nHealth score = 100 × (1 − smoothed risk)", 9)
    text(ax, 52, 67.8, "Weights: oxygen .28 / fever .24 / HR .20\nactivity .12 / anomaly .16. First risk initializes\nthe smoother; history is separate for each pet.", 7.8, MUTED)
    text(ax, 6, 58.7, "ACTUAL SAVED-MODEL OUTPUT • SYNTHETIC DEMO", 9.5, TEAL, "bold")
    text(ax, 6, 56.5, "Shaded interval: low-oxygen input episode, 07:30–08:29. Lines end at window timestamps.", 8.2, MUTED)
    normal = results["01_normal_pattern.csv"]
    oxygen = results["02_low_oxygen.csv"]
    for bottom, height, column, ylabel, limits in [
        (.39, .145, "hypoxemia_probability", "Low-oxygen score", (0, 1.05)),
        (.205, .145, "health_score_0_100", "Health score", (0, 105)),
    ]:
        chart = fig.add_axes([.12, bottom, .79, height], facecolor="white")
        for frame, label, color in [(normal, "Reference input", BLUE), (oxygen, "Low-oxygen input", TEAL)]:
            minutes = (frame.window_end - pd.Timestamp("2026-09-20 07:00")).dt.total_seconds() / 60
            chart.plot(minutes, frame[column], color=color, lw=2, label=label)
        chart.axvspan(30, 90, color="#F4CDAB", alpha=.35, lw=0)
        if column == "hypoxemia_probability":
            chart.axhline(thresholds["hypoxemia"], color=ORANGE, lw=1, linestyle="--")
            chart.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
        else:
            for value in [60, 80]:
                chart.axhline(value, color="#9DAEB8", lw=.8, linestyle="--")
        chart.set(xlim=(0, 120), ylim=limits, ylabel=ylabel, xticks=[0, 30, 60, 90, 120],
                  xticklabels=["07:00", "07:30", "08:00", "08:30", "09:00"])
        chart.tick_params(labelsize=7.5, length=0)
        chart.yaxis.label.set_size(8)
        chart.grid(axis="y", color="#DDE5EA", lw=.6)
        for spine in chart.spines.values():
            spine.set_visible(False)
    for x, label, color in [(6, "<60  ATTENTION", "#FBE9E1"), (36, "60–<80  WATCH", "#FBF1D6"), (66, "≥80  STABLE", "#E3F3EC")]:
        box(ax, x, 15.1, 28, 3.3, color)
        text(ax, x + 14, 17.7, label, 8.8, weight="bold", ha="center")
    text(ax, 6, 13.6, "Interpretation: anomaly is a percentile, not disease probability. Reference inputs may still\ntrigger alerts. These charts demonstrate software behavior, not real-pet accuracy.", 8.4, MUTED)
    text(ax, 6, 9.4, "TRAINING / EVALUATION", 8, TEAL, "bold")
    text(ax, 6, 7.9, "Saved run: 8 synthetic pets split 5 train / 2 validation / 1 test. Evaluation reports\nprecision, recall, F1 and AUC. Synthetic performance does not establish clinical validity.", 8, MUTED)
    save(pdf, fig, 2)


def scenarios(pdf):
    fig, ax = page(3, "Test scenarios & walkthrough", "Sample pack: data/test_samples/  •  Six simulated pets  •  No target labels needed")
    steps = [(6, "1  LOAD", "Built-in six-pet demo\nor upload a CSV"),
             (36, "2  SELECT", "Choose Pet to inspect\nthen Run predictions"),
             (66, "3  CHECK", "Inspect trends + alerts\nthen download results")]
    for x, title, body in steps:
        box(ax, x, 75.8, 28, 7)
        text(ax, x + 2, 81.5, title, 9.5, TEAL, "bold")
        text(ax, x + 2, 79, body, 8.5)
    for start, end in [((34.5, 79), (35.6, 79)), ((64.5, 79), (65.6, 79))]:
        arrow(ax, start, end)
    text(ax, 6, 73.9, "PER PET: 120 READINGS → 22 WINDOWS", 9, TEAL, "bold")
    for x, width, color, label in [(6, 22, "#DDEFEA", "07:00  Reference"), (28, 44, "#FBE6D7", "07:30–08:29  Injected episode"), (72, 22, "#DDEFEA", "08:30  Recovery")]:
        ax.add_patch(Rectangle((x, 68.7), width, 3.2, color=color))
        text(ax, x + width / 2, 71.1, label, 7.9, ha="center")
    text(ax, 6, 66.7, "SCENARIO FILE", 8, MUTED, "bold")
    text(ax, 48, 66.7, "WHAT TO LOOK FOR DURING THE EPISODE", 8, MUTED, "bold")
    cases = [
        ("01_normal_pattern.csv", "No injected condition; inspect false alerts."),
        ("02_low_oxygen.csv", "SpO2 drops 7 points; inspect oxygen scores."),
        ("03_fever_pattern.csv", "Temperature +1.15°C; inspect fever scores."),
        ("04_high_resting_hr.csv", "High HR with low movement; inspect HR scores."),
        ("05_reduced_activity.csv", "Low morning movement; inspect activity scores."),
        ("06_combined_patterns.csv", "Oxygen + fever signals can rise together."),
    ]
    for i, (filename, check) in enumerate(cases):
        y = 63.8 - i * 3.4
        ax.add_patch(Rectangle((6, y - 2.8), 88, 3.1, color=LIGHT if i % 2 == 0 else BG))
        text(ax, 7.5, y - .2, filename, 8.4, weight="bold")
        text(ax, 48, y - .2, check, 8.1)
    text(ax, 6, 41.9, "ADDITIONAL INPUT CHECKS", 9, TEAL, "bold")
    text(ax, 6, 39.6, "07_multiple_pets.csv", 8.6, weight="bold")
    text(ax, 48, 39.6, "720 rows / 6 pets. Switch pets, then rerun.", 8.5)
    text(ax, 6, 36.8, "08_missing_values.csv", 8.6, weight="bold")
    text(ax, 48, 36.8, "Interpolation warning; prediction completes.", 8.5)
    text(ax, 6, 34, "09_minimal_columns.csv", 8.6, weight="bold")
    text(ax, 48, 34, "Baseline fallback warning; compare scores.", 8.5)
    box(ax, 6, 22.2, 88, 8.4, "#FBEDE6")
    text(ax, 8, 29.5, "INVALID FILES SHOULD STOP PREDICTION", 9, ORANGE, "bold")
    text(ax, 8, 27, "invalid/too_few_readings.csv → fewer than 15 rows\ninvalid/missing_spo2_column.csv → missing sensor column\ninvalid/sampling_gap.csv → timestamps not one minute apart", 8.4)
    text(ax, 6, 20.2, "VERIFIED AGAINST THE CURRENT SAVED MODELS", 9, TEAL, "bold")
    text(ax, 6, 17.9, "All 9 valid files produced finite scores and the expected window counts. All 3 invalid\nfiles were rejected. This checks execution and validation, not detection accuracy.", 8.4, MUTED)
    text(ax, 6, 13.8, "Review the full timeline: the final readings are in recovery. Confirm that switching pets\nchanges the result ID. One invalid pet must not hide other pets. Scoring is per batch;\nthis app does not connect to a live device or retain risk across separate uploads.", 8.3)
    text(ax, 6, 8, "RUN:  .\\.venv\\Scripts\\python.exe -m streamlit run streamlit_app.py", 8, TEAL, "bold")
    save(pdf, fig, 3)


def main():
    results = verify_samples()
    path = OUT / "KTINOSKARE_Prediction_Model_Testing_Guide.pdf"
    with PdfPages(path, metadata={"Title": "KTINOSKARE | Prediction Model & Testing Guide",
                                  "Author": "KTINOSKARE Project", "Subject": "Architecture, scoring, and synthetic test scenarios"}) as pdf:
        architecture(pdf)
        decisions(pdf, results)
        scenarios(pdf)
        assert pdf.get_pagecount() == 3
    print(f"Created three-page PDF: {path}")
    print("Verified 9 valid fixtures and 3 invalid fixtures; PNG previews saved alongside PDF.")


if __name__ == "__main__":
    main()
