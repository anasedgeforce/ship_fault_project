"""Create a shareable architecture summary from the current project design."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "simulator_fault_prediction_system_design.pdf"
NAVY = "#102238"
TEAL = "#087f8c"
BLUE = "#eaf3f7"
GRAY = "#526274"
LINE = "#c9d8e0"


def page():
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    return fig, ax


def title(ax, heading, subheading, number):
    ax.text(5, 94, heading, color=NAVY, fontsize=21, weight="bold", va="center")
    ax.text(5, 89.5, subheading, color=GRAY, fontsize=9.6, va="center")
    ax.plot([5, 95], [86.5, 86.5], color=TEAL, lw=2)
    ax.text(95, 3, f"{number} / 2", color=GRAY, fontsize=8, ha="right")


def box(ax, x, y, w, h, heading, body="", fill=BLUE, head_size=10, body_size=8.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.7,rounding_size=1.7",
                                fc=fill, ec=LINE, lw=0.9))
    ax.text(x + 2, y + h - 3.1, heading, color=NAVY, fontsize=head_size,
            weight="bold", va="top")
    if body:
        ax.text(x + 2, y + h - 7.6, body, color=GRAY, fontsize=body_size,
                va="top", linespacing=1.4)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                                 lw=1.5, color=TEAL, connectionstyle="arc3"))


def section(ax, x, y, heading, lines, width=43):
    ax.text(x, y, heading, color=TEAL, fontsize=11.4, weight="bold", va="top")
    ax.text(x, y - 4.2, "\n".join(lines), color=NAVY, fontsize=9.1,
            va="top", linespacing=1.55)


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with PdfPages(OUTPUT, metadata={"Title": "Simulator Fault Prediction - System Design",
                                "Author": "Simulator Fault Prediction Project"}) as pdf:
    fig, ax = page()
    title(ax, "Simulator Fault Prediction", "System architecture | 13 November 2026", 1)

    ax.text(5, 82, "MANUAL ANALYSIS", color=TEAL, fontsize=10, weight="bold")
    box(ax, 5, 64, 20, 14, "Input", "CSV / folder / pasted\nreadings / saved example")
    box(ax, 31, 64, 21, 14, "FastAPI", "Validate six channels,\n4 kHz timing and quality")
    box(ax, 58, 64, 20, 14, "Inference", "Window features + two\nsaved classifiers")
    box(ax, 83, 64, 12, 14, "Output", "One result\nper CSV", head_size=9.4)
    for start, end in [((25.7, 71), (30.3, 71)), ((52.7, 71), (57.3, 71)),
                       ((78.7, 71), (82.3, 71))]:
        arrow(ax, start, end)
    ax.text(5, 60, "Manual results appear in the page and digital twin; they are not saved to MongoDB.",
            color=GRAY, fontsize=8.5)

    ax.text(5, 54, "LIVE MONITORING", color=TEAL, fontsize=10, weight="bold")
    box(ax, 5, 35, 20, 15, "Sensor sender", "Timestamped six-channel\nbatches to HTTP API")
    box(ax, 31, 35, 21, 15, "live_data", "Save accepted batch, then\nread and verify it")
    box(ax, 58, 35, 20, 15, "Inference", "Rolling windows; family,\nsubtype and review status")
    box(ax, 83, 35, 12, 15, "Outputs", "live_faults\n+ digital twin", head_size=9.4)
    for start, end in [((25.7, 42.5), (30.3, 42.5)), ((52.7, 42.5), (57.3, 42.5)),
                       ((78.7, 42.5), (82.3, 42.5))]:
        arrow(ax, start, end)
    ax.text(5, 31, "MongoDB: ship_fault_monitoring | One fixed equipment ID | Twin animation continues while watching.",
            color=GRAY, fontsize=8.5)

    ax.text(5, 25, "FUTURE RETRAINING - SEPARATE COMMAND", color=TEAL, fontsize=10, weight="bold")
    box(ax, 5, 8, 27, 13, "Verified labeled captures", "ship_simulator_training.\nSim_data_for_traning")
    box(ax, 38, 8, 27, 13, "train_from_mongo.bat", "Validate, evaluate, train\nand publish new artifacts")
    box(ax, 71, 8, 24, 13, "Restart server", "New artifacts load only\nafter successful training")
    arrow(ax, (32.7, 14.5), (37.3, 14.5))
    arrow(ax, (65.7, 14.5), (70.3, 14.5))
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax = page()
    title(ax, "How the system works", "Inputs, decisions, storage and model lifecycle", 2)

    section(ax, 5, 82, "Signal contract", [
        "Six velocity axes: DEV, DE H, DE A, NDE V, NDE H, NDE A.",
        "Sampling: 4,000 samples/second; values in mm/s.",
        "A prediction window is 1 second (4,000 samples).",
        "A new window completes every 0.5 seconds (2,000 samples).",
        "Flat, duplicated or possibly clipped channels block prediction.",
    ])
    section(ax, 52, 82, "Prediction", [
        "Family: two blended Extra Trees models.",
        "Subtype: one Random Forest model, selected within family.",
        "Six families including Normal; 33 detailed conditions.",
        "A CSV gets one overall result from its valid windows.",
        "Model scores are uncalibrated; they are not accuracy.",
    ])

    ax.plot([5, 95], [55, 55], color=LINE, lw=1)
    section(ax, 5, 52, "Manual route", [
        "Browser -> FastAPI -> quality checks -> models -> result.",
        "Each uploaded CSV is reported separately; a folder can show",
        "several different fault candidates. Optional verified-label JSON",
        "measures accuracy for that upload. Manual results are temporary.",
    ])
    section(ax, 52, 52, "Live route", [
        "Sender -> API -> Mongo live_data -> verified readback -> models.",
        "Predicted labels and review state update live_data; one valid",
        "prediction event is also saved in live_faults. An alert needs",
        "three consecutive matching classified fault windows.",
    ])

    ax.plot([5, 95], [29, 29], color=LINE, lw=1)
    section(ax, 5, 26, "Digital twin", [
        "Exact subtype IDs map to red component highlights.",
        "Live red clears after three classified Normal windows.",
        "Only vibration is used; temperature is not measured.",
    ])
    section(ax, 52, 26, "Training and validation", [
        "Active run: 20261104T041252Z_164da514.",
        "Earlier day-held-out accuracy: family 80.10%;",
        "subtype 56.53%. New-equipment accuracy is untested.",
        "Mongo-only retraining runs on explicit command.",
    ])
    ax.text(5, 6, "Sources: README.md; docs/WEB_GUIDE.md; docs/LIVE_MONGO_GUIDE.md; docs/MONGO_RETRAINING.md; serving manifest.",
            color=GRAY, fontsize=7.6)
    pdf.savefig(fig)
    plt.close(fig)

print(OUTPUT)
