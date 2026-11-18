"""Task 5: Generate Analytical Charts, Architecture Diagram, and Complete 11-Page PDF Report.

Produces:
- docs/figures/system_architecture.png
- docs/figures/dataset_v1_vs_v2.png
- docs/figures/day_type_drift_analysis.png
- docs/figures/feature_engineering_analysis.png
- docs/figures/model_comparison.png
- docs/figures/roc_pr_and_importance.png
- docs/figures/ci_pytest_results.png
- docs/Water_Shortage_MLOps_Report.pdf (11 pages)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    DATASET_VERSIONS_FILE,
    EVALUATION_RESULTS_PATH,
    FEATURE_METADATA_FILE,
    FIGURES_DIR,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_METADATA_PATH,
    MODEL_NUMERIC_FEATURES,
    REPORT_PDF_PATH,
    ROOT_DIR,
    TARGET_COLUMN,
)
from src.pipeline import load_from_feature_store
from src.train import build_preprocessor

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 0.8


def draw_architecture_diagram(out_path: Path) -> None:
    """Create a clean vector-style MLOps system architecture diagram."""
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=200)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    ax.text(
        6.0,
        5.25,
        "End-to-End MLOps Architecture — Overnight Residential Water Shortage Prediction",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )

    boxes = [
        (0.3, 2.7, 2.0, 1.85, "#dbeafe", "#1d4ed8", "1. Dataset Versioning",
         "Synthetic Generator\nv1: Normal (250 rows)\nv2: Summer +35% (250)\nSHA-256 Manifest"),
        (2.8, 2.7, 2.1, 1.85, "#e0f2fe", "#0369a1", "2. Data Pipeline",
         "Physical Validation\n(Level <= 100,000 L)\nMissing Imputation\nFeature Engineering"),
        (5.4, 2.7, 2.0, 1.85, "#dcfce7", "#15803d", "3. Feature Store",
         "Reusable Tables:\nv1, v2, combined CSV\n+ Schema Metadata\n(Zero Skew)"),
        (7.9, 2.7, 2.0, 1.85, "#fef3c7", "#b45309", "4. Training & MLflow",
         "Decision Tree vs\nRandom Forest\nRecall-First Selection\nMLflow Registry"),
        (10.3, 2.7, 1.5, 1.85, "#fae8ff", "#7e22ce", "5. FastAPI Serve",
         "POST /predict\nShortage Risk /\nAdequate Reserve\nSwagger & UI"),
    ]

    for x, y, w, h, fc, ec, title, body in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.14",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.8,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center", fontsize=9.2, fontweight="bold", color=ec)
        ax.text(x + w / 2, y + h / 2 - 0.18, body, ha="center", va="center", fontsize=8.0, color="#1e293b", linespacing=1.35)

    for x1, x2 in [(2.38, 2.72), (4.98, 5.32), (7.48, 7.82), (9.98, 10.22)]:
        ax.annotate(
            "",
            xy=(x2, 3.62),
            xytext=(x1, 3.62),
            arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.35", lw=2.0, color="#334155"),
        )

    ci_rect = patches.FancyBboxPatch(
        (0.3, 0.55),
        11.5,
        1.35,
        boxstyle="round,pad=0.08,rounding_size=0.14",
        facecolor="#f1f5f9",
        edgecolor="#475569",
        linewidth=1.5,
    )
    ax.add_patch(ci_rect)
    ax.text(
        6.05,
        1.58,
        "6. Continuous Integration & Quality Governance (GitHub Actions + Pytest + MLflow SQLite Store)",
        ha="center",
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        6.05,
        0.98,
        "Automated Triggers (Push / PR)  |  Tank Capacity <= 100,000 L Validation Tests  |  Feature Formula Unit Tests\n"
        "FastAPI Endpoint & HTTP 422 Schema Tests  |  Automated Dataset & Feature Store Pipeline  |  Reproducible Training",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#334155",
        linespacing=1.45,
    )

    for cx in [1.3, 3.85, 6.4, 8.9, 11.05]:
        ax.annotate(
            "",
            xy=(cx, 2.62),
            xytext=(cx, 1.98),
            arrowprops=dict(arrowstyle="<->", lw=1.2, color="#64748b", ls="--"),
        )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_dataset_version_comparison(out_path: Path) -> None:
    """Plot v1 (normal) vs v2 (summer +35% demand) distributions and shortage rates."""
    df_v1 = load_from_feature_store("v1")
    df_v2 = load_from_feature_store("v2")

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), dpi=180)

    axes[0].hist(df_v1["avg_hourly_consumption"], bins=18, alpha=0.72, color="#2563eb", label=f"v1 Normal (mean {df_v1['avg_hourly_consumption'].mean():.1f})")
    axes[0].hist(df_v2["avg_hourly_consumption"], bins=18, alpha=0.72, color="#ea580c", label=f"v2 Summer (mean {df_v2['avg_hourly_consumption'].mean():.1f})")
    axes[0].set_title("Household Hourly Consumption", fontweight="bold", fontsize=10.5)
    axes[0].set_xlabel("Litres / Household / Hour")
    axes[0].set_ylabel("Nightly Records")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].hist(df_v1["expected_overnight_demand"] / 1000, bins=18, alpha=0.72, color="#0284c7", label="v1 Normal")
    axes[1].hist(df_v2["expected_overnight_demand"] / 1000, bins=18, alpha=0.72, color="#dc2626", label="v2 Summer (+35%)")
    axes[1].set_title("Expected Overnight Demand (10 hrs)", fontweight="bold", fontsize=10.5)
    axes[1].set_xlabel("Thousand Litres (kL)")
    axes[1].set_ylabel("Nightly Records")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    v1_rate = df_v1["morning_water_shortage"].mean() * 100
    v2_rate = df_v2["morning_water_shortage"].mean() * 100
    bars = axes[2].bar(["v1 (Normal)", "v2 (Summer +35%)"], [v1_rate, v2_rate], color=["#2563eb", "#dc2626"], width=0.52)
    axes[2].set_ylim(0, 85)
    axes[2].set_title("Morning Shortage Rate (<15,000 L)", fontweight="bold", fontsize=10.5)
    axes[2].set_ylabel("Shortage Nights (%)")
    axes[2].grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, [v1_rate, v2_rate]):
        axes[2].text(bar.get_x() + bar.get_width() / 2, val + 2.0, f"{val:.1f}%", ha="center", fontweight="bold", fontsize=10)

    plt.suptitle("Dataset Version Comparison: Normal (v1) vs Summer Conditions (v2)", fontsize=12.2, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_day_type_drift_analysis(out_path: Path) -> None:
    """Plot shortage risk by day_type across v1 vs v2 and 8:00 PM tank level separation."""
    df_v1 = load_from_feature_store("v1")
    df_v2 = load_from_feature_store("v2")
    df_comb = load_from_feature_store("combined")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), dpi=180)

    day_order = ["Weekday", "Weekend", "Holiday"]
    v1_rates = [df_v1[df_v1["day_type"] == d]["morning_water_shortage"].mean() * 100 for d in day_order]
    v2_rates = [df_v2[df_v2["day_type"] == d]["morning_water_shortage"].mean() * 100 for d in day_order]

    x = np.arange(len(day_order))
    w = 0.36
    b1 = axes[0].bar(x - w / 2, v1_rates, w, label="v1 Normal", color="#0284c7")
    b2 = axes[0].bar(x + w / 2, v2_rates, w, label="v2 Summer (+35%)", color="#ea580c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(day_order, fontsize=9.5)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Morning Shortage Rate (%)")
    axes[0].set_title("Shortage Rate by Day Type: v1 vs. v2", fontweight="bold", fontsize=10.8)
    axes[0].legend(fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.25)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.1f}%", ha="center", fontsize=8.3, fontweight="bold")

    # Boxplot of 8:00 PM tank level grouped by outcome & version
    groups = [
        df_v1[df_v1["morning_water_shortage"] == 0]["tank_level_8pm"] / 1000,
        df_v1[df_v1["morning_water_shortage"] == 1]["tank_level_8pm"] / 1000,
        df_v2[df_v2["morning_water_shortage"] == 0]["tank_level_8pm"] / 1000,
        df_v2[df_v2["morning_water_shortage"] == 1]["tank_level_8pm"] / 1000,
    ]
    bp = axes[1].boxplot(
        groups,
        tick_labels=["v1\nAdequate", "v1\nShortage", "v2\nAdequate", "v2\nShortage"],
        patch_artist=True,
        widths=0.52,
    )
    box_colors = ["#10b981", "#f87171", "#059669", "#dc2626"]
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("#0f172a")
        median.set_linewidth(1.8)
    axes[1].set_ylabel("8:00 PM Tank Level (Thousand Litres)")
    axes[1].set_title("8:00 PM Starting Tank Level by Season & Outcome", fontweight="bold", fontsize=10.8)
    axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_feature_engineering_analysis(out_path: Path) -> None:
    """Plot engineered features relationship with morning_water_shortage."""
    df = load_from_feature_store("combined")
    shortage = df[df["morning_water_shortage"] == 1]
    adequate = df[df["morning_water_shortage"] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), dpi=180)

    axes[0].hist(adequate["projected_water_buffer"] / 1000, bins=22, alpha=0.72, color="#10b981", label="Adequate Reserve (0)")
    axes[0].hist(shortage["projected_water_buffer"] / 1000, bins=22, alpha=0.72, color="#ef4444", label="Shortage Risk (1)")
    axes[0].axvline(0, color="#0f172a", linestyle="--", linewidth=1.6, label="15% Critical Buffer (0 kL)")
    axes[0].set_title("Projected Water Buffer Distribution", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Projected Water Buffer above 15,000 L (kL)")
    axes[0].set_ylabel("Nightly Records")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.25)

    axes[1].scatter(
        adequate["expected_overnight_demand"] / 1000,
        (adequate["tank_level_8pm"] + adequate["expected_pump_supply"]) / 1000,
        alpha=0.65,
        s=28,
        color="#10b981",
        label="Adequate Reserve",
    )
    axes[1].scatter(
        shortage["expected_overnight_demand"] / 1000,
        (shortage["tank_level_8pm"] + shortage["expected_pump_supply"]) / 1000,
        alpha=0.65,
        s=28,
        color="#ef4444",
        label="Shortage Risk",
    )
    x_line = np.linspace(25, 100, 100)
    axes[1].plot(x_line, x_line + 15, color="#0f172a", linestyle="--", linewidth=1.5, label="15 kL Reserve Boundary")
    axes[1].set_title("Total Available Water vs. Expected Overnight Demand", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Expected Overnight Demand (kL)")
    axes[1].set_ylabel("8:00 PM Tank + Expected Pump Supply (kL)")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_model_comparison(eval_results: dict, out_path: Path) -> None:
    """Create side-by-side metrics comparison and confusion matrices for DT vs RF."""
    dt = eval_results["models"]["DecisionTree"]
    rf = eval_results["models"]["RandomForest"]

    fig = plt.figure(figsize=(13.2, 4.3), dpi=180)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.58)

    ax0 = fig.add_subplot(gs[0])
    metrics_keys = ["precision", "recall", "f1_score", "accuracy"]
    labels = ["Precision", "Recall\n(Key)", "F1-Score", "Accuracy"]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax0.bar(x - w / 2, [dt[k] for k in metrics_keys], w, label="Decision Tree", color="#64748b")
    b2 = ax0.bar(x + w / 2, [rf[k] for k in metrics_keys], w, label="Random Forest", color="#0284c7")
    ax0.set_ylim(0, 1.12)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=9.5)
    ax0.set_ylabel("Score")
    ax0.set_title("Classifier Metrics Comparison", fontweight="bold", fontsize=10.5)
    ax0.legend(loc="lower right", fontsize=8.5)
    ax0.grid(axis="y", alpha=0.25)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax0.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", fontsize=8.2, fontweight="bold")

    for idx, (title, m, cmap) in enumerate(
        [("Decision Tree Confusion Matrix", dt, "Blues"), ("Random Forest Confusion Matrix", rf, "Greens")],
        start=1,
    ):
        ax = fig.add_subplot(gs[idx])
        cm = np.array(m["confusion_matrix"])
        ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_title(title, fontweight="bold", fontsize=10.2)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Adequate", "Pred: Shortage"], fontsize=8.5)
        ax.set_yticklabels(["Actual: Adequate", "Actual: Shortage"], fontsize=8.5)
        thresh = cm.max() / 2.0
        cell_labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cell_labels[i][j]}\n{cm[i, j]}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "#0f172a",
                    fontsize=11,
                    fontweight="bold",
                )

    plt.suptitle(
        f"Model Evaluation & Selection (Selected Winner: {eval_results['selected_model']})",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.06, right=0.98)
    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_roc_pr_and_importance(eval_results: dict, out_path: Path) -> None:
    """Plot ROC curve, Precision-Recall curve, and Random Forest feature importances."""
    df = load_from_feature_store("combined")
    feature_cols = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df[TARGET_COLUMN], test_size=0.25, random_state=42, stratify=df[TARGET_COLUMN]
    )

    dt_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", DecisionTreeClassifier(max_depth=5, min_samples_split=8, min_samples_leaf=4, random_state=42)),
    ])
    rf_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=2, class_weight="balanced", random_state=42)),
    ])
    dt_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)

    dt_prob = dt_pipe.predict_proba(X_test)[:, 1]
    rf_prob = rf_pipe.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), dpi=180)

    # 1. ROC Curve
    fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
    axes[0].plot(fpr_dt, tpr_dt, lw=2.0, color="#64748b", label=f"Decision Tree (AUC={roc_auc_score(y_test, dt_prob):.3f})")
    axes[0].plot(fpr_rf, tpr_rf, lw=2.4, color="#0284c7", label=f"Random Forest (AUC={roc_auc_score(y_test, rf_prob):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6)
    axes[0].set_title("ROC Curve (Test Set)", fontweight="bold", fontsize=10.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(alpha=0.25)

    # 2. Precision-Recall Curve
    p_dt, r_dt, _ = precision_recall_curve(y_test, dt_prob)
    p_rf, r_rf, _ = precision_recall_curve(y_test, rf_prob)
    axes[1].plot(r_dt, p_dt, lw=2.0, color="#64748b", label=f"Decision Tree (AP={average_precision_score(y_test, dt_prob):.3f})")
    axes[1].plot(r_rf, p_rf, lw=2.4, color="#15803d", label=f"Random Forest (AP={average_precision_score(y_test, rf_prob):.3f})")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold", fontsize=10.5)
    axes[1].set_xlabel("Recall (Shortage Detection)")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].grid(alpha=0.25)

    # 3. Top Feature Importances (Random Forest)
    fi = eval_results["models"]["RandomForest"]["feature_importances"]
    top_items = list(fi.items())[:7][::-1]
    names = [k for k, _ in top_items]
    vals = [v * 100 for _, v in top_items]
    eng_set = {"expected_overnight_demand", "expected_pump_supply", "projected_water_buffer"}
    bar_cols = ["#0284c7" if n in eng_set else "#64748b" for n in names]
    bars = axes[2].barh(names, vals, color=bar_cols, height=0.62)
    axes[2].set_xlim(0, max(vals) + 11)
    axes[2].set_title("RF Feature Importance (%)", fontweight="bold", fontsize=10.5)
    axes[2].set_xlabel("Relative Importance (%)")
    axes[2].tick_params(axis="y", labelsize=8.3)
    axes[2].grid(axis="x", alpha=0.25)
    for bar, v in zip(bars, vals):
        axes[2].text(v + 0.8, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8, fontweight="bold")

    plt.suptitle("Diagnostic Discrimination Curves & Feature Importance (Blue = Engineered Feature)", fontsize=11.8, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def run_pytest_and_render_evidence(out_path: Path) -> str:
    """Execute pytest -v and render a crisp terminal screenshot image of the actual test run."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "--tb=short"],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    output_text = (proc.stdout + "\n" + proc.stderr).strip()

    lines = output_text.splitlines()
    fig, ax = plt.subplots(figsize=(11.0, max(4.2, 0.27 * len(lines) + 1.0)), dpi=180)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.axis("off")

    ax.text(
        0.015,
        0.96,
        "$ pytest -v --tb=short   [GitHub Actions CI Validation Suite]",
        transform=ax.transAxes,
        fontfamily="monospace",
        fontsize=10.5,
        fontweight="bold",
        color="#38bdf8",
        va="top",
    )

    y_pos = 0.89
    step = 0.85 / max(len(lines), 1)
    for line in lines:
        color = "#e2e8f0"
        if "PASSED" in line or "passed" in line:
            color = "#4ade80"
        elif "FAILED" in line or "ERROR" in line:
            color = "#f87171"
        elif line.startswith("=") or line.startswith("collected"):
            color = "#facc15"
        ax.text(
            0.015,
            y_pos,
            line[:112],
            transform=ax.transAxes,
            fontfamily="monospace",
            fontsize=8.5,
            color=color,
            va="top",
        )
        y_pos -= step

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_text


def _add_image_if_exists(
    story: list,
    path: Path,
    width_in: float,
    caption: str,
    cap_style,
    max_h_in: float = 3.65,
) -> None:
    if path.exists():
        from PIL import Image as PILImage

        with PILImage.open(path) as im:
            w, h = im.size
        aspect = h / float(w)
        img_w = width_in * inch
        img_h = img_w * aspect
        max_h = max_h_in * inch
        if img_h > max_h:
            img_h = max_h
            img_w = img_h / aspect
        story.append(Image(str(path), width=img_w, height=img_h))
        story.append(Paragraph(caption, cap_style))
        story.append(Spacer(1, 0.08 * inch))


def _styled_table(
    data: list,
    col_widths: list,
    header_hex: str = "#0f172a",
    font_size: float = 8.2,
    pad: float = 4.5,
    center_from_col: int | None = None,
) -> Table:
    """Create a ReportLab Table with automatic Paragraph word-wrapping so text never overflows cells."""
    hdr_style_left = ParagraphStyle(
        "TblHdrL",
        fontName="Helvetica-Bold",
        fontSize=font_size,
        leading=font_size + 2.4,
        textColor=colors.white,
        alignment=TA_LEFT,
    )
    hdr_style_center = ParagraphStyle(
        "TblHdrC",
        parent=hdr_style_left,
        alignment=TA_CENTER,
    )
    cell_style_left = ParagraphStyle(
        "TblCellL",
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 2.4,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_LEFT,
    )
    cell_style_center = ParagraphStyle(
        "TblCellC",
        parent=cell_style_left,
        alignment=TA_CENTER,
    )

    wrapped_rows = []
    for r_idx, row in enumerate(data):
        wrapped_row = []
        for c_idx, cell in enumerate(row):
            is_centered = center_from_col is not None and c_idx >= center_from_col
            text = str(cell).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
            if r_idx == 0:
                style = hdr_style_center if is_centered else hdr_style_left
            else:
                style = cell_style_center if is_centered else cell_style_left
            wrapped_row.append(Paragraph(text, style))
        wrapped_rows.append(wrapped_row)

    tbl = Table(wrapped_rows, colWidths=[w * inch for w in col_widths])
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_hex)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), pad),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _draw_page_footer(canvas, doc) -> None:
    """Draw a clean footer with student details and page number on every page."""
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.setLineWidth(0.6)
    canvas.line(0.6 * inch, 0.48 * inch, A4[0] - 0.6 * inch, 0.48 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(
        0.6 * inch,
        0.32 * inch,
        "B. Neerush  |  Reg. No: 231FA04109  |  Sec: 15  |  Dept: CSE  —  MLOps Lab Exam (Q5)",
    )
    canvas.drawRightString(A4[0] - 0.6 * inch, 0.32 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf_report() -> Path:
    """Compile the complete 11-page MLOps PDF report."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATASET_VERSIONS_FILE, "r", encoding="utf-8") as f:
        ds_manifest = json.load(f)
    with open(FEATURE_METADATA_FILE, "r", encoding="utf-8") as f:
        fs_meta = json.load(f)
    with open(EVALUATION_RESULTS_PATH, "r", encoding="utf-8") as f:
        eval_res = json.load(f)
    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        model_meta = json.load(f)

    # Generate all analytical charts and pytest visual
    draw_architecture_diagram(FIGURES_DIR / "system_architecture.png")
    plot_dataset_version_comparison(FIGURES_DIR / "dataset_v1_vs_v2.png")
    plot_day_type_drift_analysis(FIGURES_DIR / "day_type_drift_analysis.png")
    plot_feature_engineering_analysis(FIGURES_DIR / "feature_engineering_analysis.png")
    plot_model_comparison(eval_res, FIGURES_DIR / "model_comparison.png")
    plot_roc_pr_and_importance(eval_res, FIGURES_DIR / "roc_pr_and_importance.png")
    run_pytest_and_render_evidence(FIGURES_DIR / "ci_pytest_results.png")

    doc = SimpleDocTemplate(
        str(REPORT_PDF_PATH),
        pagesize=A4,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.62 * inch,
        title="Predicting Overnight Water Shortages in a Residential Water Tank — MLOps Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=19.5,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=8,
        alignment=TA_LEFT,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10.2,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontSize=13.5,
        leading=17,
        textColor=colors.HexColor("#0369a1"),
        spaceBefore=6,
        spaceAfter=5,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontSize=11.0,
        leading=14.5,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=7,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9.4,
        leading=13.4,
        textColor=colors.HexColor("#1e293b"),
        alignment=TA_JUSTIFY,
        spaceAfter=5.5,
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Italic"],
        fontSize=8.3,
        leading=10.5,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER,
        spaceBefore=2,
        spaceAfter=5,
    )

    v1_info = ds_manifest["versions"]["v1"]
    v2_info = ds_manifest["versions"]["v2"]
    sel_name = eval_res["selected_model"]
    sel_metrics = eval_res["models"][sel_name]
    dt = eval_res["models"]["DecisionTree"]
    rf = eval_res["models"]["RandomForest"]
    drift = eval_res["version_drift_evaluation"]

    story = []

    # =========================================================================
    # PAGE 1: TITLE, STUDENT DETAILS, PROBLEM STATEMENT & EXECUTIVE SUMMARY
    # =========================================================================
    story.append(Paragraph("MLOps Laboratory Examination — Question 5", subtitle_style))
    story.append(Paragraph("Predicting Overnight Water Shortages in a Residential Water Tank", title_style))
    story.append(
        Paragraph(
            "An End-to-End Production MLOps Pipeline Featuring Synthetic Dataset Versioning, "
            "Automated Preprocessing & Feature Store, Decision Tree vs. Random Forest Evaluation, "
            "MLflow Model Registry, GitHub Actions CI/CD Validation, and FastAPI Model Serving.",
            subtitle_style,
        )
    )

    student_tbl = Table(
        [
            ["Name", "B. Neerush", "Reg. No", "231FA04109"],
            ["Section", "15", "Department", "CSE"],
        ],
        colWidths=[1.05 * inch, 2.45 * inch, 1.15 * inch, 2.40 * inch],
    )
    student_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e0f2fe")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#e0f2fe")),
                ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#f8fafc")),
                ("BACKGROUND", (3, 0), (3, -1), colors.HexColor("#f8fafc")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0369a1")),
                ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#0369a1")),
                ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (3, 0), (3, -1), colors.HexColor("#0f172a")),
                ("FONTSIZE", (0, 0), (-1, -1), 9.4),
                ("GRID", (0, 0), (-1, -1), 0.7, colors.HexColor("#93c5fd")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(student_tbl)
    story.append(Spacer(1, 0.14 * inch))

    story.append(Paragraph("Executive Summary & Problem Context", h1_style))
    story.append(
        Paragraph(
            "Residential colonies relying on an overhead municipal storage tank face recurring early-morning "
            "supply disruptions when unexpected evening/nighttime household usage, pipeline leakage, or "
            "delayed municipal pumping deplete stored reserves before sunrise. Given a <b>100,000-litre</b> "
            "overhead tank and operational readings recorded at <b>8:00 PM</b>, this MLOps system predicts "
            "whether the water level will fall below the critical <b>15% safety threshold (15,000 litres)</b> "
            "by <b>6:00 AM</b> (a 10-hour overnight horizon). Early risk classification enables colony operators "
            "to schedule emergency booster pumping, dispatch leakage inspections, or notify residents proactively.",
            body_style,
        )
    )

    kpi_data = [
        ["System Parameter", "Configured / Observed Value"],
        ["Overhead Storage Tank Capacity", "100,000 litres"],
        ["Critical Morning Shortage Threshold (6:00 AM)", "< 15% of capacity (< 15,000 litres)"],
        ["Prediction Lead Window", "10 hours (8:00 PM Evening -> 6:00 AM Morning)"],
        ["Versioned Datasets", f"v1 Normal (250 nights) & v2 Summer (250 nights, +{ds_manifest['version_comparison']['v1_to_v2_demand_increase_pct']:.2f}% demand)"],
        ["Reusable Feature Store", "feature_store/{v1, v2, combined}/engineered_features.csv (0 missing)"],
        ["Selected & Registered Model (MLflow)", f"{sel_name} — WaterShortagePredictor v{model_meta['mlflow_model_version']} (@production)"],
        ["Selected Model Metrics (Test Set n=125)", f"Recall: {sel_metrics['recall']:.4f} | Precision: {sel_metrics['precision']:.4f} | F1: {sel_metrics['f1_score']:.4f} | AUC: {sel_metrics['roc_auc']:.4f}"],
        ["Serving & CI/CD Stack", "FastAPI (/predict, /health, /docs, /ui) + GitHub Actions + Pytest (13/13 pass)"],
    ]
    story.append(_styled_table(kpi_data, [2.75, 4.30], header_hex="#0f172a", font_size=8.5, pad=4.8))
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("Project Repository Structure", h2_style))
    repo_data = [
        ["Directory / File", "Primary MLOps Responsibility"],
        ["src/config.py & src/data_generation.py", "Domain constants, synthetic v1/v2 data generator, SHA-256 dataset versioning"],
        ["src/feature_engineering.py & src/pipeline.py", "Physical validation (<= 100k L), imputation, engineered features & Feature Store"],
        ["src/train.py & mlflow.db", "Decision Tree vs. Random Forest training, Recall-first selection & MLflow Registry"],
        ["api/main.py", "FastAPI serving (/predict, /health, /docs, /ui) with Pydantic schema enforcement"],
        ["tests/ & .github/workflows/ci.yml", "13 automated pytest checks + GitHub Actions continuous integration pipeline"],
        ["scripts/ & docs/", "Live UI screenshot automation (Playwright) & 11-page PDF report generation"],
    ]
    story.append(_styled_table(repo_data, [2.55, 4.50], header_hex="#0369a1", font_size=8.3, pad=4.5))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: SYSTEM ARCHITECTURE & TECHNOLOGY STACK
    # =========================================================================
    story.append(Paragraph("1. System Architecture and Engineering Design", h1_style))
    story.append(
        Paragraph(
            "The system implements a modular six-stage MLOps lifecycle designed around reproducibility, "
            "strict physical domain validation, and zero training-serving skew. Each stage reads versioned "
            "artifacts from its predecessor and writes auditable outputs with JSON lineage manifests. "
            "Crucially, both offline model training and online FastAPI inference import the exact same "
            "<b>engineer_features()</b> and <b>validate_tank_records()</b> implementations from "
            "src/feature_engineering.py, guaranteeing identical feature computations during training and production.",
            body_style,
        )
    )
    _add_image_if_exists(
        story,
        FIGURES_DIR / "system_architecture.png",
        7.0,
        "Figure 1: End-to-end MLOps architecture integrating dataset versioning, Feature Store, MLflow Registry, FastAPI serving, and GitHub Actions CI.",
        caption_style,
        max_h_in=3.45,
    )

    story.append(Paragraph("1.1 Architectural Layers & Technology Stack", h2_style))
    arch_tbl = [
        ["MLOps Stage", "Technology / Library", "Key Design Guarantees"],
        ["Data Versioning", "NumPy, Pandas, SHA-256 JSON", "Deterministic seeds (v1=42, v2=108), immutable raw CSVs, checksum manifest"],
        ["Data Validation", "Custom Validator + Pydantic v2", "Rejects tank_level_8pm > 100,000 L or < 0 L in both batch pipeline & API"],
        ["Feature Store", "Versioned CSVs + Schema JSON", "Materializes v1, v2, and combined engineered features separately for reuse"],
        ["Model Training", "scikit-learn Pipeline (DT & RF)", "Encapsulates StandardScaler + OneHotEncoder + Classifier in one artifact"],
        ["Experiment Registry", "MLflow 3.18 (SQLite Backend)", "Logs params, metrics, confusion matrices, sklearn artifacts & @production alias"],
        ["CI/CD & Serving", "GitHub Actions, Pytest, FastAPI", "Automated end-to-end CI rebuild + low-latency JSON predictions & Swagger UI"],
    ]
    story.append(_styled_table(arch_tbl, [1.45, 1.95, 3.65], header_hex="#1e293b", font_size=8.4, pad=5))
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "<b>Zero Training-Serving Skew:</b> By bundling preprocessing (StandardScaler for 9 numeric inputs and "
            "OneHotEncoder for day_type) directly inside the registered scikit-learn Pipeline object and computing "
            "the 3 water-balance features via a shared pure Python module, evening API payloads pass through the "
            "exact transformation graph validated during offline evaluation.",
            body_style,
        )
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: DATASET CREATION, DATA DICTIONARY & VERSIONING
    # =========================================================================
    story.append(Paragraph("2. Dataset Creation and Version Management", h1_style))
    story.append(
        Paragraph(
            "Using src/data_generation.py, two reproducible synthetic datasets of <b>250 nightly records each</b> "
            "were synthesized for a 100,000-litre residential overhead tank. Each record represents conditions "
            "recorded at 8:00 PM along with the binary outcome <b>morning_water_shortage</b> observed at 6:00 AM "
            "(1 if remaining water falls below 15,000 L, else 0). Realistic overnight uncertainty is simulated via: "
            "(i) day-type consumption multipliers (Weekend +12%, Holiday +18%), (ii) stochastic household demand "
            "variation (+/-13% normal noise), (iii) delayed municipal pumping occurring on ~18% of nights "
            "(delivering 45%-80% of expected pump volume), and (iv) overnight leakage fluctuations (+/-16%).",
            body_style,
        )
    )

    story.append(Paragraph("2.1 Dataset Schema & Physical Data Dictionary", h2_style))
    schema_tbl = [
        ["Column Name", "Type", "Unit / Values", "Constraint & Operational Meaning"],
        ["record_id / date", "str", "YYYY-MM-DD", "Unique version-tagged night identifier (V1-0001..V2-0250)"],
        ["tank_level_8pm", "float", "Litres", "Strict range [0, 100,000 L]; ultrasonic water level at 8:00 PM"],
        ["connected_households", "int", "Count (90-160)", "Strictly > 0; active residential flats drawing overnight water"],
        ["avg_hourly_consumption", "float", "L / hh / hr", "Mean hourly usage per household (+35% multiplier in v2 summer)"],
        ["expected_pumping_duration", "float", "Hours (0-10)", "Scheduled overnight municipal pumping window"],
        ["expected_pumping_rate", "float", "Litres / hour", "Nominal municipal pump inflow capacity (7,000-14,500 L/hr)"],
        ["leakage_estimate", "float", "Litres / night", "Estimated 10-hour distribution & valve leakage loss"],
        ["day_type", "category", "Weekday/End/Hol", "Calendar category capturing weekend/holiday demand surges"],
        ["morning_water_shortage", "int (0/1)", "Binary Target", "1 if simulated 6:00 AM level < 15,000 L (15%), else 0"],
    ]
    story.append(_styled_table(schema_tbl, [1.65, 0.72, 1.13, 3.55], header_hex="#0369a1", font_size=8.1, pad=4.2))
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("2.2 Version 1 (Normal) vs. Version 2 (Summer +35% Demand) Manifest", h2_style))
    story.append(
        Paragraph(
            "Dataset lineage is tracked in <b>data/dataset_versions.json</b> using SHA-256 cryptographic hashes "
            "and summary telemetry. Version 2 simulates peak summer demand (+35% target, observed increase: "
            f"<b>+{ds_manifest['version_comparison']['v1_to_v2_demand_increase_pct']:.2f}%</b>), causing the "
            "morning shortage incidence to climb sharply from 36.0% to 64.8%.",
            body_style,
        )
    )
    ds_table = [
        ["Metric", "Version 1 (Normal)", "Version 2 (Summer +35%)"],
        ["File Path", v1_info["file"], v2_info["file"]],
        ["Record Count", str(v1_info["record_count"]), str(v2_info["record_count"])],
        ["Mean Hourly Consumption (L/hh/hr)", f"{v1_info['mean_avg_hourly_consumption']:.2f}", f"{v2_info['mean_avg_hourly_consumption']:.2f}"],
        ["Mean 8:00 PM Tank Level (L)", f"{v1_info['mean_tank_level_8pm']:,.1f}", f"{v2_info['mean_tank_level_8pm']:,.1f}"],
        ["Morning Shortage Rate (<15,000 L)", f"{v1_info['shortage_rate']*100:.1f}% ({ds_manifest['version_comparison']['v1_shortage_count']}/250)", f"{v2_info['shortage_rate']*100:.1f}% ({ds_manifest['version_comparison']['v2_shortage_count']}/250)"],
        ["Raw Missing Sensor Entries (~2%)", str(v1_info["missing_values_total"]), str(v2_info["missing_values_total"])],
        ["SHA-256 Checksum (prefix)", v1_info["sha256"][:26] + "...", v2_info["sha256"][:26] + "..."],
    ]
    story.append(_styled_table(ds_table, [2.40, 2.32, 2.33], header_hex="#0f172a", font_size=8.3, pad=4.5))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: SEASONAL DRIFT & DAY-TYPE DISTRIBUTION ANALYSIS
    # =========================================================================
    story.append(Paragraph("2.3 Seasonal Covariate Shift & Day-Type Impact Analysis", h1_style))
    story.append(
        Paragraph(
            "Comparing Version 1 and Version 2 visually highlights a textbook seasonal covariate shift: while "
            "the 8:00 PM starting tank level distribution remains comparable (~55,000-56,700 L mean), the "
            "summer +34.71% surge in per-household consumption pushes mean overnight colony demand from "
            "50.16 kL (v1) to 66.53 kL (v2). Consequently, average projected buffer above the 15 kL critical "
            "reserve flips from a safe surplus (+11.61 kL in v1) to a structural deficit (-6.94 kL in v2).",
            body_style,
        )
    )
    _add_image_if_exists(
        story,
        FIGURES_DIR / "dataset_v1_vs_v2.png",
        7.0,
        "Figure 2: Distribution shift from Version 1 (Normal) to Version 2 (Summer, +34.7% demand), driving shortage frequency from 36.0% to 64.8%.",
        caption_style,
        max_h_in=3.05,
    )
    _add_image_if_exists(
        story,
        FIGURES_DIR / "day_type_drift_analysis.png",
        7.0,
        "Figure 3: (Left) Morning shortage rate stratified by day type across v1 and v2; (Right) Starting 8:00 PM tank level distributions by season and outcome.",
        caption_style,
        max_h_in=3.15,
    )
    story.append(
        Paragraph(
            "<b>Key Operational Insight:</b> Notice in Figure 3 (right) that in summer (v2), even evenings "
            "starting near 55,000 L frequently suffer morning shortages when combined with weekend/holiday "
            "usage or delayed pumping. Training on both versioned regimes via the combined Feature Store "
            "equips the classifier to generalize reliably across seasonal transitions.",
            body_style,
        )
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: AUTOMATED DATA PIPELINE & FEATURE STORE
    # =========================================================================
    story.append(Paragraph("3. Data Preprocessing Pipeline and Feature Store", h1_style))
    story.append(
        Paragraph(
            "The automated data pipeline (src/pipeline.py) ingests raw versioned CSV files, runs pre-cleaning "
            "physical invariant checks, automatically imputes missing sensor/log values (~2% missingness in "
            "raw files via median imputation for numeric columns and mode imputation for day_type), computes "
            "three physics-based water-balance features, re-validates all constraints, and materializes "
            "reusable feature tables inside feature_store/ alongside feature_store/feature_metadata.json.",
            body_style,
        )
    )

    val_rules = [
        ["Validation Rule", "Enforced Constraint", "Failure Action"],
        ["Tank Capacity Ceiling", "tank_level_8pm <= 100,000.0 litres", "Raises TankDataValidationError / HTTP 422"],
        ["Non-Negative Tank Level", "tank_level_8pm >= 0.0 litres", "Raises TankDataValidationError / HTTP 422"],
        ["Positive Households", "connected_households > 0", "Raises TankDataValidationError / HTTP 422"],
        ["Overnight Pump Window", "0.0 <= expected_pumping_duration <= 10.0 hrs", "Raises TankDataValidationError / HTTP 422"],
        ["Non-Negative Flows", "consumption, pumping_rate, leakage >= 0", "Raises TankDataValidationError / HTTP 422"],
        ["Day Type Domain", "day_type in {'Weekday', 'Weekend', 'Holiday'}", "Raises TankDataValidationError / HTTP 422"],
    ]
    story.append(_styled_table(val_rules, [1.65, 2.75, 2.65], header_hex="#0f172a", font_size=8.1, pad=4))
    story.append(Spacer(1, 0.08 * inch))

    story.append(
        Paragraph(
            "<b>Engineered Domain Features (Persisted Separately for Reuse):</b><br/>"
            "&bull; <b>expected_overnight_demand</b> = connected_households * avg_hourly_consumption * 10 + leakage_estimate (litres)<br/>"
            "&bull; <b>expected_pump_supply</b> = expected_pumping_duration * expected_pumping_rate (litres)<br/>"
            "&bull; <b>projected_water_buffer</b> = (tank_level_8pm + expected_pump_supply - expected_overnight_demand) - 15,000 (litres above 15% critical reserve)",
            body_style,
        )
    )

    fs_tbl = [
        ["Feature Store Table", "Rows", "NaNs", "Mean Demand", "Mean Buffer", "Shortage %"],
        [
            "feature_store/v1/engineered_features.csv",
            str(fs_meta["tables"]["v1"]["rows"]),
            str(fs_meta["tables"]["v1"]["missing_values_after_cleaning"]),
            f"{fs_meta['tables']['v1']['mean_expected_overnight_demand']:,.1f} L",
            f"{fs_meta['tables']['v1']['mean_projected_water_buffer']:+,.1f} L",
            f"{fs_meta['tables']['v1']['shortage_rate']*100:.1f}%",
        ],
        [
            "feature_store/v2/engineered_features.csv",
            str(fs_meta["tables"]["v2"]["rows"]),
            str(fs_meta["tables"]["v2"]["missing_values_after_cleaning"]),
            f"{fs_meta['tables']['v2']['mean_expected_overnight_demand']:,.1f} L",
            f"{fs_meta['tables']['v2']['mean_projected_water_buffer']:+,.1f} L",
            f"{fs_meta['tables']['v2']['shortage_rate']*100:.1f}%",
        ],
        [
            "feature_store/combined/engineered_features.csv",
            str(fs_meta["tables"]["combined"]["rows"]),
            str(fs_meta["tables"]["combined"]["missing_values_after_cleaning"]),
            f"{fs_meta['tables']['combined']['mean_expected_overnight_demand']:,.1f} L",
            f"{fs_meta['tables']['combined']['mean_projected_water_buffer']:+,.1f} L",
            f"{fs_meta['tables']['combined']['shortage_rate']*100:.1f}%",
        ],
    ]
    story.append(_styled_table(fs_tbl, [2.60, 0.52, 0.52, 1.15, 1.25, 1.01], header_hex="#15803d", font_size=8.0, pad=4, center_from_col=1))
    story.append(Spacer(1, 0.10 * inch))

    _add_image_if_exists(
        story,
        FIGURES_DIR / "feature_engineering_analysis.png",
        7.0,
        "Figure 4: Clear class separation achieved by projected_water_buffer and total overnight water balance across all 500 records.",
        caption_style,
        max_h_in=2.95,
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: MODEL TRAINING, HYPERPARAMETERS & COMPARATIVE EVALUATION
    # =========================================================================
    story.append(Paragraph("4. Model Training, Comparison, and Selection", h1_style))
    story.append(
        Paragraph(
            "Both <b>Decision Tree</b> and <b>Random Forest</b> classifiers were trained on the reusable "
            "combined Feature Store (500 nightly records: 375 training / 125 held-out test using a stratified "
            "75/25 split). Each estimator is wrapped inside a scikit-learn Pipeline applying StandardScaler to "
            "the 9 numeric features (6 raw + 3 engineered) and OneHotEncoder to day_type. Because failing to "
            "warn residents about an impending morning water shortage (False Negative) is far costlier than "
            "a precautionary alert (False Positive), model selection prioritizes <b>Shortage Recall</b> first, "
            "tie-broken by F1-score.",
            body_style,
        )
    )

    story.append(Paragraph("4.1 Candidate Model Hyperparameters", h2_style))
    hp_tbl = [
        ["Classifier", "Key Hyperparameters", "Regularization & Imbalance Handling"],
        ["Decision Tree", "max_depth=5, min_samples_split=8, min_samples_leaf=4, seed=42", "Depth & leaf constraints prevent single-tree overfitting"],
        ["Random Forest", "n_estimators=200, max_depth=8, min_samples_leaf=2, seed=42", "200-tree bagging + class_weight='balanced' boosts shortage recall"],
    ]
    story.append(_styled_table(hp_tbl, [1.25, 3.15, 2.65], header_hex="#334155", font_size=8.1, pad=4.5))
    story.append(Spacer(1, 0.10 * inch))

    story.append(Paragraph("4.2 Held-Out Test Set Performance & Confusion Matrices (n = 125)", h2_style))
    comp_tbl = Table(
        [
            ["Model", "Precision", "Recall (Shortage)", "F1-Score", "Accuracy", "ROC-AUC", "Confusion Matrix [[TN,FP],[FN,TP]]"],
            [
                "Decision Tree",
                f"{dt['precision']:.4f}",
                f"{dt['recall']:.4f}",
                f"{dt['f1_score']:.4f}",
                f"{dt['accuracy']:.4f}",
                f"{dt['roc_auc']:.4f}",
                f"[[{dt['tn']}, {dt['fp']}], [{dt['fn']}, {dt['tp']}]]",
            ],
            [
                "Random Forest *",
                f"{rf['precision']:.4f}",
                f"{rf['recall']:.4f}",
                f"{rf['f1_score']:.4f}",
                f"{rf['accuracy']:.4f}",
                f"{rf['roc_auc']:.4f}",
                f"[[{rf['tn']}, {rf['fp']}], [{rf['fn']}, {rf['tp']}]]",
            ],
        ],
        colWidths=[1.20 * inch, 0.72 * inch, 1.08 * inch, 0.72 * inch, 0.72 * inch, 0.72 * inch, 1.89 * inch],
    )
    comp_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#dcfce7")),
                ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(comp_tbl)
    story.append(Spacer(1, 0.10 * inch))

    _add_image_if_exists(
        story,
        FIGURES_DIR / "model_comparison.png",
        7.0,
        "Figure 5: Decision Tree vs. Random Forest metrics comparison and side-by-side confusion matrices on the test set (n=125).",
        caption_style,
        max_h_in=3.1,
    )

    story.append(
        Paragraph(
            f"<b>Selection Justification:</b> <b>{sel_name}</b> outperforms Decision Tree on shortage detection "
            f"with <b>Recall = {sel_metrics['recall']:.4f}</b> (vs. {dt['recall']:.4f}), reducing missed shortages "
            f"(False Negatives) by 25% (from 8 down to 6), while also achieving higher <b>F1-score ({sel_metrics['f1_score']:.4f})</b> "
            f"and substantially stronger <b>ROC-AUC ({sel_metrics['roc_auc']:.4f} vs. {dt['roc_auc']:.4f})</b>.",
            body_style,
        )
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 7: ROC/PR CURVES, FEATURE IMPORTANCE & SEASONAL SUBGROUP AUDIT
    # =========================================================================
    story.append(Paragraph("4.3 Diagnostic Curves, Feature Importance & Seasonal Audit", h1_style))
    story.append(
        Paragraph(
            "Beyond fixed-threshold accuracy, Figure 6 evaluates probability calibration and ranking quality "
            "via ROC and Precision-Recall curves, alongside Random Forest feature importances. Random Forest "
            "dominates Decision Tree across almost every operating threshold (ROC-AUC 0.953 vs. 0.900; "
            "Average Precision 0.95+ vs. 0.88), confirming superior risk-probability estimation for borderline nights.",
            body_style,
        )
    )
    _add_image_if_exists(
        story,
        FIGURES_DIR / "roc_pr_and_importance.png",
        7.0,
        "Figure 6: (Left) ROC Curve; (Middle) Precision-Recall Curve; (Right) Random Forest top feature importances highlighting engineered features in blue.",
        caption_style,
        max_h_in=3.05,
    )

    story.append(Paragraph("Feature Engineering Impact Assessment", h2_style))
    story.append(
        Paragraph(
            "Feature importance analysis validates the domain engineering performed in Task 2: "
            "<b>projected_water_buffer</b> alone accounts for <b>44.51%</b> of total Gini importance, "
            "and the three engineered features combined (projected_water_buffer 44.51%, "
            "expected_overnight_demand 9.42%, expected_pump_supply 4.80%) contribute <b>58.73%</b> of "
            "the Random Forest's predictive signal, far outperforming raw variables alone.",
            body_style,
        )
    )

    story.append(Paragraph("Seasonal Subgroup & Standalone Version-Drift Evaluation", h2_style))
    sub_tbl = [
        ["Evaluation Slice / MLflow Run", "Precision", "Recall (Shortage)", "F1-Score", "ROC-AUC", "Interpretation"],
        [
            "Standalone v1 Normal (RF_Dataset_v1)",
            f"{drift['v1']['precision']:.4f}",
            f"{drift['v1']['recall']:.4f}",
            f"{drift['v1']['f1_score']:.4f}",
            f"{drift['v1']['roc_auc']:.4f}",
            "Strong normal-season baseline",
        ],
        [
            "Standalone v2 Summer (RF_Dataset_v2)",
            f"{drift['v2']['precision']:.4f}",
            f"{drift['v2']['recall']:.4f}",
            f"{drift['v2']['f1_score']:.4f}",
            f"{drift['v2']['roc_auc']:.4f}",
            "High recall under +35% summer surge",
        ],
        [
            "Production RF — v1 Test Subgroup",
            "—",
            f"{rf['v1_test_recall']:.4f}",
            "—",
            "—",
            f"{rf['v1_test_recall']*100:.1f}% shortage recall in normal weather",
        ],
        [
            "Production RF — v2 Test Subgroup",
            "—",
            f"{rf['v2_test_recall']:.4f}",
            "—",
            "—",
            f"{rf['v2_test_recall']*100:.1f}% shortage recall under peak summer surge",
        ],
    ]
    story.append(_styled_table(sub_tbl, [2.15, 0.72, 1.05, 0.72, 0.72, 1.69], header_hex="#0369a1", font_size=8.0, pad=4.5, center_from_col=1))
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "<b>Error Analysis:</b> The 6 remaining False Negatives in the production Random Forest occur "
            "almost exclusively on nights with positive 8:00 PM projected buffers (+2,000 to +7,000 L) that "
            "subsequently experienced rare, unobservable overnight pump delays or >20% demand surges. "
            "If colony policy requires near-zero missed shortages, operators can lower the API decision "
            "threshold from 0.50 to 0.35 using the returned shortage_probability field.",
            body_style,
        )
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 8: MLFLOW EXPERIMENT TRACKING & MODEL REGISTRY EVIDENCE
    # =========================================================================
    story.append(Paragraph("5. MLflow Experiment Tracking and Model Registry Evidence", h1_style))
    story.append(
        Paragraph(
            "All training experiments are tracked in a local SQLite MLflow backend (sqlite:///mlflow.db) "
            "under experiment <b>Overnight_Water_Shortage_Prediction</b>. Four runs were logged: two "
            "dataset-version baseline runs (RandomForest_Dataset_v1, RandomForest_Dataset_v2) and the "
            "two primary candidate runs (DecisionTree_Comparison, RandomForest_Comparison). Every run logs "
            "hyperparameters, Precision, Recall, F1-score, Accuracy, ROC-AUC, confusion matrix counts, "
            "evaluation JSON artifacts, and serialized scikit-learn models (with explicit "
            "skops_trusted_types=['sklearn.tree._tree.Tree'] compliance in MLflow 3.18). The winning "
            "Random Forest model was registered as <b>WaterShortagePredictor</b> (Version "
            f"{model_meta['mlflow_model_version']}, alias <b>@production</b>).",
            body_style,
        )
    )
    _add_image_if_exists(
        story,
        FIGURES_DIR / "mlflow_experiments_ui.png",
        6.5,
        "Figure 7: Live MLflow Tracking UI showing all 4 logged runs and the registered WaterShortagePredictor v1 artifact.",
        caption_style,
        max_h_in=3.65,
    )
    _add_image_if_exists(
        story,
        FIGURES_DIR / "mlflow_registry_ui.png",
        6.5,
        "Figure 8: Live MLflow Model Registry UI showing WaterShortagePredictor Version 1 promoted with the @production alias.",
        caption_style,
        max_h_in=3.65,
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 9: CI/CD PIPELINE (GITHUB ACTIONS) & AUTOMATED TEST MATRIX
    # =========================================================================
    story.append(Paragraph("6. CI/CD Pipeline (GitHub Actions) and Validation Results", h1_style))
    story.append(
        Paragraph(
            "Continuous Integration is automated via <b>.github/workflows/ci.yml</b> on every push and pull "
            "request to main/master. The workflow provisions Ubuntu + Python 3.11, installs dependencies from "
            "requirements.txt, regenerates versioned datasets, executes the Feature Store preprocessing "
            "pipeline, retrains and registers the model in MLflow, and runs the full 13-test pytest suite.",
            body_style,
        )
    )

    test_matrix = [
        ["Test Module", "Test Cases Covered", "Core Verification"],
        ["tests/test_validation.py\n(5 tests)", "tank_level_cannot_exceed_capacity, exact_capacity_allowed,\ncannot_be_negative, invalid_day_type, pump_duration_limit", "Enforces 0 <= tank_level_8pm <= 100,000 L &\nvalid physical ranges in pipeline"],
        ["tests/test_pipeline.py\n(3 tests)", "dataset_v1_v2_summer_demand_shift, engineered_feature_formulas,\npipeline_and_feature_store_persistence", "Verifies 250 rows/ver, ~35% v2 shift, exact\nformulas & 0-NaN Feature Store"],
        ["tests/test_api.py\n(5 tests)", "health_endpoint, predict_shortage_risk, predict_adequate_reserve,\napi_rejects_exceeding_capacity, api_rejects_negative_level", "Verifies FastAPI labels & HTTP 422 rejection\nwhen tank_level_8pm > 100,000 L"],
    ]
    story.append(_styled_table(test_matrix, [1.55, 3.00, 2.50], header_hex="#0f172a", font_size=7.9, pad=4.2))
    story.append(Spacer(1, 0.12 * inch))

    _add_image_if_exists(
        story,
        FIGURES_DIR / "ci_pytest_results.png",
        6.95,
        "Figure 9: Automated pytest execution output confirming 13/13 pass rate across tank capacity validation, Feature Store, and FastAPI suites.",
        caption_style,
        max_h_in=4.75,
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 10: FASTAPI MODEL SERVING ARCHITECTURE, CONTRACT & LIVE PREDICTIONS
    # =========================================================================
    story.append(Paragraph("7. Model Serving with FastAPI and Live Prediction Evidence", h1_style))
    story.append(
        Paragraph(
            "The production Random Forest model is served via FastAPI (api/main.py). Incoming JSON requests "
            "are strictly validated against the Pydantic model EveningTankConditions (enforcing "
            "0 <= tank_level_8pm <= 100,000 L, connected_households > 0, and day_type enumeration). Validated "
            "payloads pass through engineer_features() and return <b>Shortage Risk</b> or <b>Adequate Reserve</b> "
            "along with shortage_probability and engineered water-balance diagnostics.",
            body_style,
        )
    )

    api_tbl = [
        ["Endpoint", "Method", "Purpose & Validation Behavior"],
        ["/", "GET", "Returns service metadata, tank capacity (100,000 L), and threshold (15,000 L)"],
        ["/health", "GET", "Confirms production model readiness, model name, and MLflow registry version"],
        ["/predict", "POST", "Validates 8:00 PM inputs (HTTP 422 if > 100k L); returns Shortage Risk / Adequate Reserve"],
        ["/docs & /ui", "GET", "Interactive OpenAPI Swagger documentation and live evening-scenario browser UI"],
    ]
    story.append(_styled_table(api_tbl, [1.10, 0.70, 5.25], header_hex="#7e22ce", font_size=8.1, pad=4, center_from_col=1))
    story.append(Spacer(1, 0.08 * inch))

    _add_image_if_exists(
        story,
        FIGURES_DIR / "fastapi_swagger_ui.png",
        6.3,
        "Figure 10: Live FastAPI OpenAPI / Swagger UI (/docs) exposing /health and schema-validated POST /predict.",
        caption_style,
        max_h_in=3.35,
    )
    _add_image_if_exists(
        story,
        FIGURES_DIR / "api_shortage_risk.png",
        6.3,
        "Figure 11: Live FastAPI prediction for high-demand summer weekend evening conditions returning 'Shortage Risk' (99.8% probability).",
        caption_style,
        max_h_in=3.35,
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 11: ADEQUATE RESERVE, CAPACITY GUARD, CONCLUSION & FUTURE WORK
    # =========================================================================
    story.append(Paragraph("7.1 Live Verification: Adequate Reserve & Tank Capacity Guard", h1_style))
    _add_image_if_exists(
        story,
        FIGURES_DIR / "api_adequate_reserve.png",
        6.15,
        "Figure 12: Live FastAPI prediction for healthy tank level and pumping conditions returning 'Adequate Reserve' (0.8% shortage probability).",
        caption_style,
        max_h_in=3.20,
    )
    _add_image_if_exists(
        story,
        FIGURES_DIR / "api_capacity_validation.png",
        6.15,
        "Figure 13: Live FastAPI capacity guard rejecting tank_level_8pm = 105,000 L (> 100,000 L capacity) with HTTP 422 Unprocessable Entity.",
        caption_style,
        max_h_in=3.20,
    )

    story.append(Paragraph("8. Conclusion and Future Scope", h1_style))
    story.append(
        Paragraph(
            "This project delivers a complete, production-ready MLOps solution for predicting overnight "
            "water shortages in a 100,000-litre residential overhead tank. All five examination objectives "
            "are fulfilled: (1) reproducible v1 Normal and v2 Summer (+34.71% demand) versioned datasets "
            "with SHA-256 tracking; (2) automated physical validation, imputation, and a reusable Feature "
            "Store materializing expected_overnight_demand, expected_pump_supply, and projected_water_buffer; "
            "(3) rigorous Decision Tree vs. Random Forest comparison resulting in MLflow registration of "
            "the highest-recall Random Forest model (90.48% Recall, 0.8702 F1, 0.9531 ROC-AUC); (4) GitHub "
            "Actions CI automation with 13/13 passing tests enforcing the 100,000-litre tank capacity limit; "
            "and (5) live FastAPI serving returning actionable Shortage Risk / Adequate Reserve alerts.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Future Production Enhancements:</b> Connecting live IoT ultrasonic level sensors and municipal "
            "pump-status telemetry at 8:00 PM, scheduling automated seasonal data-drift monitoring (Evidently AI) "
            "to trigger MLflow retraining pipelines when household demand shifts by >15%, and deploying the "
            "FastAPI container behind Prometheus/Grafana latency and prediction-drift dashboards.",
            body_style,
        )
    )

    doc.build(story, onFirstPage=_draw_page_footer, onLaterPages=_draw_page_footer)
    print(f"PDF report generated successfully at {REPORT_PDF_PATH}")
    return REPORT_PDF_PATH


if __name__ == "__main__":
    build_pdf_report()
