"""
01_eda.py — Exploratory Data Analysis for JobFit AI

Generates 8+ publication-ready charts from synthetic data.
Run: python notebooks/01_eda.py

Charts produced:
  1. Skill frequency distribution (top 25)
  2. Class balance (shortlisted vs not)
  3. Experience distribution by class
  4. Top skills by target role (heatmap)
  5. Education level distribution
  6. Skill coverage vs shortlist outcome
  7. Text length distribution by class
  8. Skill category breakdown
  9. Role distribution
 10. Skill co-occurrence (top pairs)
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from src.data.load import load_data
from src.features.skills import (
    extract_skills,
    get_skill_category,
    SKILL_CATALOG,
    ROLE_TO_SKILLS,
)

# ── Style ─────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.1)
CHART_DIR = PROJECT_ROOT / "notebooks" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name: str):
    """Save figure to charts/ directory and close it."""
    path = CHART_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved {path.relative_to(PROJECT_ROOT)}")


# ═══════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════

print("Loading data...")
resumes, jobs, matches = load_data()
print(f"  Resumes: {len(resumes)}, Jobs: {len(jobs)}, Matches: {len(matches)}")
print(f"  Shortlisted: {matches['is_shortlisted'].sum()} / {len(matches)} "
      f"({matches['is_shortlisted'].mean():.1%})")
print()


# ═══════════════════════════════════════════════════════════════
#  CHART 1 — Skill Frequency Distribution (Top 25)
# ═══════════════════════════════════════════════════════════════

print("Chart 1: Skill frequency...")
all_skills_flat = []
for skill_list in resumes["skills"]:
    if isinstance(skill_list, str):
        skill_list = eval(skill_list)
    all_skills_flat.extend(skill_list)

skill_counts = Counter(all_skills_flat)
top_25 = skill_counts.most_common(25)

fig, ax = plt.subplots(figsize=(12, 7))
skills_names = [s[0] for s in top_25]
skills_freqs = [s[1] for s in top_25]
bars = ax.barh(skills_names[::-1], skills_freqs[::-1], color=sns.color_palette("viridis", len(top_25)))
ax.set_xlabel("Frequency across resumes")
ax.set_title("Top 25 Skills by Frequency", fontsize=16, fontweight="bold")
for bar, val in zip(bars, skills_freqs[::-1]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=9)
save_fig(fig, "01_skill_frequency")


# ═══════════════════════════════════════════════════════════════
#  CHART 2 — Class Balance
# ═══════════════════════════════════════════════════════════════

print("Chart 2: Class balance...")
fig, ax = plt.subplots(figsize=(8, 5))
class_counts = matches["is_shortlisted"].value_counts().sort_index()
colors = ["#e74c3c", "#2ecc71"]
bars = ax.bar(["Not Shortlisted (0)", "Shortlisted (1)"],
              class_counts.values, color=colors, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f"{val}\n({val/len(matches):.1%})", ha="center", fontsize=12)
ax.set_ylabel("Count")
ax.set_title("Class Balance: Shortlisted vs Not", fontsize=16, fontweight="bold")
save_fig(fig, "02_class_balance")


# ═══════════════════════════════════════════════════════════════
#  CHART 3 — Experience Distribution by Class
# ═══════════════════════════════════════════════════════════════

print("Chart 3: Experience distribution...")
merged = matches.merge(resumes, left_on="resume_id", right_on="resume_id")

fig, ax = plt.subplots(figsize=(10, 5))
for label, color in [(0, "#e74c3c"), (1, "#2ecc71")]:
    subset = merged[merged["is_shortlisted"] == label]
    ax.hist(subset["experience_years"], bins=20, alpha=0.6, color=color,
            label=f"{'Shortlisted' if label else 'Not Shortlisted'}", edgecolor="white")
ax.set_xlabel("Experience (years)")
ax.set_ylabel("Count")
ax.set_title("Experience Distribution by Shortlist Outcome", fontsize=16, fontweight="bold")
ax.legend()
save_fig(fig, "03_experience_distribution")


# ═══════════════════════════════════════════════════════════════
#  CHART 4 — Top Skills by Role (Heatmap)
# ═══════════════════════════════════════════════════════════════

print("Chart 4: Skills-by-role heatmap...")
role_skill_counts = defaultdict(Counter)
for _, row in resumes.iterrows():
    role = row["target_role"]
    skill_list = row["skills"] if isinstance(row["skills"], list) else eval(row["skills"])
    for s in skill_list:
        role_skill_counts[role][s] += 1

# Get top 15 skills overall
top_15_skills = [s for s, _ in skill_counts.most_common(15)]
roles = sorted(role_skill_counts.keys())

heatmap_data = pd.DataFrame(0, index=roles, columns=top_15_skills)
for role in roles:
    for skill in top_15_skills:
        heatmap_data.loc[role, skill] = role_skill_counts[role].get(skill, 0)

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Count"})
ax.set_title("Top 15 Skills × Target Role", fontsize=16, fontweight="bold")
ax.set_xlabel("Skill")
ax.set_ylabel("Target Role")
plt.xticks(rotation=45, ha="right")
save_fig(fig, "04_skills_by_role_heatmap")


# ═══════════════════════════════════════════════════════════════
#  CHART 5 — Education Level Distribution
# ═══════════════════════════════════════════════════════════════

print("Chart 5: Education level...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall
edu_counts = resumes["education_level"].value_counts()
axes[0].pie(edu_counts, labels=edu_counts.index, autopct="%1.1f%%",
            colors=sns.color_palette("Set2"), startangle=140)
axes[0].set_title("Education Level Distribution", fontsize=14, fontweight="bold")

# By shortlist outcome
merged_edu = merged.groupby(["education_level", "is_shortlisted"]).size().unstack(fill_value=0)
merged_edu.plot(kind="bar", ax=axes[1], color=["#e74c3c", "#2ecc71"], edgecolor="white")
axes[1].set_title("Education × Shortlist Outcome", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Education Level")
axes[1].set_ylabel("Count")
axes[1].legend(["Not Shortlisted", "Shortlisted"])
axes[1].tick_params(axis="x", rotation=45)
fig.tight_layout()
save_fig(fig, "05_education_distribution")


# ═══════════════════════════════════════════════════════════════
#  CHART 6 — Skill Coverage vs Shortlist Outcome
# ═══════════════════════════════════════════════════════════════

print("Chart 6: Skill coverage vs outcome...")
fig, ax = plt.subplots(figsize=(10, 6))
for label, color, marker in [(0, "#e74c3c", "o"), (1, "#2ecc71", "^")]:
    subset = matches[matches["is_shortlisted"] == label]
    ax.scatter(subset["skill_coverage"], subset.index,
               alpha=0.3, c=color, s=15, marker=marker,
               label=f"{'Shortlisted' if label else 'Not Shortlisted'}")
ax.set_xlabel("Skill Coverage Ratio")
ax.set_ylabel("Match Index")
ax.set_title("Skill Coverage vs Shortlist Outcome", fontsize=16, fontweight="bold")
ax.legend(markerscale=3)
ax.axvline(x=0.6, color="orange", linestyle="--", alpha=0.7, label="60% threshold")
ax.axvline(x=0.4, color="gray", linestyle="--", alpha=0.5, label="40% threshold")
save_fig(fig, "06_coverage_vs_outcome")


# ═══════════════════════════════════════════════════════════════
#  CHART 7 — Text Length Distribution
# ═══════════════════════════════════════════════════════════════

print("Chart 7: Text length distribution...")
resumes["text_length"] = resumes["raw_text"].str.len()
resumes["word_count"] = resumes["raw_text"].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(resumes["text_length"], bins=30, color="#3498db", edgecolor="white", alpha=0.8)
axes[0].set_xlabel("Character Count")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Resume Text Length Distribution", fontsize=14, fontweight="bold")

axes[1].hist(resumes["word_count"], bins=30, color="#9b59b6", edgecolor="white", alpha=0.8)
axes[1].set_xlabel("Word Count")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Resume Word Count Distribution", fontsize=14, fontweight="bold")

fig.tight_layout()
save_fig(fig, "07_text_length_distribution")


# ═══════════════════════════════════════════════════════════════
#  CHART 8 — Skill Category Breakdown
# ═══════════════════════════════════════════════════════════════

print("Chart 8: Skill category breakdown...")
category_counts = Counter()
for skill in all_skills_flat:
    cat = get_skill_category(skill)
    category_counts[cat] += 1

fig, ax = plt.subplots(figsize=(10, 6))
cats = sorted(category_counts.keys())
vals = [category_counts[c] for c in cats]
colors = sns.color_palette("husl", len(cats))
bars = ax.bar(cats, vals, color=colors, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(val), ha="center", fontsize=10)
ax.set_xlabel("Skill Category")
ax.set_ylabel("Total Mentions")
ax.set_title("Skill Category Breakdown Across All Resumes", fontsize=16, fontweight="bold")
plt.xticks(rotation=30, ha="right")
save_fig(fig, "08_skill_categories")


# ═══════════════════════════════════════════════════════════════
#  CHART 9 — Role Distribution
# ═══════════════════════════════════════════════════════════════

print("Chart 9: Role distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
role_counts = resumes["target_role"].value_counts()
role_counts.plot(kind="barh", ax=ax, color=sns.color_palette("coolwarm", len(role_counts)))
ax.set_xlabel("Number of Resumes")
ax.set_title("Resume Distribution by Target Role", fontsize=16, fontweight="bold")
for i, (val, name) in enumerate(zip(role_counts.values, role_counts.index)):
    ax.text(val + 0.5, i, str(val), va="center", fontsize=10)
save_fig(fig, "09_role_distribution")


# ═══════════════════════════════════════════════════════════════
#  CHART 10 — Coverage Ratio Distribution (Histogram)
# ═══════════════════════════════════════════════════════════════

print("Chart 10: Coverage ratio distribution...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(matches["skill_coverage"], bins=25, color="#1abc9c", edgecolor="white", alpha=0.85)
ax.axvline(x=matches["skill_coverage"].mean(), color="red", linestyle="--",
           label=f"Mean: {matches['skill_coverage'].mean():.2f}")
ax.axvline(x=0.6, color="orange", linestyle="--", alpha=0.7, label="60% threshold")
ax.set_xlabel("Skill Coverage Ratio")
ax.set_ylabel("Count")
ax.set_title("Distribution of Skill Coverage Across All Matches", fontsize=16, fontweight="bold")
ax.legend()
save_fig(fig, "10_coverage_distribution")


# ═══════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"EDA Complete — 10 charts saved to {CHART_DIR}")
print(f"{'='*60}")
print(f"\nDataset Summary:")
print(f"  Resumes:   {len(resumes):>5}")
print(f"  Jobs:      {len(jobs):>5}")
print(f"  Matches:   {len(matches):>5}")
print(f"  Shortlisted:  {matches['is_shortlisted'].sum():>5} ({matches['is_shortlisted'].mean():.1%})")
print(f"  Unique skills: {len(skill_counts)}")
print(f"  Avg skills/resume: {np.mean([len(s) if isinstance(s, list) else len(eval(s)) for s in resumes['skills']]):.1f}")
print(f"  Coverage range: [{matches['skill_coverage'].min():.2f}, {matches['skill_coverage'].max():.2f}]")
print(f"  Avg text length: {resumes['text_length'].mean():.0f} chars, {resumes['word_count'].mean():.0f} words")
