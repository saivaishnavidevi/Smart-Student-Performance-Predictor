"""
generate_dataset.py
-------------------
Generates a synthetic dataset of 300 student records for the
Smart Student Performance Predictor project.

DISCLAIMER: This dataset is entirely synthetic. The values and
relationships are simulated for demonstration purposes only and
do not represent real students or establish real-world predictive
performance.
"""

import os
import numpy as np
import pandas as pd

# ── Reproducibility ────────────────────────────────────────────
RANDOM_SEED = 42
N_SAMPLES = 300
rng = np.random.default_rng(RANDOM_SEED)

# ── Synthetic feature generation ───────────────────────────────
# study_hours: weekly study hours, range 1–20 h
study_hours = rng.uniform(1.0, 20.0, N_SAMPLES).round(1)

# attendance_pct: class attendance percentage, range 40–100 %
attendance_pct = rng.uniform(40.0, 100.0, N_SAMPLES).round(1)

# prev_exam_score: score from the previous exam, range 30–100
prev_exam_score = rng.uniform(30.0, 100.0, N_SAMPLES).round(1)

# assignment_score: average assignment score, range 20–100
assignment_score = rng.uniform(20.0, 100.0, N_SAMPLES).round(1)

# ── Target: final_score ─────────────────────────────────────────
# Linear combination with plausible weights:
#   study_hours      contributes ~1.5 pts per hour
#   attendance_pct   contributes ~0.2 pts per %
#   prev_exam_score  contributes ~0.3 pts (prior knowledge)
#   assignment_score contributes ~0.25 pts (consistent effort)
#   intercept        baseline offset
#   noise            Gaussian noise (std=5) simulates real-world variance
noise = rng.normal(loc=0.0, scale=5.0, size=N_SAMPLES)

final_score_raw = (
    1.5  * study_hours
    + 0.20 * attendance_pct
    + 0.30 * prev_exam_score
    + 0.25 * assignment_score
    + 5.0  # intercept
    + noise
)

# Clip to valid exam score range and round to one decimal place
final_score = np.clip(final_score_raw, 0.0, 100.0).round(1)

# ── Assemble DataFrame with columns in specified order ──────────
df = pd.DataFrame({
    "study_hours":      study_hours,
    "attendance_pct":   attendance_pct,
    "prev_exam_score":  prev_exam_score,
    "assignment_score": assignment_score,
    "final_score":      final_score,
})

# ── Save to CSV ─────────────────────────────────────────────────
output_dir = os.path.dirname(os.path.abspath(__file__))  # same as data/
output_path = os.path.join(output_dir, "students.csv")

os.makedirs(output_dir, exist_ok=True)
df.to_csv(output_path, index=False)

# ── Verification printout ───────────────────────────────────────
print(f"Dataset shape   : {df.shape}")
print(f"Columns         : {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nSaved to        : {output_path}")
