"""results/ EDA 시각화 재현 — pandas + matplotlib (TensorFlow 불필요)

NSMC 데이터(ratings_train.csv, TSV: id/document/label)를 준비한 뒤 실행:
    python make_figures.py path/to/ratings_train.csv
NSMC 원본: https://github.com/e9t/nsmc
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"      # 한글 라벨 (Windows)
plt.rcParams["axes.unicode_minus"] = False
GREEN, RED = "#76B900", "#E15759"

path = sys.argv[1] if len(sys.argv) > 1 else "ratings_train.csv"
R = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(R, exist_ok=True)

d = pd.read_csv(path, sep="\t")
d.columns = [c.lower() for c in d.columns]
doc = "document" if "document" in d.columns else d.columns[1]
lab = "label" if "label" in d.columns else d.columns[-1]
d = d.dropna(subset=[doc])

vc = d[lab].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["부정(0)", "긍정(1)"], [vc.get(0, 0), vc.get(1, 0)], color=[RED, GREEN])
for i, v in enumerate([vc.get(0, 0), vc.get(1, 0)]): ax.text(i, v, f"{v:,}", ha="center", va="bottom")
ax.set(title=f"NSMC 학습셋 라벨 분포 (n={len(d):,})", ylabel="리뷰 수")
fig.tight_layout(); fig.savefig(f"{R}/label_distribution.png", dpi=120); plt.close(fig)

d["len"] = d[doc].astype(str).str.len()
fig, ax = plt.subplots(figsize=(6.5, 4))
for l, c, name in [(0, RED, "부정"), (1, GREEN, "긍정")]:
    ax.hist(d[d[lab] == l]["len"].clip(upper=150), bins=40, alpha=.6, color=c, label=name)
ax.set(xlabel="리뷰 길이 (문자 수, 150+ 클리핑)", ylabel="빈도", title="리뷰 길이 분포 (긍/부정)"); ax.legend()
fig.tight_layout(); fig.savefig(f"{R}/review_length.png", dpi=120); plt.close(fig)
print("saved figures to", R)
