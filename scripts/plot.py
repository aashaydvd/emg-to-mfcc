import numpy as np
import matplotlib.pyplot as plt
import json
# -------------------------
# Your data (paste as-is)
# -------------------------
with open("results_server/evaluation_summary.json", 'r') as f:
    data = json.load(f)



# -------------------------
# Extract values
# -------------------------
blocks = ["Block1-Initial", "Block3-Eval1", "Block5-Eval2", "Block7-Eval3"]

baseline = []
invariant = []

for spk in data:
    b_vals = []
    i_vals = []
    
    for blk in blocks:
        b_vals.append(data[spk][blk]["baseline_mcd"])
        i_vals.append(data[spk][blk]["invariant_mcd"])
    
    baseline.append(b_vals)
    invariant.append(i_vals)

baseline = np.array(baseline)   # shape (8, 4)
invariant = np.array(invariant)


# -------------------------
# Compute mean + std
# -------------------------
b_mean = baseline.mean(axis=0)
b_std  = baseline.std(axis=0)

i_mean = invariant.mean(axis=0)
i_std  = invariant.std(axis=0)


# -------------------------
# Plot
# -------------------------
x = np.arange(len(blocks))

plt.figure(figsize=(8,5))

# Baseline
plt.plot(x, b_mean, marker='o', linestyle='--', label="Baseline")
plt.fill_between(x, b_mean - b_std, b_mean + b_std, alpha=0.2)

# Invariant
plt.plot(x, i_mean, marker='o', linestyle='-', label="Invariant")
plt.fill_between(x, i_mean - i_std, i_mean + i_std, alpha=0.2)

# Labels
plt.xticks(x, ["B1", "B3", "B5", "B7"])
plt.xlabel("Blocks")
plt.ylabel("MCD (↓ better)")
plt.title("Baseline vs Invariant MCD Across Speakers")
plt.legend()

plt.tight_layout()

plt.savefig("visualisations/mcd_comparison.png", dpi=300)
plt.close()