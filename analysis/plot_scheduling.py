import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ── Load real data ──
df = pd.read_csv(
    "/data3/projects/2025_Assembly/eyh/c_elegans/assemblies/bucket_complexity/bucket_complexity.tsv",
    sep='\t'
)
scores = df.sort_values('cpu_score', ascending=False)['cpu_score'].tolist()

def assign_cpu(score, rank, top_n=5):
    if rank < top_n or score > 12: return (32,   '#1A5E9A', '#E6F1FB')
    elif score > 9:                 return (20,   '#1D9E75', '#E1F5EE')
    elif score > 6:                 return (12,   '#6BBFA0', '#085041')
    else:                           return ('≤8', '#B4B2A9', '#444441')

assignments = [assign_cpu(s, i) for i, s in enumerate(scores)]

COLS = 10
ROWS = 5
pad  = 0.10
cell = 1.0
total_w = COLS * (cell + pad) - pad
total_h = ROWS * (cell + pad) - pad

# ── Reserve bottom space for legend inside figure ──
fig = plt.figure(figsize=(3.8, 3.6))
ax  = fig.add_axes([0.12, 0.28, 0.85, 0.65])  # [left, bottom, width, height]
ax.set_aspect('equal')
ax.axis('off')

# ── Draw grid ──
for idx, (score, (cpu, facecolor, textcolor)) in enumerate(zip(scores, assignments)):
    row = idx // COLS
    col = idx  % COLS
    x   = col * (cell + pad)
    y   = (ROWS - 1 - row) * (cell + pad)

    rect = mpatches.FancyBboxPatch(
        (x, y), cell, cell,
        boxstyle="round,pad=0.06",
        facecolor=facecolor, edgecolor='white',
        linewidth=1.5, zorder=2
    )
    ax.add_patch(rect)
    ax.text(x + cell/2, y + cell/2, str(cpu),
            ha='center', va='center',
            fontsize=7, fontweight='bold',
            color=textcolor, zorder=3)

# ── Y-axis arrow + label ──
ax.annotate('', xy=(-0.5, total_h), xytext=(-0.5, 0),
            arrowprops=dict(arrowstyle='->', color='#888780', lw=1.0))
ax.text(-0.85, total_h/2, 'Complexity', ha='center', va='center',
        fontsize=6.5, color='#5F5E5A', rotation=90)

# ── Bucket labels ──
ax.text(0,       -0.3, 'Bucket 1',  ha='left',  va='top', fontsize=6, color='#5F5E5A')
ax.text(total_w, -0.3, 'Bucket 50', ha='right', va='top', fontsize=6, color='#5F5E5A')

ax.set_xlim(-1.1, total_w + 0.2)
ax.set_ylim(-0.6, total_h + 0.3)

# ── Title ──
ax.set_title('K-mer complexity-based adaptive CPU scheduling\n(K=50 buckets, C. elegans)',
             fontsize=7.5, fontweight='bold', color='#2C2C2A', pad=5, loc='left')

# ── Manual legend drawn in figure coordinates (fully inside canvas) ──
legend_ax = fig.add_axes([0.08, 0.02, 0.90, 0.22])
legend_ax.axis('off')
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)

items = [
    ('#1A5E9A', '32 CPUs  (top 10%, score > 12)'),
    ('#1D9E75', '20 CPUs  (score 9–12)'),
    ('#6BBFA0', '12 CPUs  (score 6–9)'),
    ('#B4B2A9', '4–8 CPUs  (score < 6)'),
]

# Two columns, two rows
positions = [(0.02, 0.55), (0.52, 0.55), (0.02, 0.05), (0.52, 0.05)]
for (x, y), (color, label) in zip(positions, items):
    legend_ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), 0.04, 0.35,
        boxstyle="round,pad=0.01",
        facecolor=color, edgecolor='white',
        linewidth=0.8, transform=legend_ax.transAxes
    ))
    legend_ax.text(x + 0.06, y + 0.17, label,
                   ha='left', va='center', fontsize=6,
                   color='#2C2C2A', transform=legend_ax.transAxes)

plt.savefig('cpu_scheduling_grid.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('cpu_scheduling_grid.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print("Saved: cpu_scheduling_grid.pdf / .png")