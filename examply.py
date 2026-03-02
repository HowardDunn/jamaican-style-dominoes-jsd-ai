import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. Define the data points (X: Technical Domain, Y: Hierarchy)
# X-axis: -1.0 (Firmware/Embedded) to 1.0 (Backend/Cloud)
# Y-axis: 0.0 (Individual Contributor) to 1.0 (Executive)

team_data = {
    "Pattabi\n(Vice President)": {"coords": (0.0, 0.90), "color": "#E6A119"},
    "Kris\n(EM - Backend)": {"coords": (0.6, 0.50), "color": "#9b59b6"},
    "Valentin\n(Backend Engineer)": {"coords": (0.8, 0.15), "color": "#8e44ad"},
    "Damien\n(EM - Embedded)": {"coords": (-0.6, 0.50), "color": "#1abc9c"},
    "William\n(Firmware Engineer)": {"coords": (-0.8, 0.15), "color": "#16a085"}
}

# 2. Set up the plot
plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.set_facecolor('#f8f9fa')
plt.grid(color='white', linestyle='-', linewidth=1.5)

# 3. Plot each person
for name, info in team_data.items():
    x, y = info["coords"]
    plt.scatter(x, y, color=info["color"], s=150, zorder=5, edgecolor='black')
    
    # Add text labels with slight offsets
    if "Pattabi" in name:
        plt.text(x, y + 0.05, name, fontsize=10, ha='center', va='bottom', fontweight='bold')
    else:
        plt.text(x, y - 0.04, name, fontsize=10, ha='center', va='top', fontweight='bold')

# 4. Draw clusters (Ellipses to show learned association)
# Embedded Cluster
embedded_ellipse = patches.Ellipse((-0.7, 0.325), 0.6, 0.55, angle=-30, 
                                   alpha=0.2, color='#1abc9c', zorder=1)
ax.add_patch(embedded_ellipse)
plt.text(-0.95, 0.4, "Embedded\nCluster", color='#16a085', fontweight='bold', fontsize=11)

# Backend Cluster
backend_ellipse = patches.Ellipse((0.7, 0.325), 0.6, 0.55, angle=30, 
                                  alpha=0.2, color='#9b59b6', zorder=1)
ax.add_patch(backend_ellipse)
plt.text(0.95, 0.4, "Backend\nCluster", color='#8e44ad', fontweight='bold', ha='right', fontsize=11)

# 5. Format axes
plt.xlim(-1.2, 1.2)
plt.ylim(0, 1.1)

plt.axvline(0, color='gray', linestyle='--', linewidth=1, zorder=2) # Center line

plt.xlabel('Technical Domain / Focus \n<-- Embedded / Firmware  |  Backend / Cloud -->', 
           fontsize=12, labelpad=10)
plt.ylabel('Organizational Hierarchy\n<-- Individual Contributor  |  Executive Leader -->', 
           fontsize=12, labelpad=10)

plt.title('Self-Attention: Visualizing Learned Team Clusters', fontsize=16, pad=20, fontweight='bold')

# 6. Clean up borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 7. Show and/or save the plot
plt.tight_layout()
# plt.savefig('team_cluster_graph.png', dpi=300) # Uncomment to save as a high-res image
plt.show()