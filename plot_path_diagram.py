"""
Path Diagram: IVs → Factors → DV
Updated for the 1000-record dataset results
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis('off')

# Colors
iv_color = '#E8F4FD'
factor_color = '#FFF3E0'
dv_color = '#E8F5E9'
sig_arrow = '#C62828'  # Red for significant negative
nonsig_arrow = '#9E9E9E'

# =============================================================================
# Draw IVs (rectangles on the left)
# =============================================================================
ivs = [
    ('LINEITEMUNITPRICE', 9.2),
    ('LINEITEMTOTALPRICE', 7.9),
    ('ESTIMATETOTALPRICE', 6.6),
    ('ESTIMATETOTALQUANTITY', 5.3),
    ('QTYPERLINEITEM', 4.0),
    ('MATERIALUNITWEIGHT', 2.7),
    ('MATERIALCENTWEIGHTCOST', 1.4),
]

for iv_name, y_pos in ivs:
    box = FancyBboxPatch((0.3, y_pos - 0.35), 3.8, 0.7,
                          boxstyle="round,pad=0.05",
                          facecolor=iv_color, edgecolor='#1976D2', linewidth=2)
    ax.add_patch(box)
    ax.text(2.2, y_pos, iv_name, ha='center', va='center', fontsize=8, fontweight='bold')

# =============================================================================
# Draw Factors (ellipses in the middle)
# =============================================================================
factor1_y = 6.0  # Order Volume
factor2_y = 3.0  # Unit Characteristics

# Factor 1 ellipse
ellipse1 = mpatches.Ellipse((8, factor1_y), 3.2, 1.8,
                             facecolor=factor_color, edgecolor='#E65100', linewidth=2)
ax.add_patch(ellipse1)
ax.text(8, factor1_y + 0.2, 'Factor 1', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(8, factor1_y - 0.25, '"Order Volume"', ha='center', va='center', fontsize=9, style='italic')

# Factor 2 ellipse
ellipse2 = mpatches.Ellipse((8, factor2_y), 3.2, 1.8,
                             facecolor=factor_color, edgecolor='#E65100', linewidth=2)
ax.add_patch(ellipse2)
ax.text(8, factor2_y + 0.2, 'Factor 2', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(8, factor2_y - 0.25, '"Unit Characteristics"', ha='center', va='center', fontsize=9, style='italic')

# =============================================================================
# Draw DV (rectangle on the right)
# =============================================================================
dv_box = FancyBboxPatch((12, 4.0), 3.2, 1.4,
                         boxstyle="round,pad=0.05",
                         facecolor=dv_color, edgecolor='#388E3C', linewidth=2)
ax.add_patch(dv_box)
ax.text(13.6, 4.7, 'MARGINACTUAL', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(13.6, 4.3, '(DV: 4% - 60%)', ha='center', va='center', fontsize=9, color='#666')

# =============================================================================
# Draw arrows: IVs → Factors (with loadings)
# =============================================================================
def draw_arrow(start, end, label, color='#333', linewidth=2, label_offset=(0, 0.15)):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth))
    mid_x = (start[0] + end[0]) / 2 + label_offset[0]
    mid_y = (start[1] + end[1]) / 2 + label_offset[1]
    ax.text(mid_x, mid_y, label, fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.9))

# Factor 1 (Order Volume) loadings - totals and quantities
draw_arrow((4.1, 7.9), (6.4, factor1_y + 0.4), '0.75', color='#1976D2', linewidth=2)  # LINEITEMTOTALPRICE
draw_arrow((4.1, 6.6), (6.4, factor1_y + 0.2), '0.86', color='#1976D2', linewidth=2.5)  # ESTIMATETOTALPRICE
draw_arrow((4.1, 5.3), (6.4, factor1_y - 0.1), '0.84', color='#1976D2', linewidth=2.5)  # ESTIMATETOTALQUANTITY
draw_arrow((4.1, 4.0), (6.4, factor1_y - 0.4), '0.76', color='#1976D2', linewidth=2)  # QTYPERLINEITEM

# Factor 2 (Unit Characteristics) loadings - unit price and weight
draw_arrow((4.1, 9.2), (6.4, factor2_y + 0.5), '1.00', color='#E65100', linewidth=3)  # LINEITEMUNITPRICE
draw_arrow((4.1, 2.7), (6.4, factor2_y - 0.3), '0.99', color='#E65100', linewidth=3)  # MATERIALUNITWEIGHT

# Weak/no loading - MATERIALCENTWEIGHTCOST
draw_arrow((4.1, 1.4), (6.4, factor2_y - 0.6), '0.02', color='#BDBDBD', linewidth=1)

# =============================================================================
# Draw arrows: Factors → DV (with regression coefficients)
# =============================================================================
# Factor 1 → DV (NOT significant)
ax.annotate('', xy=(12, 5.0), xytext=(9.6, factor1_y - 0.2),
            arrowprops=dict(arrowstyle='->', color=nonsig_arrow, lw=2, linestyle='dashed'))
ax.text(10.8, 5.8, 'β = +0.11 (ns)', fontsize=10, ha='center', va='center',
        color=nonsig_arrow,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=nonsig_arrow, alpha=0.9))

# Factor 2 → DV (SIGNIFICANT - negative!)
ax.annotate('', xy=(12, 4.4), xytext=(9.6, factor2_y + 0.3),
            arrowprops=dict(arrowstyle='->', color=sig_arrow, lw=3))
ax.text(10.8, 3.2, 'β = -1.40***', fontsize=10, ha='center', va='center', fontweight='bold',
        color=sig_arrow,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=sig_arrow, alpha=0.9))

# =============================================================================
# Labels and annotations
# =============================================================================
ax.text(2.2, 10.3, 'Independent Variables', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#1976D2')
ax.text(8, 10.3, 'Latent Factors', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#E65100')
ax.text(13.6, 10.3, 'Dependent Variable', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#388E3C')

# R-squared annotation (low!)
ax.text(13.6, 2.8, 'R² = 0.021', ha='center', va='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', edgecolor='#C62828'))
ax.text(13.6, 2.3, '(Only 2.1% explained!)', ha='center', va='center', fontsize=9, color='#C62828')

# Title
ax.text(8, 0.4, 'EFA Path Diagram: Order Characteristics → Latent Factors → Margin (n=1000)',
        ha='center', va='center', fontsize=14, fontweight='bold')

# Key finding box
finding_box = FancyBboxPatch((0.3, 10.5), 15.4, 0.4,
                              boxstyle="round,pad=0.05",
                              facecolor='#FFF8E1', edgecolor='#FF8F00', linewidth=1)
ax.add_patch(finding_box)
ax.text(8, 10.7, '⚠️  Key Finding: Higher unit price/weight → LOWER margin | Order volume has no significant effect',
        ha='center', va='center', fontsize=10, fontweight='bold', color='#E65100')

plt.tight_layout()
plt.savefig('outputs/path_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Path diagram saved to: outputs/path_diagram.png")
