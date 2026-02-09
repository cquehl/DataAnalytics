"""
Path Diagram: IVs → 3 Factors → Price Tier (DV)
Updated for the new 3-factor EFA results from efa_price_tier.py

Factor Structure (Varimax rotation):
- Factor 1: Unit Characteristics (LINEITEMUNITPRICE, MATERIALUNITWEIGHT)
- Factor 2: Order Value (LINEITEMTOTALPRICE, ESTIMATETOTALPRICE)
- Factor 3: Order Quantity (ESTIMATETOTALQUANTITY, QTYPERLINEITEM)

Relationship to DV (PRICETIERKEY):
- All 3 factors significantly discriminate between tiers (ANOVA)
- Factor 1: η² = 0.138 (Medium effect)
- Factor 2: η² = 0.015 (Small effect)
- Factor 3: η² = 0.070 (Medium effect)
"""
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')

# Colors
iv_color = '#E8F4FD'
factor_color = '#FFF3E0'
dv_color = '#E8F5E9'
strong_arrow = '#1565C0'  # Blue for strong loadings
medium_arrow = '#7986CB'  # Light blue for medium loadings
weak_arrow = '#BDBDBD'    # Gray for weak/no loading
sig_high = '#2E7D32'      # Green for large effect
sig_med = '#F57C00'       # Orange for medium effect
sig_low = '#9E9E9E'       # Gray for small effect

# =============================================================================
# Draw IVs (rectangles on the left)
# =============================================================================
ivs = [
    ('LINEITEMUNITPRICE', 10.0),
    ('MATERIALUNITWEIGHT', 8.5),
    ('LINEITEMTOTALPRICE', 6.5),
    ('ESTIMATETOTALPRICE', 5.0),
    ('ESTIMATETOTALQUANTITY', 3.0),
    ('QTYPERLINEITEM', 1.5),
]

for iv_name, y_pos in ivs:
    box = FancyBboxPatch((0.3, y_pos - 0.35), 3.8, 0.7,
                          boxstyle="round,pad=0.05",
                          facecolor=iv_color, edgecolor='#1976D2', linewidth=2)
    ax.add_patch(box)
    ax.text(2.2, y_pos, iv_name, ha='center', va='center', fontsize=8, fontweight='bold')

# MATERIALCENTWEIGHTCOST (weak loader - shown separately)
box = FancyBboxPatch((0.3, -0.35), 3.8, 0.7,
                      boxstyle="round,pad=0.05",
                      facecolor='#F5F5F5', edgecolor='#BDBDBD', linewidth=1)
ax.add_patch(box)
ax.text(2.2, 0, 'MATERIALCENTWEIGHTCOST', ha='center', va='center', fontsize=7, color='#757575')

# =============================================================================
# Draw Factors (ellipses in the middle)
# =============================================================================
factor1_y = 9.25  # Unit Characteristics
factor2_y = 5.75  # Order Value
factor3_y = 2.25  # Order Quantity

# Factor 1 ellipse
ellipse1 = mpatches.Ellipse((9, factor1_y), 3.4, 1.8,
                             facecolor=factor_color, edgecolor='#E65100', linewidth=2)
ax.add_patch(ellipse1)
ax.text(9, factor1_y + 0.25, 'Factor 1', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(9, factor1_y - 0.25, '"Unit Characteristics"', ha='center', va='center', fontsize=9, style='italic')

# Factor 2 ellipse
ellipse2 = mpatches.Ellipse((9, factor2_y), 3.4, 1.8,
                             facecolor=factor_color, edgecolor='#E65100', linewidth=2)
ax.add_patch(ellipse2)
ax.text(9, factor2_y + 0.25, 'Factor 2', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(9, factor2_y - 0.25, '"Order Value"', ha='center', va='center', fontsize=9, style='italic')

# Factor 3 ellipse
ellipse3 = mpatches.Ellipse((9, factor3_y), 3.4, 1.8,
                             facecolor=factor_color, edgecolor='#E65100', linewidth=2)
ax.add_patch(ellipse3)
ax.text(9, factor3_y + 0.25, 'Factor 3', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(9, factor3_y - 0.25, '"Order Quantity"', ha='center', va='center', fontsize=9, style='italic')

# =============================================================================
# Draw DV (rectangle on the right)
# =============================================================================
dv_box = FancyBboxPatch((13.5, 4.5), 4, 2.5,
                         boxstyle="round,pad=0.05",
                         facecolor=dv_color, edgecolor='#388E3C', linewidth=2)
ax.add_patch(dv_box)
ax.text(15.5, 5.75, 'PRICETIERKEY', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(15.5, 5.25, '(Categorical DV)', ha='center', va='center', fontsize=9, color='#666')
ax.text(15.5, 4.8, '5 Tiers Analyzed', ha='center', va='center', fontsize=8, color='#888')

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

# Factor 1 (Unit Characteristics) loadings
draw_arrow((4.1, 10.0), (7.3, factor1_y + 0.3), '0.94', color=strong_arrow, linewidth=2.5)  # LINEITEMUNITPRICE
draw_arrow((4.1, 8.5), (7.3, factor1_y - 0.3), '0.98', color=strong_arrow, linewidth=3)    # MATERIALUNITWEIGHT

# Factor 2 (Order Value) loadings
draw_arrow((4.1, 6.5), (7.3, factor2_y + 0.3), '0.97', color=strong_arrow, linewidth=3)    # LINEITEMTOTALPRICE
draw_arrow((4.1, 5.0), (7.3, factor2_y - 0.3), '0.70', color=medium_arrow, linewidth=2)    # ESTIMATETOTALPRICE

# Factor 3 (Order Quantity) loadings
draw_arrow((4.1, 3.0), (7.3, factor3_y + 0.3), '1.00', color=strong_arrow, linewidth=3)    # ESTIMATETOTALQUANTITY
draw_arrow((4.1, 1.5), (7.3, factor3_y - 0.3), '0.60', color=medium_arrow, linewidth=2)    # QTYPERLINEITEM

# Cross-loadings (weaker, shown as dashed)
ax.annotate('', xy=(7.3, factor2_y + 0.5), xytext=(4.1, 10.0),
            arrowprops=dict(arrowstyle='->', color=weak_arrow, lw=1, linestyle='dashed'))
ax.text(5.5, 8.5, '0.21', fontsize=7, ha='center', va='center', color='#999')

ax.annotate('', xy=(7.3, factor3_y + 0.5), xytext=(4.1, 5.0),
            arrowprops=dict(arrowstyle='->', color=weak_arrow, lw=1, linestyle='dashed'))
ax.text(5.5, 3.9, '0.32', fontsize=7, ha='center', va='center', color='#999')

# MATERIALCENTWEIGHTCOST - no significant loading (very weak to all factors)
ax.annotate('', xy=(7.3, factor1_y - 0.7), xytext=(4.1, 0),
            arrowprops=dict(arrowstyle='->', color='#E0E0E0', lw=1, linestyle='dotted'))
ax.text(5.0, 3.5, '<0.05', fontsize=7, ha='center', va='center', color='#BDBDBD')

# =============================================================================
# Draw arrows: Factors → DV (with ANOVA effect sizes)
# =============================================================================
# Factor 1 → DV (MEDIUM effect, significant)
ax.annotate('', xy=(13.5, 6.2), xytext=(10.7, factor1_y - 0.4),
            arrowprops=dict(arrowstyle='->', color=sig_high, lw=3))
ax.text(12.4, 8.2, 'η² = 0.138***', fontsize=10, ha='center', va='center', fontweight='bold',
        color=sig_high,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=sig_high, alpha=0.9))
ax.text(12.4, 7.7, '(Medium)', fontsize=8, ha='center', va='center', color=sig_high)

# Factor 2 → DV (SMALL effect, significant)
ax.annotate('', xy=(13.5, 5.75), xytext=(10.7, factor2_y),
            arrowprops=dict(arrowstyle='->', color=sig_low, lw=2, linestyle='dashed'))
ax.text(12.1, 5.75, 'η² = 0.015***', fontsize=9, ha='center', va='center',
        color=sig_low,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=sig_low, alpha=0.9))

# Factor 3 → DV (MEDIUM effect, significant)
ax.annotate('', xy=(13.5, 5.3), xytext=(10.7, factor3_y + 0.4),
            arrowprops=dict(arrowstyle='->', color=sig_med, lw=3))
ax.text(12.4, 3.3, 'η² = 0.070***', fontsize=10, ha='center', va='center', fontweight='bold',
        color=sig_med,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=sig_med, alpha=0.9))
ax.text(12.4, 2.8, '(Medium)', fontsize=8, ha='center', va='center', color=sig_med)

# =============================================================================
# Labels and annotations
# =============================================================================
ax.text(2.2, 11.3, 'Independent Variables', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#1976D2')
ax.text(9, 11.3, 'Latent Factors', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#E65100')
ax.text(15.5, 11.3, 'Dependent Variable', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#388E3C')

# Legend
legend_box = FancyBboxPatch((13.2, 1.0), 4.5, 2.2,
                             boxstyle="round,pad=0.05",
                             facecolor='#FAFAFA', edgecolor='#E0E0E0', linewidth=1)
ax.add_patch(legend_box)
ax.text(15.45, 2.95, 'Effect Size Legend', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(15.45, 2.5, '● Large: η² > 0.14', ha='center', va='center', fontsize=8, color=sig_high)
ax.text(15.45, 2.1, '● Medium: η² > 0.06', ha='center', va='center', fontsize=8, color=sig_med)
ax.text(15.45, 1.7, '● Small: η² ≤ 0.06', ha='center', va='center', fontsize=8, color=sig_low)
ax.text(15.45, 1.3, '*** p < 0.001', ha='center', va='center', fontsize=8, color='#666')

# Title
ax.text(9, -0.8, 'EFA Path Diagram: Order Characteristics → 3 Latent Factors → Price Tier',
        ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(9, -1.2, '(Sabel Steel + Coosa, Aug 2025+, n=2,024)',
        ha='center', va='center', fontsize=10, color='#666')

# Key finding box
finding_box = FancyBboxPatch((0.3, 11.5), 17.4, 0.4,
                              boxstyle="round,pad=0.05",
                              facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=1)
ax.add_patch(finding_box)
ax.text(9, 11.7, '✓ All 3 factors significantly discriminate between price tiers — Unit Characteristics has the strongest effect',
        ha='center', va='center', fontsize=10, fontweight='bold', color='#2E7D32')

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/path_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Path diagram saved to: outputs/path_diagram.png")
