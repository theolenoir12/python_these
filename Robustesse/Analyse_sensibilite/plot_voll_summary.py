"""
plot_voll_summary.py -- UNIFIED summary of the EMS sensitivity analyses.
============================================================================
Publication-ready figures designed to sit next to the (LaTeX/TikZ) workflow
figure of the thesis: Computer Modern serif typography, restrained palette,
normal-weight titles, light spines, vector PDF with embedded editable fonts.

Runs NO simulation: only re-reads results_meso/*.txt and applies the unified
indicator of voll_common (total_cost = degradation cost + financial cost of the
unserved load, valued with the VOLL defined there).

EDITION FOR HIGH PROFESSIONALISM AND READABILITY IN SCIENTIFIC PUBLICATIONS:
Increased font sizes, adjusted spacing, slightly thicker lines.
HIGHLIGHTING: Top 3 ranks are bold, Rank 1 has an asterisk (*).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive backend for scripting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Supposing voll_common is reachable
try:
    import voll_common as V
    OUT = V.MESO_DIR
except ImportError:
    # Fallback for testing purposes if voll_common is not present
    print("Warning: voll_common not found. Using dummy data/paths.")
    class DummyV:
        MESO_DIR = "."
        EMS_ORDER = ["RuleBased", "Optimization", "Predictive", "AI-Enhanced", "Legacy_A", "Legacy_B"]
        VOLL_TIERS = [(0.01, 10), (0.05, 50), (None, 100)]
        E_REF_KWH = 1e6
        HORIZON_Y = 25
        @staticmethod
        def total_cost_keur(lpsp, deg): return lpsp * 10 + deg
        @staticmethod
        def cost_lpsp_keur(lpsp): return lpsp * 10
        @staticmethod
        def voll_eur_per_kwh(lpsp): return 50
        @staticmethod
        def parse_soh(): return {"bias": [(0.01, 5, 50)], "sigma": [(0.01, 5, 50)]}
        @staticmethod
        def build_cases():
            # Generate slightly more varied dummy data to see ranking effect
            ems = DummyV.EMS_ORDER
            return [
                ("Nominal", None, {e: (1.0 + i*0.1, 50.0 - i*2) for i, e in enumerate(ems)}),
                ("High Load", None, {e: (2.0 + i*0.2, 60.0 - i*1) for i, e in enumerate(ems)}),
                ("Low PV", None, {e: (0.5 + i*0.05, 40.0 - i*3) for i, e in enumerate(ems)})
            ]
    V = DummyV
    OUT = "."

# --- Typography tokens (Computer Modern via mathtext) ---
# Unified spelling as cmr10 lacks Euro symbol. Kept for ASCII compatibility.
KEUR   = "kEUR"
EURKWH = "EUR/kWh"
SIG    = r"$\sigma$"      # Computer Modern sigma
TIMES  = r"$\times$"      # Computer Modern multiplication sign

# --- Restrained palette (muted, professional) ---
C_DEG   = "#4C72B0"       # degradation (muted blue)
C_LPSP  = "#DD8452"       # unserved-energy / LPSP cost (muted orange)
C_LINE  = "#7D5BA6"       # primary line (muted purple)
C_LINE2 = "#55A868"       # secondary line (muted green) - Changed from blue for contrast with C_DEG
C_RED   = "#C44E52"       # VOLL curve / Nominal marker (muted red)
C_BOX   = "#9BB7D4"       # uniform box fill (soft blue)
CMAP    = "RdYlGn_r"      # rank heatmap (green = best)


def set_pub_style():
    """
    Sets a hyper-professional matplotlib style tailored for scientific publications.
    Increases font sizes significantly and adjusts line weights for readability.
    """
    # Base font size. Everything else scales from here.
    # 18-20pt is large for slides, but guarantees readability when shrunken in a paper.
    base_size = 24

    plt.rcParams.update({
        # --- Typography ---
        "font.family": "serif",
        # Force Computer Modern Roman, fallback to DejaVu
    "font.serif": ["DejaVu Serif", "Computer Modern Serif", "serif"],
    "axes.formatter.use_mathtext": True,        # Ensure math text (subscripts, sigma) uses CM
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,           # Fixes hyphen vs minus sign issues in CM

        # --- Interactive/Backend ---
        "timezone": "UTC",

        # --- Font Sizes ---
        "font.size": base_size,                # Global default
        "axes.titlesize": base_size + 2,       # Facet titles
        "axes.titleweight": "normal",          # Titles not bold for LaTeX look
        "axes.labelsize": base_size + 1,       # X/Y Label size
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,
        "figure.titlesize": base_size + 4,      # SupTitle

        # --- Linewidths & Spines ---
        # Slightly thicker than original (0.9) for better print definition
        "axes.linewidth": 1.2,
        "axes.edgecolor": "black",             # Pure black for high contrast
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "lines.linewidth": 2.5,                # Thicker plot lines
        "patch.linewidth": 1.0,                # Bar edge widths

        # --- Grids ---
        "axes.grid": False,                    # Disabled by default, enabled explicitly per plot
        "grid.color": "0.85",                  # Light gray grid
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",

        # --- Legend ---
        "legend.frameon": False,               # No box around legend
        "legend.loc": "best",

        # --- Saving ---
        "pdf.fonttype": 42,                    # Embed real fonts (editable in Illustrator)
        "ps.fonttype": 42,
        "savefig.bbox": "tight",               # Minimize white margins
        "savefig.pad_inches": 0.05,            # Very tight padding
        "savefig.dpi": 300,                    # High res raster fallback
        "figure.dpi": 100,
    })


def _despine(ax, keep=("left", "bottom")):
    """Removes top and right spines, standard for scientific plots."""
    for side, sp in ax.spines.items():
        sp.set_visible(side in keep)


# -------------------------------------------------------------------------
def _ranks(values):
    """values: dict {ems: total_cost}. -> {ems: rank} (1 = lowest cost)."""
    order = sorted(values, key=lambda e: values[e])
    return {e: i + 1 for i, e in enumerate(order)}


def _rank_matrix(cases):
    """-> (R[ems, case], mean_rank[ems], labels) with EMS in V.EMS_ORDER."""
    ems = V.EMS_ORDER
    labels = [c[0] for c in cases]
    R = np.full((len(ems), len(cases)), np.nan)
    for jc, (_, _grp, d) in enumerate(cases):
        # Calculate costs and ranks. Assuming input data structure matches expectations.
        rk = _ranks({e: V.total_cost_keur(d[e][0], d[e][1]) for e in d if e in d})
        for ir, e in enumerate(ems):
            if e in rk:
                R[ir, jc] = rk[e]
    return R, np.nanmean(R, axis=1), labels


# =========================================================================
def figure_ranking(cases):
    """Heatmap strategy x case of the total-cost rank ; rows sorted best-first."""
    R, mean_rank, labels = _rank_matrix(cases)
    n = len(V.EMS_ORDER)
    # Row sorting logic: best mean rank on top
    order = sorted(range(n), key=lambda i: mean_rank[i])
    Rs = R[order, :]
    mr = mean_rank[order]
    ems = [V.EMS_ORDER[i] for i in order]
    nrows, ncols = Rs.shape
    cmap = plt.get_cmap(CMAP)

    def txtcolor(rk):
        """Chooses text color based on background intensity for readability."""
        if np.isnan(rk): return "white"
        # Ranks 1,2 (dark green) and n-1, n (dark red) get white text.
        return "white" if (rk <= 2 or rk >= n - 1) else "#1a1a1a"

    # --- Adjusted Figure Size ---
    # Increased vertical scaling (0.7 -> 0.95) to accommodate larger fonts and formatting.
    fig, (ax, axm) = plt.subplots(
        1, 2, figsize=(4.0 + 1.8 * ncols, 0.65 * nrows + 3.2),
        gridspec_kw=dict(width_ratios=[ncols, 1.2], wspace=0.10),
        constrained_layout=True)

    # 1. Main Heatmap
    im = ax.imshow(Rs, cmap=cmap, aspect="auto", vmin=1, vmax=n,
               interpolation='nearest', rasterized=True)
    # X-axis ticks (sensitivity cases)
    ax.set_xticks(range(ncols))
    # Rotation adjusted for readability.
    ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")

    # Y-axis ticks (EMS strategies)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(ems)

    # Cell text (the ranks)
    # Fontsize explicitly set slightly smaller than global to fit in cells, but still large.
    cell_fontsize = plt.rcParams['font.size'] * 0.9

    for ir in range(nrows):
        for jc in range(ncols):
            if not np.isnan(Rs[ir, jc]):
                rk = int(Rs[ir, jc])

                # --- START MODIFICATION: HIGHLIGHTING LOGIC ---
                text_str = "%d" % rk
                weight = 'normal'

                if rk <= 3:
                    weight = 'bold' # Top 3 are bold
                    if rk == 1:
                        text_str += "*" # Best gets an asterisk
                # --- END MODIFICATION ---

                ax.text(jc, ir, text_str, ha="center", va="center",
                        fontsize=cell_fontsize, color=txtcolor(rk), fontweight=weight)

    # Thicker white grid to separate cells clearly
    ax.set_xticks(np.arange(-.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3.0)
    ax.tick_params(which="both", length=0) # Hide tick marks
    for sp in ax.spines.values(): sp.set_visible(False) # Hide outer border

    # ax.set_title("EMS ranking by total cost\n(degradation + unserved-energy cost)", pad=20)

    # 2. Mean Rank Column
    axm.imshow(mr.reshape(-1, 1), cmap=cmap, aspect="auto", vmin=1, vmax=n, interpolation='nearest')
    axm.set_xticks([0])
    # Multi-line label to keep column narrow
    axm.set_xticklabels(["Mean\nRank"])
    axm.set_yticks([]) # Hide Y ticks as they match main plot

    # Text in mean rank cells
    for ir in range(nrows):
        val = mr[ir]
        # Since rows are sorted by mean rank, ir=0 is Rank 1 overall, ir=1 is Rank 2, etc.
        overall_rank = ir + 1

        # --- START MODIFICATION: HIGHLIGHTING LOGIC (SUMMARY) ---
        text_str = "%.1f" % val
        weight = 'normal'

        if overall_rank <= 3:
            weight = 'bold'
            if overall_rank == 1:
                text_str += "*"
        # --- END MODIFICATION ---

        axm.text(0, ir, text_str, ha="center", va="center",
                 fontsize=cell_fontsize, color=txtcolor(overall_rank), fontweight=weight)

    # Grid for mean rank column
    axm.set_xticks([-.5, .5], minor=True)
    axm.set_yticks(np.arange(-.5, nrows, 1), minor=True)
    axm.grid(which="minor", color="white", linewidth=3.0)
    axm.tick_params(which="both", length=0)
    for sp in axm.spines.values(): sp.set_visible(False)

    # Colorbar - Adjusted size and padding for larger font labels
    cb = fig.colorbar(im, ax=axm, fraction=0.7, pad=0.15)
    cb.set_ticks(range(1, n + 1))
    cb.set_label("Rank (1 = lowest cost)", labelpad=15)
    cb.outline.set_linewidth(1.0)

    fig.savefig(os.path.join(OUT, "voll_ranking.pdf"))
    plt.close()
    return R, mean_rank, labels


def figure_distribution(cases):
    """Boxplot of the total cost per strategy over all cases (log), sorted by median."""
    ems = V.EMS_ORDER
    # Extract data, filter out missing entries
    data = {e: [V.total_cost_keur(d[e][0], d[e][1])
                for _, _g, d in cases if e in d] for e in ems}
    # Sort strategies by median cost
    order = sorted(ems, key=lambda e: np.median(data[e]) if data[e] else np.inf)
    vals = [data[e] for e in order]

    # Figure size adjusted for 1D categorical plot
    fig, ax = plt.subplots(figsize=(14, 7.5))

    # Professional boxplot styling: thicker lines, muted colors
    bp = ax.boxplot(vals, vert=True, patch_artist=True, widths=0.6,
                    medianprops=dict(color="#1a1a1a", lw=2.5),
                    whiskerprops=dict(lw=1.5, color="#333333"),
                    capprops=dict(lw=1.5, color="#333333"),
                    boxprops=dict(lw=1.5, color="#333333"),
                    flierprops=dict(marker="o", markersize=5, alpha=0.5,
                                    markerfacecolor="0.5", markeredgecolor="0.5"))

    # Fill boxes with restrained blue
    for patch in bp["boxes"]:
        patch.set_facecolor(C_BOX)
        patch.set_alpha(0.9)

    # Plot nominal case as a distinct red diamond
    nominal = cases[0][2]
    for i, e in enumerate(order, 1):
        if e in nominal:
            ax.plot(i, V.total_cost_keur(nominal[e][0], nominal[e][1]),
                    "D", color=C_RED, markersize=10, zorder=10, label="_nolegend_")

    # Formatting
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=25, ha="right")

    # Use generic kEUR currency token
    ax.set_ylabel("Total cost [%s]\n(log scale)" % KEUR, labelpad=15)
    ax.set_title("Total cost distribution across %d sensitivity cases" % len(cases), pad=20)

    # Enable grid for log scale readability
    ax.grid(True, axis="y", which="both", ls="-", alpha=0.6)
    ax.set_axisbelow(True) # Grid behind plots
    _despine(ax)

    # Clean legend
    ax.plot([], [], "D", color=C_RED, markersize=10, label="Nominal case")
    ax.legend(loc="upper left", frameon=True, shadow=False, fancybox=False)

    fig.savefig(os.path.join(OUT, "voll_distribution.pdf"))
    plt.close()
    return order, data


def figure_decomposition(cases):
    """Stacked bars: degradation cost vs unserved-energy cost at the nominal case."""
    nominal = cases[0][2]
    # Filter strategies present in nominal case
    ems = [e for e in V.EMS_ORDER if e in nominal]
    deg = np.array([nominal[e][1] for e in ems])
    clp = np.array([V.cost_lpsp_keur(nominal[e][0]) for e in ems])
    tot = deg + clp

    # Sort bars by total cost
    idx = np.argsort(tot)
    ems = [ems[i] for i in idx]
    deg, clp, tot = deg[idx], clp[idx], tot[idx]
    x = np.arange(len(ems))

    fig, ax = plt.subplots(figsize=(13, 7.5))

    # Stacked bars with white edges for definition
    ax.bar(x, deg, label="Degradation cost", color=C_DEG, edgecolor="white", lw=1.0)
    ax.bar(x, clp, bottom=deg, label="Unserved-energy cost (VOLL)",
           color=C_LPSP, edgecolor="white", lw=1.0)

    # Add total values on top of bars
    # Fontsize explicitly set slightly larger for final numbers.
    val_fontsize = plt.rcParams['font.size']

    max_tot = np.max(tot) if len(tot) > 0 else 1
    for xi, t in zip(x, tot):
        ax.text(xi, t + max_tot * 0.015, "%.0f" % t, ha="center", va="bottom",
                fontsize=val_fontsize, fontweight='bold', color="#1a1a1a")

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(ems, rotation=25, ha="right")
    ax.set_ylabel("Cost [%s]" % KEUR, labelpad=15)
    ax.set_title("Total-cost breakdown: nominal case", pad=20)

    # Legend at bottom right, often less intrusive
    ax.legend(loc="lower right", frameon=True)

    # Light horizontal grid
    ax.grid(True, axis="y", ls="-", alpha=0.6)
    ax.set_axisbelow(True)
    _despine(ax)

    fig.savefig(os.path.join(OUT, "voll_decomposition.pdf"))
    plt.close()


def figure_voll_function():
    """The VOLL (tiered or flat) and the resulting unserved-energy cost vs LPSP."""
    # High resolution line for smooth curves
    lp = np.linspace(0.0, 38.0, 3000)
    voll = np.array([V.voll_eur_per_kwh(x) for x in lp])
    clp  = np.array([V.cost_lpsp_keur(x) for x in lp])

    tiered = len(V.VOLL_TIERS) > 1
    # Clean title based on whether VOLL is constant or tiered
    if tiered:
        ttl = "Tiered Value Of Lost Load (VOLL)"
    else:
        # Format constant value cleanly
        val = V.VOLL_TIERS[0][1]
        ttl = "Constant VOLL: %g %s" % (val, EURKWH)

    # Two panel plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.0), constrained_layout=True)

    # 1. VOLL curve
    ax1.plot(lp, voll, color=C_RED, lw=3.5) # Extra thick line for main concept
    ax1.set_xlabel("Loss of Power Supply Probability (LPSP) [%]", labelpad=10)
    ax1.set_ylabel("VOLL [%s]" % EURKWH, labelpad=10)
    ax1.set_title(ttl, pad=15)

    # Force Y-ticks to match tier values for clarity
    ytick_vals = sorted({v for _, v in V.VOLL_TIERS if v is not None})
    if ytick_vals:
        ax1.set_yticks(ytick_vals)
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    ax1.grid(True, ls="-", alpha=0.6)
    ax1.set_axisbelow(True)
    _despine(ax1)

    # Dotted vertical lines for thresholds
    for thr, _ in V.VOLL_TIERS:
        if thr is not None:
            ax1.axvline(thr * 100.0, color="0.5", ls="--", lw=1.5, alpha=0.7)

    # 2. Resulting Cost Curve
    ax2.plot(lp, clp, color=C_LINE, lw=3.5)
    ax2.set_xlabel("LPSP [%]", labelpad=10)
    ax2.set_ylabel("Unserved-energy cost [%s]" % KEUR, labelpad=10)
    # Math text for multiplication sign
    ax2.set_title("Cost %s VOLL(LPSP) %s Unserved Energy" % (r'$\propto$', TIMES), pad=15)

    ax2.grid(True, ls="-", alpha=0.6)
    ax2.set_axisbelow(True)
    _despine(ax2)

    # Same thresholds
    for thr, _ in V.VOLL_TIERS:
        if thr is not None:
            ax2.axvline(thr * 100.0, color="0.5", ls="--", lw=1.5, alpha=0.7)

    # Informative annotation about assumptions.
    # Fontsize standard, text color dimmed.
    annot_text = (r"Reference energy $E_\mathrm{ref}=%.1f$ GWh over %g years"
                  % (V.E_REF_KWH / 1e6, V.HORIZON_Y))
    ax2.annotate(annot_text,
                 xy=(0.98, 0.04), xycoords="axes fraction", ha="right",
                 fontsize=plt.rcParams['font.size'] * 0.85, color="0.3")

    fig.savefig(os.path.join(OUT, "voll_voll_function.pdf"))
    plt.close()


def figure_soh_annexe():
    """Appendix RB2(SoH): total cost vs estimation bias (regime 1) and noise (regime 2)."""
    soh = V.parse_soh()

    # Determine which plots to generate based on data availability
    has_bias = "bias" in soh and soh["bias"]
    has_sigma = "sigma" in soh and soh["sigma"]

    if not (has_bias or has_sigma):
        print("Skipping SoH appendix: no data.")
        return

    # Adjust subplot creation
    n_plots = (1 if has_bias else 0) + (1 if has_sigma else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 7.0), squeeze=False, constrained_layout=True)
    curr_ax = 0

    # 1. Systematic Bias
    if has_bias:
        ax1 = axes[0, curr_ax]
        curr_ax += 1
        b = np.array([r[0] for r in soh["bias"]]) * 100.0 # Convert to %
        tot = np.array([V.total_cost_keur(r[1], r[2]) for r in soh["bias"]])
        deg = np.array([r[2] for r in soh["bias"]])

        # Line plot with markers
        ax1.plot(b, tot, "-o", color=C_LINE, label="Total cost", markersize=8)
        ax1.plot(b, deg, "--s", color=C_LINE2, label="Degradation cost", alpha=0.8, markersize=7)

        ax1.set_xlabel("SoH estimation bias [%]", labelpad=10)
        ax1.set_ylabel("Cost [%s]" % KEUR, labelpad=10)
        ax1.set_title("Impact of systematic bias", pad=15)
        ax1.grid(True, ls="-", alpha=0.6)
        ax1.set_axisbelow(True)
        _despine(ax1)
        ax1.legend(frameon=True)

    # 2. Gaussian Noise
    if has_sigma:
        ax2 = axes[0, curr_ax]
        sg = np.array([r[0] for r in soh["sigma"]]) * 100.0 # Convert to %
        tot = np.array([V.total_cost_keur(r[1], r[2]) for r in soh["sigma"]])

        ax2.plot(sg, tot, "-o", color=C_LINE, markersize=8)
        # Mathtext for Sigma symbol
        ax2.set_xlabel("SoH estimation noise %s [%%]" % SIG, labelpad=10)
        ax2.set_ylabel("Total cost [%s]" % KEUR, labelpad=10)
        ax2.set_title("Impact of stochastic noise (MC mean)", pad=15)
        ax2.grid(True, ls="-", alpha=0.6)
        ax2.set_axisbelow(True)
        _despine(ax2)

    # Main Figure Title (SupTitle)
    fig.suptitle("Appendix: RB2(SoH) robustness to SoH-estimation error", fontweight='normal', y=1.02)

    fig.savefig(os.path.join(OUT, "voll_soh_annexe.pdf"), bbox_inches="tight")
    plt.close()


def write_recap(cases, mean_rank, labels, data):
    """Writes ASCII text recap. Unchanged logic, ensure UTF-8."""
    path = os.path.join(OUT, "voll_summary.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Unified summary -- total_cost = degradation cost + unserved-energy cost\n")
            f.write("# VOLL [EUR/kWh]: " +
                    " ; ".join("<%g%% -> %g" % ((t * 100.0) if t else 999, v)
                               for t, v in V.VOLL_TIERS) + "\n")
            f.write("# E_ref = %.3f MWh (net planned energy, %g yr) ; "
                    "unserved_energy = LPSP%%/100 * E_ref\n\n" % (V.E_REF_KWH/1000.0, V.HORIZON_Y))

            f.write("## Global ranking (mean total-cost rank over %d cases)\n" % len(cases))
            f.write("mean_rank;ems;total_cost_median_kEUR;min;max\n")

            # Get indices for sorting based on mean rank
            ems_list = V.EMS_ORDER
            sorted_idx = sorted(range(len(ems_list)), key=lambda k: mean_rank[k])

            for i in sorted_idx:
                e = ems_list[i]
                vals = data[e]
                if vals:
                    f.write("%.2f;%s;%.2f;%.2f;%.2f\n"
                            % (mean_rank[i], e, np.median(vals), min(vals), max(vals)))
                else:
                    f.write("%.2f;%s;NA;NA;NA\n" % (mean_rank[i], e))

            f.write("\n## Total cost per case and strategy [kEUR]\n")
            f.write("ems;" + ";".join(labels) + "\n")
            for e in ems_list:
                row = [e]
                for _, _g, d in cases:
                    if e in d:
                        row.append("%.2f" % V.total_cost_keur(d[e][0], d[e][1]))
                    else:
                        row.append("NA")
                f.write(";".join(row) + "\n")
        return path
    except IOError as e:
        print(f"Error writing recap file: {e}")
        return None


def main():
    print("=== Unified summary (degradation cost + unserved-energy cost) ===", flush=True)

    # 1. Apply the professional style globally
    set_pub_style()

    # 2. Load data (assuming V.build_cases() handles IO/parsing)
    cases = V.build_cases()
    if not cases:
        print("Error: No cases loaded.")
        return
    print("    %d cases: %s" % (len(cases), ", ".join(c[0] for c in cases)), flush=True)

    # 3. Generate individual figures
    print("Generating figures...", end="", flush=True)
    R, mean_rank, labels = figure_ranking(cases)
    order, data = figure_distribution(cases)
    figure_decomposition(cases)
    figure_voll_function()
    figure_soh_annexe()
    print(" Done.")

    # 4. Write text summary
    recap = write_recap(cases, mean_rank, labels, data)

    # 5. CLI output
    print("\n" + "=" * 64)
    print("GLOBAL RANKING (mean total-cost rank, 1 = best)")
    print("-" * 64)
    ems_list = V.EMS_ORDER
    sorted_idx = sorted(range(len(ems_list)), key=lambda k: mean_rank[k])
    for i in sorted_idx:
        e = ems_list[i]
        med_cost = np.median(data[e]) if data[e] else np.nan
        print("  mean_rank %5.2f  %-12s  (median total cost %.1f kEUR)"
              % (mean_rank[i], e, med_cost))
    print("=" * 64)
    print("Figures (PDF) -> %s" % OUT)
    if recap:
        print("Recap (Txt)   -> %s" % recap)


if __name__ == "__main__":
    # Ensure output directory exists
    if not os.path.exists(OUT) and OUT != ".":
        os.makedirs(OUT, exist_ok=True)
    main()