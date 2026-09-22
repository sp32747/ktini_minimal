"""Build a two-page, print-ready livestock product sales brief."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

OUT = Path(__file__).resolve().parent
NAVY = "#142F37"
TEAL = "#0D7977"
MINT = "#A8E2CC"
CREAM = "#FAF8F2"
INK = "#1C343C"
GRAY = "#536A70"
PALE = "#E9F1EB"
GOLD = "#DCA957"

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def txt(ax, x, y, value, size=10, color=INK, bold=False, **kwargs):
    return ax.text(x, y, value, fontsize=size, color=color, va="top",
                   weight="bold" if bold else "normal", linespacing=1.45, **kwargs)


def card(ax, x, y, w, h, color=PALE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.35,rounding_size=1.1",
                              facecolor=color, edgecolor="none"))


def page(number):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=CREAM)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    ax.add_patch(Rectangle((0, 98.9), 100, 1.1, facecolor=TEAL))
    txt(ax, 6, 96, "KTINOSKARE", 14, TEAL, True)
    txt(ax, 94, 95.6, "LIVESTOCK INTELLIGENCE", 8.5, GRAY, ha="right")
    ax.plot([6, 94], [5.1, 5.1], color="#CDD8D0", lw=.8)
    txt(ax, 6, 3.9, "CATTLE & BUFFALO  /  PRODUCT & PILOT BRIEF", 7.5, GRAY)
    txt(ax, 94, 3.9, f"{number} / 2", 8, TEAL, True, ha="right")
    return fig, ax


def save(pdf, fig, index):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for item in ax.texts:
            extent = item.get_window_extent(renderer)
            assert extent.x0 >= 0 and extent.x1 <= fig.bbox.width + 1, item.get_text()
            assert extent.y0 >= 0 and extent.y1 <= fig.bbox.height + 1, item.get_text()
    pdf.savefig(fig, facecolor=CREAM)
    fig.savefig(OUT / f"livestock_sales_page_{index}.png", dpi=150, facecolor=CREAM)
    plt.close(fig)


def product_page(pdf):
    fig, ax = page(1)
    card(ax, 6, 64, 88, 26.5, NAVY)
    txt(ax, 10, 87.7, "CLIMATE-AWARE ANIMAL REVIEW", 9, MINT, True)
    txt(ax, 10, 83.8, "See animal patterns.\nAdd climate context.", 29, "white", True)
    txt(ax, 10, 72.7, "Turn cow and buffalo sensor history into condition scores,\nalerts and visual trends that support a more informed review.", 11, "#D8E8E2")
    txt(ax, 10, 67.3, "WORKING SOFTWARE PROTOTYPE  •  AVAILABLE FOR DEMONSTRATION", 8, MINT, True)

    txt(ax, 6, 61.4, "One animal. Six screening signals.", 19, bold=True)
    txt(ax, 6, 57.7, "Designed for dairy operators, farm managers and animal-health teams.", 9.8, GRAY)
    signals = [
        (6, 44.8, "01", "Low oxygen", "SpO2 patterns and\nchanges from baseline"),
        (36, 44.8, "02", "Fever pattern", "Body-temperature trends\nwith sensor context"),
        (66, 44.8, "03", "Resting heart rate", "Elevated heart rate\nwhen movement is low"),
        (6, 33.7, "04", "Reduced activity", "Movement patterns\nacross sensor windows"),
        (36, 33.7, "05", "Heat stress", "Temperature, humidity\nand animal response"),
        (66, 33.7, "06", "Cold stress", "Cold-weather context\nand animal response"),
    ]
    for x, y, number, title, description in signals:
        card(ax, x, y, 28, 9.8)
        txt(ax, x + 2, y + 8.6, number, 8, TEAL, True)
        txt(ax, x + 2, y + 6.7, title, 11, bold=True)
        txt(ax, x + 2, y + 4, description, 8.5, GRAY)

    txt(ax, 6, 30.8, "From uploaded readings to a clearer review", 15, bold=True)
    for i, (title, body) in enumerate([
        ("UPLOAD", "Sensor CSV\nwith climate data"), ("SELECT", "Region, species\nand animal"),
        ("SCORE", "Six model scores\nand threshold alerts"), ("REVIEW", "Trends, health score\nand CSV export"),
    ]):
        x = 6 + i * 23
        txt(ax, x, 26.9, title, 9, TEAL, True)
        txt(ax, x, 24.6, body, 8.5, GRAY)
        if i < 3:
            ax.annotate("", xy=(x + 22, 25.3), xytext=(x + 19.5, 25.3),
                        arrowprops={"arrowstyle": "-|>", "color": TEAL, "lw": 1.2})
    card(ax, 6, 11.1, 88, 8.7, "#E1ECE8")
    txt(ax, 9, 18.5, "A practical starting point for a farm pilot", 11, TEAL, True)
    txt(ax, 9, 15.9, "Bring vital signs and local conditions into one view. Inspect changes over time.\nDownload prediction histories to support discussion with your animal-health team.", 9.5)
    txt(ax, 6, 8.9, "Scores support human review; they do not\ndiagnose illness, prescribe treatment or establish validated early detection.", 8.1, GRAY)
    save(pdf, fig, 1)


def pilot_page(pdf):
    fig, ax = page(2)
    txt(ax, 6, 90.4, "Start with your animals.\nValidate in your conditions.", 25, bold=True)
    txt(ax, 6, 80.3, "A focused pilot can establish data readiness and practical usefulness before rollout.", 10, GRAY)

    txt(ax, 6, 75.9, "Five climate scenarios in the demonstration", 14, bold=True)
    regions = [("Rajasthan", "Hot and dry", "#F0D8BA"), ("Kerala", "Hot and humid", "#BCE0D1"),
               ("Punjab", "Warm plains", "#E5DFB5"), ("Karnataka", "Mild plateau", "#BFD9DB"),
               ("Kashmir", "Cold winter", "#CDD9E9")]
    for i, (region, climate, color) in enumerate(regions):
        x = 6 + i * 18
        card(ax, x, 66.9, 16, 6.4, color)
        txt(ax, x + 8, 72, region, 9, bold=True, ha="center")
        txt(ax, x + 8, 69.5, climate, 8, GRAY, ha="center")
    txt(ax, 6, 65.2, "Illustrative seasonal environments, not proof of performance across India's real farms.", 8.3, GRAY)

    for x, number, label in [(6, "30", "Livestock"), (36, "129,600", "readings"), (66, "6", "trained classifiers")]:
        txt(ax, x, 61.3, number, 24, TEAL, True)
        txt(ax, x, 56.8, label, 9, GRAY)
    txt(ax, 6, 53.9, "15 cows + 15 buffaloes • 72 hours each • Separate training, validation and test animals", 8.3, GRAY)

    txt(ax, 6, 49.7, "What you can demonstrate today", 14, bold=True)
    txt(ax, 6, 46.7, "• Local browser dashboard and CSV upload\n• Animal-specific condition scores and alerts\n• Health-score trends and downloadable results", 9.5)
    txt(ax, 54, 46.7, "• Separate cow and buffalo features\n• Temperature, humidity and wind context\n• Review of six distinct condition patterns", 9.2)

    txt(ax, 6, 37.8, "Proposed pilot: four clear steps", 14, bold=True)
    stages = [
        ("01", "Agree the scope", "Select farm, animals, priority conditions and review criteria."),
        ("02", "Check the data", "Confirm sensors, units, timestamps and one-minute exports."),
        ("03", "Evaluate together", "Compare predictions with independently reviewed farm records."),
        ("04", "Plan deployment", "Use findings to scope calibration, integrations and rollout."),
    ]
    for i, (number, title, body) in enumerate(stages):
        y = 34.2 - i * 4.1
        ax.add_patch(Circle((7.8, y - .9), 1.6, facecolor=TEAL, edgecolor="none"))
        txt(ax, 7.8, y - .2, number, 7.5, "white", True, ha="center")
        txt(ax, 11, y, title, 9.3, bold=True)
        txt(ax, 38, y, body, 8.3, GRAY)

    card(ax, 6, 11.3, 88, 7.2, NAVY)
    txt(ax, 9, 17.1, "sensor animal-data export to the system.", 12, "white", True)
    txt(ax, 9, 14.2, "Define the  scope, data availability and success criteria.", 9, "#D8E8E2")
    txt(ax, 6, 9.6, "Current delivery is batch CSV analysis. Live-device ingestion, external notifications and\nproduction integrations need further work. Clinical accuracy and financial outcomes are undercheck.", 7.8, GRAY)
    save(pdf, fig, 2)


def main():
    path = OUT / "KTINOSKARE_Livestock_Sales_Brief.pdf"
    with PdfPages(path, metadata={"Title": "KTINOSKARE | Cattle & Buffalo Product and Pilot Brief",
                                  "Author": "KTINOSKARE", "Subject": "Two-page livestock prediction prototype sales document"}) as pdf:
        product_page(pdf)
        pilot_page(pdf)
        assert pdf.get_pagecount() == 2
    print(f"Created two-page PDF: {path}")


if __name__ == "__main__":
    main()
