"""Build anonymous manuscript and author-side submission metadata template."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.quantum_reservoir import QuantumReservoir

OUTPUT = ROOT / "output" / "q2_submission"
RESULTS = ROOT / "results" / "q2_study"


def shade(cell, color="D9E2F3"):
    properties = cell._tc.get_or_add_tcPr()
    element = OxmlElement("w:shd")
    element.set(qn("w:fill"), color)
    properties.append(element)


def base_style(document):
    section = document.sections[0]
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.08
    for name, size in [("Title", 16), ("Heading 1", 13), ("Heading 2", 11)]:
        style = document.styles[name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = None
    caption = document.styles.add_style("PaperCaption", WD_STYLE_TYPE.PARAGRAPH)
    caption.font.name = "Times New Roman"
    caption.font.size = Pt(9)
    caption.font.italic = True
    caption.paragraph_format.space_after = Pt(7)


def paragraph(document, text, *, bold_label=None, alignment=None):
    item = document.add_paragraph()
    if alignment is not None:
        item.alignment = alignment
    if bold_label and text.startswith(bold_label):
        item.add_run(bold_label).bold = True
        item.add_run(text[len(bold_label):])
    else:
        item.add_run(text)
    return item


def table(document, frame, caption, digits=4, index_name="Model"):
    paragraph(document, caption, bold_label=caption.split(". ")[0] + ".")
    grid = document.add_table(rows=1, cols=len(frame.columns) + 1)
    grid.style = "Table Grid"
    labels = [frame.index.name or index_name, *frame.columns]
    for cell, value in zip(grid.rows[0].cells, labels):
        cell.text = str(value)
        shade(cell)
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for index, row in frame.iterrows():
        cells = grid.add_row().cells
        cells[0].text = str(index)
        for cell, value in zip(cells[1:], row):
            cell.text = f"{float(value):.{digits}g}" if isinstance(value, (float, np.floating, int, np.integer)) else str(value)
    document.add_paragraph()


def circuit_figure(path):
    reservoir = QuantumReservoir(5, input_scale=0.8, layers=2)
    figure = reservoir._build_circuit(np.array([0.15, -0.22, 0.31, -0.14, 0.08])).draw(
        output="mpl", fold=-1, plot_barriers=False
    )
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def panel_figure(path, metrics, ablation, noise):
    qlike = metrics.pivot(index="asset", columns="model", values="qlike")
    asset_order = ["SPY", "QQQ", "EFA", "EEM", "TLT", "GLD", "FXE"]
    qlike = qlike.loc[asset_order]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6))
    qlike[["ridge", "qrc", "garch"]].plot(kind="bar", ax=axes[0], color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0].set_title("Panel A: QLIKE across assets")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("QLIKE (lower is better)")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(fontsize=8)
    ablation["qlike"].sort_values().plot(kind="bar", ax=axes[1], color="#f58518")
    axes[1].set_title("Panel B: SPY QRC ablations")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("QLIKE")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    noise["qlike"].plot(marker="o", ax=axes[2], color="#e45756")
    axes[2].set_title("Panel C: global depolarizing stress")
    axes[2].set_xlabel("depolarizing probability")
    axes[2].set_ylabel("QLIKE")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def manuscript():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figures = OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    circuit = figures / "qrc_circuit.png"
    panels = figures / "results_panels.png"
    circuit_figure(circuit)

    panel_metrics = pd.read_csv(RESULTS / "market_panel_metrics.csv")
    bootstrap = pd.read_csv(RESULTS / "market_panel_bootstrap_qlike.csv")
    ablation = pd.read_csv(RESULTS / "spy_ablation_metrics.csv", index_col=0)
    noise = pd.read_csv(RESULTS / "spy_global_depolarizing_noise_metrics.csv", index_col=0)
    panel_figure(panels, panel_metrics, ablation, noise)
    spy = panel_metrics.query("asset == 'SPY'").set_index("model")
    spy_boot = bootstrap.query("asset == 'SPY'").set_index("comparator")
    close = pd.read_csv(RESULTS / "spy_close_return_variance" / "metrics.csv", index_col=0)
    macro = pd.read_csv(RESULTS / "macro_robustness_metrics.csv").pivot(index="series", columns="model", values="rmse")
    crisis = pd.read_csv(RESULTS / "crisis_period_metrics.csv")
    crisis_spy = crisis.query("asset == 'SPY'").pivot(index="period", columns="model", values="qlike")
    panel_qlike = panel_metrics.pivot(index="asset", columns="model", values="qlike")

    doc = Document()
    base_style(doc)
    doc.core_properties.author = ""
    doc.core_properties.last_modified_by = ""
    doc.core_properties.title = "Nested Time Series Evidence on Quantum Reservoir Style Feature Maps for Volatility Forecasting"
    doc.core_properties.subject = "Anonymous review manuscript"
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Nested Time Series Evidence on Quantum Reservoir Style Feature Maps for Volatility Forecasting")
    subtitle = paragraph(doc, "Anonymous manuscript for Journal of Forecasting free format submission", alignment=WD_ALIGN_PARAGRAPH.CENTER)
    subtitle.runs[0].italic = True

    doc.add_heading("Abstract", level=1)
    paragraph(doc, "This paper evaluates a fixed quantum reservoir style feature map for next day volatility forecasting under nested chronological model selection. The primary target is Garman Klass daily variance for seven liquid exchange traded funds spanning United States and international equities, Treasury bonds, gold, and the euro. Every learned model is tuned within each outer expanding training window by rolling origin validation; forecasts are evaluated out of sample with QLIKE, RMSE, MAE, Diebold Mariano tests, and moving block bootstrap confidence intervals. The quantum model uses repeated Ry data encoding, Rx mixing, a CNOT ring, and local Z plus nearest neighbour ZZ observables. Across assets, linear ridge has the lowest QLIKE. The quantum model substantially and often significantly improves on Student t GARCH and a matched coordinate wise cosine control, but it does not outperform ridge. SPY ablations show that entangling dynamics, pairwise observables, and two layers matter for the reported QRC result. A global depolarizing stress test produces only small degradation at 0 to 2 percent noise. A matched close return variance target is also reported to separate the range proxy question from the GARCH comparison. The evidence supports a carefully benchmarked quantum feature model, not quantum advantage.")
    paragraph(doc, "Keywords: volatility forecasting; quantum reservoir computing; range based volatility; nested time series validation; QLIKE; Garman Klass", bold_label="Keywords:")

    doc.add_heading("1. Introduction", level=1)
    paragraph(doc, "Volatility is central to risk measurement, derivative valuation, and portfolio choice. Daily variance is latent, so all empirical exercises rely on a proxy. Range based estimators exploit open, high, low, and close prices and can contain more intraday information than squared close to close returns (Garman and Klass, 1980). The question studied here is deliberately narrow: whether a transparent quantum reservoir style feature map adds forecasting value relative to strong, leakage safe classical controls.")
    paragraph(doc, "The paper strengthens a previous single asset exercise along five dimensions. It uses nested time series hyperparameter selection rather than fixed settings; it evaluates a seven asset ETF panel and crisis windows; it reports QRC ablations and a depolarizing noise stress test; it adds moving block bootstrap confidence intervals for QLIKE differentials; and it evaluates a matched close return variance target to clarify the GARCH comparison. The resulting evidence is mixed and is reported as such.")

    doc.add_heading("2. Related Literature", level=1)
    paragraph(doc, "GARCH models remain a standard conditional variance benchmark (Engle, 1982; Bollerslev, 1986). Volatility forecast comparisons must account for imperfect variance proxies; QLIKE has desirable robustness properties in this setting (Patton, 2011). Diebold and Mariano (1995) provide a test of equal predictive accuracy, while Hansen and Lunde (2005) show why broad benchmark comparisons require disciplined inference. Quantum reservoir computing fixes a quantum dynamical system and learns only its classical readout (Fujii and Nakajima, 2017). This architecture is attractive for controlled feature mapping, but simulator success does not establish hardware or computational advantage.")

    doc.add_heading("3. Data Targets and Protocol", level=1)
    paragraph(doc, "Daily OHLC data cover 3 January 2005 through 31 December 2025 for SPY, QQQ, EFA, EEM, TLT, GLD, and FXE. At feature date t, the primary target is next day Garman Klass variance: GK(t+1) = 0.5[log(H/L)] squared minus [2 log(2) minus 1][log(C/O)] squared. Inputs are five lagged log GK values ending at t. Forecasts are exponentiated before scoring, guaranteeing positive variance inputs for QLIKE. The panel starts with 756 observations; each model is refit yearly (252 trading days).")
    paragraph(doc, "Nested tuning is causal. At every outer refit, candidate parameters are selected only from the current outer training window using three rolling origin validation folds of 126 observations. Grids cover ridge penalty, cosine scale and penalty, and QRC penalty, input scale, and depth. The outer test sequence is never accessed by the inner selection step. Low frequency GDP and industrial production absolute growth proxies are retained as separate final vintage robustness exercises; they are not real time macro forecasts.")

    doc.add_heading("4. Models", level=1)
    paragraph(doc, "The QRC maps standardized five lag inputs through two repeated blocks of Ry encoding, Rx mixing, and ring CNOT gates. A ridge readout consumes five local Z and five nearest neighbor ZZ expectation values. The circuit is an exact statevector simulation. Benchmarks are variance persistence, ridge regression on the same lags, a cosine ridge control reproducing coordinate wise cos(Sx) mapping, and Student t GARCH(1,1) fit to close to close returns. GARCH uses the same outer forecast dates and yearly block refits.")
    item = doc.add_paragraph()
    item.alignment = WD_ALIGN_PARAGRAPH.CENTER
    item.add_run().add_picture(str(circuit), width=Inches(6.4))
    paragraph(doc, "Figure 1. Five qubit two layer feature circuit. Standardized lagged values enter Ry rotations. Rx mixing and a CNOT ring create dynamics; local and pairwise Z expectations form the readout features.", alignment=WD_ALIGN_PARAGRAPH.CENTER)

    doc.add_heading("5. Evaluation and Inference", level=1)
    paragraph(doc, "Primary evaluation uses QLIKE, with RMSE and MAE reported as complementary scale dependent losses. QLIKE comparisons use mean comparator minus QRC loss; positive values favor QRC. Moving block bootstrap confidence intervals use blocks of 20 trading days and 2,000 resamples. The Diebold Mariano statistic uses a Bartlett HAC long run variance. Crisis reports cover the Global Financial Crisis (September 2008 to June 2009), COVID shock (20 February to 30 June 2020), and inflation and Ukraine shock (February to October 2022).")

    doc.add_heading("6. Results", level=1)
    paragraph(doc, "Table 1 gives the central result. Ridge produces the lowest QLIKE in all seven assets. QRC improves on GARCH in six of seven assets and on the matched cosine control in every asset. QRC does not improve on ridge. Thus the expanded design reinforces a conservative conclusion: QRC can be a useful nonlinear benchmark, but no quantum advantage is established.")
    table(doc, panel_qlike[["ridge", "qrc", "cosine", "garch", "naive"]], "Table 1. Panel QLIKE by asset. Lower is better.", digits=4, index_name="Asset")
    table(doc, spy[["rmse", "mae", "qlike"]], "Table 2. SPY out of sample performance under nested selection.", digits=5)
    paragraph(doc, "For SPY, the QLIKE difference against GARCH is 0.648 with a 95 percent moving block bootstrap interval [0.480, 0.821]. The interval against cosine is [minus 0.002, 0.394], while ridge is better than QRC by 0.205 with interval [minus 0.396, minus 0.080]. The corresponding DM comparison with GARCH favors QRC (statistic 5.58, p below 0.001); the ridge comparison favors ridge (statistic minus 2.31, p = 0.021).")
    table(doc, spy_boot[["mean_difference", "ci_lower", "ci_upper", "p_value_two_sided"]], "Table 3. SPY moving block bootstrap QLIKE difference: comparator minus QRC.", digits=4, index_name="Comparator")
    item = doc.add_paragraph()
    item.alignment = WD_ALIGN_PARAGRAPH.CENTER
    item.add_run().add_picture(str(panels), width=Inches(6.6))
    paragraph(doc, "Figure 2. Panel evidence. Lower QLIKE is better. Noise is a global depolarizing stress model applied after each feature map layer, not a calibrated device specific noise model.", alignment=WD_ALIGN_PARAGRAPH.CENTER)

    doc.add_heading("7. Ablation Noise and Crisis Evidence", level=1)
    paragraph(doc, "The SPY ablation identifies material design dependence. Removing entanglement raises QLIKE from 0.671 to 1.110; removing pair observables raises it to 0.952; reducing depth to one layer raises it to 0.894. These results do not prove a quantum mechanism, but they rule out attributing the reported QRC score solely to an unchanged coordinate wise cosine map. Under the stated global depolarizing approximation, QLIKE changes from 0.67070 at zero noise to 0.67072 at 2 percent. This narrow stress test supports numerical stability but is not a hardware validation.")
    table(doc, ablation[["rmse", "mae", "qlike"]], "Table 4. SPY QRC ablations.", digits=5)
    paragraph(doc, "Crisis results are heterogeneous. In SPY, persistence and ridge outperform QRC during the Global Financial Crisis and COVID shock by QLIKE, whereas QRC and ridge are close during the 2022 inflation and Ukraine window. Therefore, whole sample GARCH improvements must not be read as uniform crisis dominance.")
    table(doc, crisis_spy[["naive", "ridge", "qrc", "garch"]], "Table 5. SPY crisis period QLIKE. Lower is better.", digits=4, index_name="Period")

    doc.add_heading("8. Proxy Alignment and Additional Robustness", level=1)
    paragraph(doc, "GARCH forecasts conditional close to close return variance, while the main target is Garman Klass range based variance. This comparison is informative for practical forecasting but is not a perfectly matched measurement comparison. The matched SPY close return variance exercise addresses that concern: GARCH QLIKE is 2.572, QRC is 6.192, ridge is 6.163, and cosine is 7.089. Hence GARCH is favored when evaluated against its closer return variance target, whereas the range based task favors the simple ridge benchmark. Both findings are retained because they define the scope of each claim.")
    table(doc, close[["rmse", "mae", "qlike"]], "Table 6. SPY matched next day close return variance target.", digits=5)
    paragraph(doc, "On GDP growth magnitude, cosine ridge has the lowest RMSE; on industrial production, ridge and QRC are nearly tied by RMSE and QRC has the lowest MAE. These finalized low frequency data remain limited robustness checks, not evidence for real time macro forecasting.")
    table(doc, macro[["ridge", "cosine", "qrc", "naive"]], "Table 7. Macroeconomic robustness RMSE for next period absolute growth magnitude.", digits=4, index_name="Series")

    doc.add_heading("9. Reproducibility Ethics and Limitations", level=1)
    paragraph(doc, "All computations use a fixed random seed, cached OHLC input files, preserved forecast panels, YAML configuration, and automated tests for causal alignment, nested training windows, bootstrap behavior, target construction, and reservoir sensitivity. The reproducibility package contains the data download routine, processed inputs where redistribution is permitted, code, configuration, and generated result files. Source market data are accessed through Yahoo Finance via yfinance; GDP and industrial production data are sourced from FRED. A public archival repository and permanent DOI must be inserted in the Data Availability Statement before submission.")
    paragraph(doc, "Limitations are substantial. The QRC is a simulator, not a hardware experiment. The global depolarizing stress test is not a calibrated noise model. The candidate grid is deliberately compact, and the study does not test transaction costs, multistep horizons, high frequency realized measures, or portfolio outcomes. Results therefore support reproducible comparative forecasting evidence rather than scalability, operational, or quantum advantage claims.")

    doc.add_heading("10. Conclusion", level=1)
    paragraph(doc, "Nested rolling evaluation across a seven asset panel changes the interpretation of the QRC result. The model improves on GARCH and a matched cosine control in much of the panel, and its performance depends on the full circuit design. Yet classical ridge is strongest on every primary asset under QLIKE. A credible Q2 submission should present that mixed result directly: quantum reservoir style feature maps merit controlled empirical study, but the evidence here does not establish superiority to a strong linear benchmark or quantum advantage.")

    doc.add_heading("Data Availability Statement", level=1)
    paragraph(doc, "The data and code supporting the findings are available from the corresponding author and will be deposited in a public archival repository before acceptance. The replication package will include a versioned source-code release, YAML configuration, generated forecasts, result tables, and instructions to retrieve public market and FRED series. Persistent repository URL and DOI to be supplied at submission.")
    doc.add_heading("Conflict of Interest Statement", level=1)
    paragraph(doc, "The authors declare no conflict of interest. This statement must be confirmed by all authors before submission.")
    doc.add_heading("Funding Statement", level=1)
    paragraph(doc, "No external funding was used. This statement must be confirmed or replaced by the authors before submission.")

    doc.add_heading("References", level=1)
    references = [
        "Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307-327. https://doi.org/10.1016/0304-4076(86)90063-1.",
        "Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business and Economic Statistics, 13(3), 253-263. https://doi.org/10.1080/07350015.1995.10524599.",
        "Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica, 50(4), 987-1007. https://doi.org/10.2307/1912773.",
        "Fujii, K., & Nakajima, K. (2017). Harnessing disordered ensemble quantum dynamics for machine learning. Physical Review Applied, 8, 024030. https://doi.org/10.1103/PhysRevApplied.8.024030.",
        "Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. Journal of Business, 53(1), 67-78. https://doi.org/10.1086/296072.",
        "Hansen, P. R., & Lunde, A. (2005). A forecast comparison of volatility models: Does anything beat a GARCH(1,1)? Journal of Applied Econometrics, 20(7), 873-889. https://doi.org/10.1002/jae.800.",
        "Nakajima, K., Fujii, K., Negoro, M., Mitarai, K., & Kitagawa, M. (2019). Boosting computational power through spatial multiplexing in quantum reservoir computing. Physical Review Applied, 11, 034005. https://doi.org/10.1103/PhysRevApplied.11.034005.",
        "Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. Journal of Econometrics, 160(1), 246-256. https://doi.org/10.1016/j.jeconom.2010.03.034.",
        "Thakkar, S., Kazdaghli, S., Mathur, N., Kerenidis, I., Ferreira-Martins, A. J., & Brito, S. (2024). Improved financial forecasting via quantum machine learning. Quantum Machine Intelligence, 6, 27. https://doi.org/10.1007/s42484-024-00157-0.",
    ]
    for entry in references:
        paragraph(doc, entry)
    doc.save(OUTPUT / "anonymous_manuscript_q2.docx")


def metadata_template():
    doc = Document()
    base_style(doc)
    doc.core_properties.author = ""
    doc.core_properties.last_modified_by = ""
    doc.core_properties.title = "Journal of Forecasting Submission Metadata and Final Author Verification"
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Journal of Forecasting Submission Metadata and Final Author Verification")
    paragraph(doc, "Companion to anonymous manuscript. Complete every field before portal submission.")
    sections = {
        "Manuscript": [
            "Title: Nested Time Series Evidence on Quantum Reservoir Style Feature Maps for Volatility Forecasting",
            "Article type: Original research article",
            "Target journal: Journal of Forecasting",
            "Submission format: Free format editable manuscript with embedded figures and tables",
        ],
        "Author and Correspondence Details": [
            "Author names and order: ________________________________",
            "Affiliations: ________________________________",
            "Corresponding author, email, postal address: ________________________________",
            "ORCID for each author: ________________________________",
        ],
        "Mandatory Declarations": [
            "Funding statement confirmed or corrected: ________________________________",
            "Conflict of interest statement confirmed or corrected: ________________________________",
            "Author contributions approved by every author: ________________________________",
            "All authors approved final manuscript and agree to submission: ________________________________",
        ],
        "Data and Code": [
            "Public archive URL: ________________________________",
            "Archive DOI or persistent identifier: ________________________________",
            "Archived contents checked: code, environment instructions, configuration, forecast panels, result tables, permitted processed data, and source-data retrieval instructions.",
            "Data Availability Statement updated with public URL and DOI: ________________________________",
        ],
        "Final File Checks": [
            "Anonymous manuscript contains no author names, affiliations, acknowledgements, document properties, or repository identifiers that reveal identity.",
            "Title page contains names, affiliations, email, ORCID, declarations, and archive link for the editorial office.",
            "Every figure/table cited in text; figures legible; references cross-checked against DOI publishers.",
            "Journal author-guideline check completed on submission date: ________________________________",
        ],
    }
    for heading, items in sections.items():
        doc.add_heading(heading, level=1)
        for value in items:
            paragraph(doc, value)
    doc.save(OUTPUT / "submission_metadata_template.docx")


if __name__ == "__main__":
    manuscript()
    metadata_template()
