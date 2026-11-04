"""Create an editable journal-style manuscript from reproducible result files."""

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

OUTPUT = ROOT / "output" / "manuscript"
RUN = ROOT / "results" / "spy_gk_2005_2025"
MACRO = ROOT / "results" / "macro_robustness"


def set_cell_shading(cell, color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), color)
    tc_pr.append(shading)


def style_document(document):
    section = document.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15
    for name, size in [("Title", 16), ("Heading 1", 14), ("Heading 2", 12)]:
        style = document.styles[name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
    caption = document.styles.add_style("CaptionText", WD_STYLE_TYPE.PARAGRAPH)
    caption.font.name = "Times New Roman"
    caption.font.size = Pt(9)
    caption.font.italic = True
    caption.paragraph_format.space_after = Pt(8)


def add_paragraph(document, text, *, bold_start=None):
    paragraph = document.add_paragraph()
    if bold_start and text.startswith(bold_start):
        paragraph.add_run(bold_start).bold = True
        paragraph.add_run(text[len(bold_start):])
    else:
        paragraph.add_run(text)
    return paragraph


def add_table(document, dataframe, caption, digits=6):
    add_paragraph(document, caption, bold_start=caption.split(". ")[0] + ".")
    table = document.add_table(rows=1, cols=len(dataframe.columns) + 1)
    table.style = "Table Grid"
    headers = [dataframe.index.name or "Model", *dataframe.columns]
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = str(value)
        set_cell_shading(cell, "D9E2F3")
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for index, row in dataframe.iterrows():
        cells = table.add_row().cells
        cells[0].text = str(index)
        for cell, value in zip(cells[1:], row):
            cell.text = f"{value:.{digits}g}" if isinstance(value, (float, np.floating)) else str(value)
    document.add_paragraph()


def create_circuit_figure(path):
    reservoir = QuantumReservoir(5, input_scale=0.8, layers=2, include_pair_observables=True)
    circuit = reservoir._build_circuit(np.array([0.15, -0.22, 0.31, -0.14, 0.08]))
    figure = circuit.draw(output="mpl", fold=-1, plot_barriers=False)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figures = OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    circuit_path = figures / "qrc_circuit.png"
    create_circuit_figure(circuit_path)

    metrics = pd.read_csv(RUN / "metrics.csv", index_col=0)
    dm = pd.read_csv(RUN / "diebold_mariano.csv", index_col=0)
    gdp = pd.read_csv(MACRO / "gdp_growth_metrics.csv", index_col=0)
    indpro = pd.read_csv(MACRO / "indpro_growth_metrics.csv", index_col=0)
    macro = pd.concat({"GDP growth": gdp, "Industrial production": indpro}, names=["Series", "Model"])

    document = Document()
    style_document(document)

    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Quantum Reservoir-Style Feature Maps for Out-of-Sample Equity Volatility Forecasting")
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Evidence from SPY Range-Based Variance and Macroeconomic Robustness Tests").italic = True
    author = document.add_paragraph()
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author.add_run("Author details omitted for anonymous review")

    document.add_heading("Abstract", level=1)
    add_paragraph(document, (
        "This study evaluates a reproducible quantum reservoir-style feature map for one-step-ahead volatility forecasting. "
        "The primary task forecasts next-day S&P 500 ETF (SPY) Garman-Klass variance from daily open, high, low and close prices over 2005-2025. "
        "The benchmark uses expanding-window estimation, fixed semiannual refitting, a common out-of-sample index, and positive variance forecasts obtained by exponentiating log-variance predictions. "
        "A five-qubit angle-encoded circuit combines repeated data encoding, non-commuting single-qubit mixing and CNOT-ring interactions; local-Z and nearest-neighbour ZZ expectations feed a ridge readout. "
        "The experiment compares naive persistence, linear ridge, a matched cosine-map control, Student-t GARCH(1,1), and the quantum feature model. "
        "The quantum model improves materially on GARCH and the matched cosine control but does not outperform linear ridge on the primary sample. "
        "Robustness tests on U.S. GDP growth and industrial-production growth yield similarly mixed evidence. "
        "Results therefore support quantum feature maps as an empirical benchmark, not as evidence of quantum advantage."
    ))
    add_paragraph(document, "Keywords: quantum machine learning; reservoir computing; volatility forecasting; Garman-Klass variance; Diebold-Mariano test; reproducibility.", bold_start="Keywords:")

    document.add_heading("1. Introduction", level=1)
    add_paragraph(document, (
        "Volatility forecasts inform portfolio allocation, derivatives pricing and risk control. Yet observed volatility is latent and any daily proxy is imperfect. "
        "Range-based estimators use information from the trading-day high, low, open and close, offering a richer target than squared close-to-close returns (Garman and Klass, 1980). "
        "This paper asks a narrower question than claims of quantum advantage: does a transparent small quantum feature map add predictive value over properly aligned classical controls for a public, daily equity-volatility task?"
    ))
    add_paragraph(document, (
        "Quantum reservoir computing (QRC) uses fixed quantum dynamics as a feature generator and trains only a classical readout (Fujii and Nakajima, 2017). "
        "That separation can be attractive in small-data settings, but it also makes careful baselines essential. "
        "In particular, a circuit whose measurements are unaffected by its entangling gates is only a classical deterministic feature map in quantum notation. "
        "The original repository had exactly this issue: CZ gates followed by local Pauli-Z measurements returned coordinate-wise cosine features. "
        "The redesigned implementation makes the entangling dynamics observable and includes that cosine mapping as an explicit control."
    ))
    add_paragraph(document, (
        "Three contributions follow. First, the code implements causal target alignment and rolling evaluation for next-day Garman-Klass variance. Second, it tests a non-commuting quantum feature map against naive, econometric and matched classical controls. Third, it reports negative and mixed findings directly: the QRC model beats GARCH and the old cosine control, but ridge is strongest on the primary sample."
    ))

    document.add_heading("2. Related Literature", level=1)
    add_paragraph(document, (
        "ARCH and GARCH models remain central conditional-variance benchmarks (Engle, 1982; Bollerslev, 1986). "
        "Forecast comparisons should account for noisy variance proxies; QLIKE is especially useful because it is robust to important classes of proxy noise (Patton, 2011). "
        "The Diebold-Mariano framework provides a test of equal predictive accuracy when competing forecasts are evaluated on a common realization sequence (Diebold and Mariano, 1995)."
    ))
    add_paragraph(document, (
        "QRC replaces a trained recurrent core with fixed quantum evolution and a trainable readout. Prior work demonstrates its potential for temporal learning and physical reservoirs, while also motivating conservative interpretation on small simulators (Fujii and Nakajima, 2017; Nakajima et al., 2019). "
        "Contemporary quantum-finance studies likewise caution that improved predictive performance on a benchmark is not evidence of scalable quantum advantage. This paper follows that discipline."
    ))

    document.add_heading("3. Data and Forecasting Target", level=1)
    add_paragraph(document, (
        "The primary dataset is daily SPY OHLC data downloaded through yfinance. The sample begins on 3 January 2005 and ends on 31 December 2025. "
        "Let O_t, H_t, L_t and C_t denote open, high, low and close. Daily realized variance is proxied by the Garman-Klass estimator:"
    ))
    equation = document.add_paragraph()
    equation.alignment = WD_ALIGN_PARAGRAPH.CENTER
    equation.add_run("GK_t = 0.5[ln(H_t/L_t)]² - (2 ln 2 - 1)[ln(C_t/O_t)]².").italic = True
    add_paragraph(document, (
        "At feature date t, the forecast target is GK_(t+1). The learning target is log(max(GK_(t+1), 10^-12)); forecasts are exponentiated before scoring. "
        "Five lagged log-variance values ending at t form each input. This timing prevents target leakage: no OHLC observation from t+1 appears in features or training data for its forecast."
    ))
    add_paragraph(document, (
        "Robustness tests use final-vintage FRED growth series: quarterly real GDP and monthly industrial production. Their targets are next-period absolute log growth, not range-based realized variance. "
        "They are therefore reported as low-frequency volatility-proxy tests and are not interpreted as real-time policy forecasts."
    ))

    document.add_heading("4. Quantum Feature Model and Baselines", level=1)
    add_paragraph(document, (
        "The five-dimensional lag vector is standardized using each training window only. Each scaled coordinate is encoded by Ry rotation. Two rounds of encoding are interleaved with Rx mixing and a CNOT ring. "
        "Because CNOT does not commute with all measured observables after the mixing operations, its state dynamics influence the output features. The readout concatenates five local Pauli-Z expectations and five nearest-neighbour ZZ expectations, then fits ridge regression."
    ))
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(circuit_path), width=Inches(6.6))
    caption = document.add_paragraph("Figure 1. Two-layer five-qubit quantum feature circuit. Ry encodes standardized lagged log variances; Rx and CNOT-ring layers create non-commuting dynamics; local and pairwise Z observables feed ridge regression.", style="CaptionText")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_paragraph(document, (
        "Baselines are: (i) persistence of current variance; (ii) ridge regression on lags; (iii) a cosine-ridge control reproducing the prior repository's coordinate-wise cos(Sx) mapping; and (iv) Student-t GARCH(1,1) fit to close-to-close log returns. "
        "GARCH refits before each semiannual block. All models share the same 4,278 forecast dates."
    ))

    document.add_heading("5. Evaluation Protocol", level=1)
    add_paragraph(document, (
        "An expanding window begins with 1,000 observations; models refit every 126 trading days and issue one-step forecasts between refits. "
        "No data are randomly shuffled. Main losses are RMSE, MAE and QLIKE. For model comparisons, the study uses the two-sided Diebold-Mariano statistic with a Bartlett heteroskedasticity-and-autocorrelation-consistent long-run variance estimate. "
        "Positive statistics in Table 2 mean QRC has lower squared-error loss than the stated comparator."
    ))

    document.add_heading("6. Results", level=1)
    add_paragraph(document, (
        "Table 1 reports the primary out-of-sample results. Ridge has the lowest RMSE, MAE and QLIKE. QRC improves on persistence, the matched cosine control and GARCH under all three losses, but it does not beat ridge. "
        "This result is substantively useful: it rules out interpreting an improvement over the legacy circuit as an intrinsically quantum effect, while showing that the redesigned feature map can still outperform selected econometric and nonlinear controls."
    ))
    add_table(document, metrics, "Table 1. SPY out-of-sample performance for next-day Garman-Klass variance.", digits=7)
    add_paragraph(document, (
        "Table 2 gives QRC comparisons. The QRC improvement over GARCH is statistically significant. The comparison with the cosine control is marginal at the 10% level and not conventionally significant at 5%; its comparison with ridge is unfavorable but not statistically significant at 5%. "
        "The evidence thus does not support a claim that QRC dominates best-performing classical methods on this sample."
    ))
    add_table(document, dm, "Table 2. Diebold-Mariano tests, comparator versus QRC; positive statistic favors QRC.", digits=7)
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(RUN / "forecast_plot.png"), width=Inches(6.6))
    caption = document.add_paragraph("Figure 2. Final 252 aligned SPY forecasts. The range-based target is highly spiky around stress events; all models smooth part of that variation.", style="CaptionText")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_heading("7. Macroeconomic Robustness Tests", level=1)
    add_paragraph(document, (
        "Table 3 summarizes low-frequency robustness tests. On GDP growth magnitude, cosine-ridge has the lowest RMSE and QRC is close to ridge. On industrial-production magnitude, ridge has the lowest RMSE and QRC has the lowest MAE among the reported models. "
        "These small samples do not justify broad macroeconomic claims; they only indicate that the feature-map comparison is not confined to daily equity data."
    ))
    add_table(document, macro, "Table 3. Robustness tests: next-period absolute growth magnitude (RMSE and MAE).", digits=6)

    document.add_heading("8. Limitations and Reproducibility", level=1)
    add_paragraph(document, (
        "The circuit is simulated exactly and has not been run on quantum hardware. Consequently, the results contain no gate, readout or decoherence noise and make no hardware-performance claim. "
        "The reported hyperparameters are fixed rather than selected through nested time-series cross-validation. A submission revision should add nested validation, broader asset coverage, alternative realized-volatility targets, noise simulations, and block-bootstrap uncertainty intervals."
    ))
    add_paragraph(document, (
        "Daily Garman-Klass variance is a proxy, not latent conditional variance. GARCH forecasts close-to-close conditional variance, whereas the target incorporates intraday range information. That mismatch is deliberate as an applied proxy comparison but must be interpreted cautiously. The GDP and industrial-production data are finalized vintages, so they cannot establish real-time macroeconomic usefulness."
    ))
    add_paragraph(document, (
        "All source code, YAML configuration, cached input data, dated forecasts and result tables are retained in the companion repository. The implementation includes tests for target alignment, positive QLIKE inputs, finite Diebold-Mariano statistics, reservoir sensitivity to entanglement, and matched forecast indices."
    ))

    document.add_heading("9. Conclusion", level=1)
    add_paragraph(document, (
        "A redesigned non-commuting quantum reservoir-style feature map provides a reproducible benchmark for equity-volatility forecasting. It improves on legacy cosine features and Student-t GARCH in this experiment, but a classical ridge model remains strongest on the primary dataset. "
        "The appropriate conclusion is not quantum advantage. Instead, the evidence motivates further controlled study of quantum feature maps under nested tuning, hardware noise and broader financial datasets."
    ))

    document.add_heading("References", level=1)
    references = [
        "Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307-327.",
        "Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253-263.",
        "Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica, 50(4), 987-1007.",
        "Fujii, K., & Nakajima, K. (2017). Harnessing disordered-ensemble quantum dynamics for machine learning. Physical Review Applied, 8, 024030.",
        "Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. Journal of Business, 53(1), 67-78.",
        "Nakajima, K., et al. (2019). Boosting computational power through spatial multiplexing in quantum reservoir computing. Physical Review Applied, 11, 034005.",
        "Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. Journal of Econometrics, 160(1), 246-256.",
        "Thakkar, S., et al. (2024). Improved financial forecasting via quantum machine learning. Quantum Machine Intelligence, 6, 27.",
    ]
    for reference in references:
        add_paragraph(document, reference)

    document.save(OUTPUT / "qrc_volatility_manuscript.docx")


if __name__ == "__main__":
    main()
