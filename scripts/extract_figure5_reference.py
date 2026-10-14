"""Read the verified PDF vector path without copying or modifying the PDF.

Run with the existing PDF-reading runtime (pdfplumber), not the research
venv. The saved comparison reference makes this optional for normal runs.
"""

import argparse
import hashlib
import json
from pathlib import Path
from importlib import metadata
import pdfplumber

PROJECT = Path(__file__).resolve().parents[1]
PDF_SHA256 = "e142a56c57c822a45fb22aea7e7b9a47f694a9c37842796a040c9f8d4c2053b7"


def main():
    """Extract Figure 5's cyan polyline and calibrate from its main axes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.pdf.read_bytes()).hexdigest() != PDF_SHA256:
        raise ValueError("This extraction is calibrated only for the verified local PDF.")
    output = PROJECT / "papers" / "figure5_exact_reference.json"
    if output.exists():
        raise FileExistsError(f"Reference already exists: {output}")
    with pdfplumber.open(args.pdf) as pdf:
        page = pdf.pages[20]
        curves = [curve for curve in page.curves
                  if curve.get("stroking_color") == (0.302, 0.745, 0.933) and len(curve["pts"]) > 100]
        if len(curves) != 1 or any(item[0] not in ("m", "l") for item in curves[0]["path"]):
            raise ValueError("Expected a unique cyan polyline, not Bezier control points.")
        # Main-plot frame endpoints in PDF bottom-origin coordinates;
        # pts use top-origin y. Axis labels were also visually inspected.
        x0, x1, y0, y1 = 190.75963, 414.83817, 550.351725, 726.413435
        for expected in [(x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1), (x1, y0, x1, y1)]:
            if not any(all(abs(line[key] - value) < 1e-6 for key, value in zip(("x0", "y0", "x1", "y1"), expected)) for line in page.lines):
                raise ValueError("Main-plot axis calibration could not be verified.")
        points = curves[0]["pts"]
        times = [round(10 * (x - x0) / (x1 - x0), 8) for x, _ in points]
        values = [round(1 - 1.2 * (y - (page.height - y1)) / (y1 - y0), 10) for _, y in points]
        record = {
            "source": "Xiao Yuan, Suguru Endo, Qi Zhao, Ying Li and Simon C. Benjamin (2019), Theory of variational quantum simulation, Quantum 3, 191",
            "doi": "10.22331/q-2019-10-07-191",
            "source_url": "https://quantum-journal.org/papers/q-2019-10-07-191/",
            "pdf_sha256": PDF_SHA256, "pdf_page": 21, "printed_page": 21,
            "pdf_version": "Journal-formatted local PDF, accepted-date footer 2019-09-07; no unverified arXiv version label",
            "method": "pdfplumber cyan polyline vertices, stroke RGB=(0.302,0.745,0.933); visually verified main axes",
            "pdfplumber_version": metadata.version("pdfplumber"),
            "frame_pdf_points": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
            "axis_limits": {"t": [0, 10], "z0": [-0.2, 1]},
            "line_width_pdf_points": curves[0]["linewidth"],
            "notes": "Rendered line vertices are not author raw data. Time-coordinate interpretations are comparison hypotheses only.",
            "t": times, "z0": values,
        }
    output.parent.mkdir(exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(f"Saved {len(times)} comparison vertices: {output}")


if __name__ == "__main__":
    main()
