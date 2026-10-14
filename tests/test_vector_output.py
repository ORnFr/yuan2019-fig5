"""Verify vector-only outputs and rejection of bitmap/rasterized content."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yuan2019_fig5.vector_output import save_vector_figure
import matplotlib.pyplot as plt


def test_vector_paths_and_no_overwrite(tmp_path):
    """SVG must contain drawing paths without images; PDF must be a real PDF."""
    figure, axis = plt.subplots()
    axis.plot([0, 1], [1, 0], marker="o")
    axis.set_xlabel("Time")
    try:
        svg, pdf = save_vector_figure(figure, tmp_path / "curve")
        elements = list(ET.parse(svg).getroot().iter())
        assert any(element.tag.endswith("}path") for element in elements)
        assert not any(element.tag.endswith("}image") for element in elements)
        assert pdf.read_bytes().startswith(b"%PDF-")
        with pytest.raises(FileExistsError):
            save_vector_figure(figure, tmp_path / "curve")
    finally:
        plt.close(figure)


@pytest.mark.parametrize("bitmap", [True, False])
def test_raster_content_is_rejected(tmp_path, bitmap):
    """Do not allow a vector extension to disguise bitmap plot content."""
    figure, axis = plt.subplots()
    if bitmap:
        axis.imshow([[0, 1], [1, 0]])
    else:
        axis.plot([0, 1], [0, 1], rasterized=True)
    try:
        with pytest.raises(ValueError, match="raster"):
            save_vector_figure(figure, tmp_path / "invalid")
        assert not list(tmp_path.glob("invalid.*"))
    finally:
        plt.close(figure)
