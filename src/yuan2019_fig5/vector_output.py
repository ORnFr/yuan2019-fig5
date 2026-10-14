"""Write genuine SVG/PDF figures, rejecting raster images and rasterized artists."""

from pathlib import Path
import xml.etree.ElementTree as ET
from .runtime import configure_project

configure_project()

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage, FigureImage, BboxImage


def save_vector_figure(figure, basename):
    """Save the same figure as SVG and PDF; return two paths without overwriting.

    basename is a suffix-free path. Curves, markers and glyphs are vector
    objects; PDF embeds TrueType fonts and SVG stores glyph outlines.
    Reject image artists and explicit rasterization rather than placing a
    bitmap inside a vector-format wrapper. Pixel DPI is not a figure-quality
    setting for these outputs. Existing artifacts are never overwritten.
    """
    basename = Path(basename)
    if basename.suffix:
        raise ValueError("Use an output basename without a file extension.")
    paths = [basename.with_suffix(suffix) for suffix in (".svg", ".pdf")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Vector outputs must not overwrite existing files.")
    for artist in figure.findobj():
        if isinstance(artist, (AxesImage, FigureImage, BboxImage)) or artist.get_rasterized():
            raise ValueError("Vector figures cannot contain raster images or rasterized artists.")
    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "path"}):
        for path in paths:
            figure.savefig(path, format=path.suffix[1:])
    # Keep generated SVG text clean in Git; newline whitespace still separates
    # path commands, so this does not change any geometry or rasterize content.
    paths[0].write_text("\n".join(line.rstrip() for line in paths[0].read_text().splitlines()) + "\n")
    root = ET.parse(paths[0]).getroot()
    if any(element.tag.rsplit("}", 1)[-1] == "image" for element in root.iter()):
        raise RuntimeError("SVG unexpectedly contains a raster image.")
    return paths
