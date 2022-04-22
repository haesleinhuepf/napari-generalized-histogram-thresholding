"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""


def generalized_histogram_threshold(image: "napari.types.ImageData", nu=0, tau=0, kappa=0, omega=0.5) -> "napari.types.LabelsData":
    """

    Otsu: nu=128 (large number), tau=0.01
    Min-Error: nu=0, kappa=0, tau and omega irellevant

    See also
    --------
    https://arxiv.org/pdf/2007.07350.pdf
    """
    import numpy as np
    from ._ght import GHT

    hist, hist_edges = np.histogram(image, bins=255)
    hist_center = (hist_edges[:-1] + hist_edges[1:]) / 2

    threshold, counts = GHT(hist, hist_center, nu, tau, kappa, omega)
    print("Threshold", threshold)
    return image > threshold
