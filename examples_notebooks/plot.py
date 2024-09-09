import os
from pathlib import Path
from typing import Union

import matplotlib
import matplotlib.cm as cm
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from scipy.stats import ttest_ind

sns.set_style("ticks")
plt.style.use("seaborn-v0_8-deep")

matplotlib.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams["animation.embed_limit"] = 2**64


PARAMS_PAD = 2
PARAMS_LENGTH = 3

plt.rcParams.update(
    {
        "mathtext.default": "regular",
        "xtick.major.pad": PARAMS_PAD,
        "ytick.major.pad": PARAMS_PAD,
        "xtick.major.size": PARAMS_LENGTH,
        "ytick.major.size": PARAMS_LENGTH,
    }
)


JET = cm.get_cmap("jet")
GRAY = cm.get_cmap("gray")
TURBO = cm.get_cmap("turbo")
GRAY2RGB = TURBO(np.arange(256))[:, :3]

TICK_FONTSIZE = 8
LABEL_FONTSIZE = 9
TITLE_FONTSIZE = 10


def set_font():
    font_path = os.getenv("MATPLOTLIB_FONT")
    if font_path is not None and os.path.exists(font_path):
        font_manager.fontManager.addfont(path=font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams.update({"font.family": [prop.get_name(), "DejaVu Sans"]})


def remove_spines(axis: Axes):
    """remove all spines"""
    axis.spines["top"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)


def remove_top_right_spines(axis: Axes):
    """Remove the ticks and spines of the top and right axis"""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def set_right_label(axis: Axes, label: str, fontsize: int = None):
    """Set y-axis label on the right-hand side"""
    right_axis = axis.twinx()
    kwargs = {"rotation": 270, "va": "center", "labelpad": 3}
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    right_axis.set_ylabel(label, **kwargs)
    right_axis.set_yticks([])
    remove_top_right_spines(right_axis)


def set_xticks(
    axis: Axes,
    ticks: Union[np.ndarray, list],
    tick_labels: Union[np.ndarray, list] = None,
    label: str = "",
    tick_fontsize: int = None,
    label_fontsize: int = None,
    rotation: int = None,
    label_pad: int = 2,
    color: str = "black",
):

    kwargs = {"fontsize": tick_fontsize}
    if rotation is not None:
        kwargs["rotation"] = rotation
        kwargs["verticalalignment"] = "center"
    axis.set_xticks(ticks, labels=tick_labels, **kwargs)
    if label:
        axis.set_xlabel(label, fontsize=label_fontsize, color=color, labelpad=label_pad)


def set_yticks(
    axis: Axes,
    ticks: Union[np.ndarray, list],
    tick_labels: Union[np.ndarray, list] = None,
    label: str = "",
    tick_fontsize: int = None,
    label_fontsize: int = None,
    rotation: int = None,
    label_pad: int = 2,
):
    kwargs = {"fontsize": tick_fontsize}
    if rotation is not None:
        kwargs["rotation"] = rotation
        kwargs["verticalalignment"] = "center"
    axis.set_yticks(ticks, labels=tick_labels, **kwargs)
    if label:
        axis.set_ylabel(label, fontsize=label_fontsize, labelpad=label_pad)


def set_ticks_params(
    axis: Axes,
    length: float = PARAMS_LENGTH,
    pad: float = PARAMS_PAD,
    colors: str = "black",
):
    axis.tick_params(
        axis="both",
        which="major",
        length=length,
        pad=pad,
        colors=colors,
        width=0.8,
    )
    axis.tick_params(
        axis="both",
        which="minor",
        length=length * 0.6,
        pad=pad,
        colors=colors,
        width=0.8,
    )


def add_p_value(
    ax: Axes,
    x0: float | np.ndarray,
    x1: float | np.ndarray,
    y: float | np.ndarray,
    y_offset: float | np.ndarray,
    array1: np.ndarray,
    array2: np.ndarray,
    fontsize: float = TICK_FONTSIZE,
) -> float:
    p_value = ttest_ind(array1, array2).pvalue
    if p_value <= 0.0001:
        text = "****"
    elif p_value <= 0.001:
        text = "***"
    elif p_value <= 0.01:
        text = "**"
    elif p_value <= 0.05:
        text = "*"
    else:
        text = "n.s."

    ns = text == "n.s."

    barx = [x0, x0, x1, x1]
    bary = [y, y + y_offset, y + y_offset, y]
    ax.plot(
        barx,
        bary,
        color="black",
        linewidth=1,
        clip_on=False,
    )
    text_height = y + 1.2 * y_offset
    ax.text(
        x=((x1 - x0) / 2) + x0,
        y=text_height,
        s=text,
        ha="center",
        va="bottom" if ns else "center",
        fontsize=fontsize - 1.6 if ns else fontsize,
    )
    return text_height


def save_figure(
    figure: plt.Figure,
    filename: Path,
    dpi: int = 120,
    close: bool = True,
):
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        filename, dpi=dpi, bbox_inches="tight", pad_inches=0.02, transparent=True
    )
    if close:
        plt.close(figure)
