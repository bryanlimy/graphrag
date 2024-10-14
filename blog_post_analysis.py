import pickle
from fileinput import filename
from pathlib import Path
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

from graphrag.query.structured_search.global_search.search import GlobalSearchResult
from dynamic_search import AP_NEWS_QUESTIONS, estimate_cost
import plot

OUTPUT_DIR = Path("results") / "AP_news"

PLOT_DIR = Path("results") / "figures" / "blog_post"
DPI = 300
TICK_FONTSIZE = 8
LABEL_FONTSIZE = 9
TITLE_FONTSIZE = 10

plot.set_font()


def load_result(
    method: str, question_type: str, question_id: int
) -> GlobalSearchResult:
    with open(
        OUTPUT_DIR / question_type / method / f"question_{question_id:02d}.pkl", "rb"
    ) as f:
        return pickle.load(f)


def write_responses(responses: str, method: str, question_type: str):
    with open(
        OUTPUT_DIR / question_type / method / "responses.md", "w", encoding="utf-8"
    ) as f:
        f.write(responses)


def append_response(question_id: int, question: str, result: GlobalSearchResult) -> str:
    return (
        f"# Question {question_id} {question}\n\n"
        f"{result.response}\n\n"
        f"-----------------------------------------------\n\n"
    )


def compare_boxplot(
    values1: dict[str, list[float]],
    values2: dict[str, list[float]],
    method1: str,
    method2: str,
    y_label: str,
    filename: Path,
):
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(1.5, 1.2), dpi=DPI)

    data = [
        np.array(list(values1.values())).flatten(),
        np.array(list(values2.values())).flatten(),
    ]

    x = np.array([0, 0.4])
    linewidth = 0.8
    bp = ax.boxplot(
        data,
        notch=False,
        vert=True,
        positions=x,
        widths=0.17,
        # patch_artist=True,
        showfliers=True,
        showmeans=True,
        boxprops={"linewidth": linewidth},
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.5},
        capprops={"linewidth": linewidth, "clip_on": False},
        whiskerprops={"linewidth": linewidth, "clip_on": False},
        meanprops={
            # "marker": "o",
            "markersize": 4,
            "markerfacecolor": "gold",
            "markeredgecolor": "black",
            "markeredgewidth": 0.75,
        },
        medianprops={
            "color": "red",
            # "zorder": 1,
            "solid_capstyle": "projecting",
            "linewidth": 1,
        },
    )
    min_value = 0
    max_value = max([whi.get_ydata()[1] for whi in bp["whiskers"]])
    # get the next multiple of 2
    max_value = ceil(max_value / 2) * 2

    # compute p-value
    plot.add_p_value(
        ax,
        x0=x[0],
        x1=x[1],
        y=max_value * 1.02,
        y_offset=max_value * 0.013,
        array1=data[0],
        array2=data[1],
        fontsize=TICK_FONTSIZE,
        tick_length=max_value * 0.02,
        linewidth=linewidth,
    )

    ax.set_xlim(x[0] - 0.2, x[-1] + 0.2)
    plot.set_xticks(
        ax,
        ticks=x,
        tick_labels=[method1, method2],
        label="",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )

    y_ticks = np.linspace(min_value, max_value, 5, dtype=int)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=y_ticks,
        label=y_label,
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )

    plot.set_ticks_params(ax, length=2, pad=1)
    ax.tick_params(axis="x", which="major", length=0, pad=3)
    sns.despine(ax=ax, bottom=True, trim=True)

    plot.save_figure(figure, filename=filename, dpi=4 * DPI)


def compare(method1: str, method2: str):
    create_dict = lambda: {k: [] for k in AP_NEWS_QUESTIONS.keys()}
    costs1, costs2 = create_dict(), create_dict()
    elapses1, elapses2 = create_dict(), create_dict()

    for question_type in AP_NEWS_QUESTIONS.keys():
        responses1, responses2 = "", ""
        for question_id, question in AP_NEWS_QUESTIONS[question_type].items():
            result1 = load_result(method1, question_type, question_id)
            result2 = load_result(method2, question_type, question_id)
            responses1 += append_response(question_id, question, result1)
            responses2 += append_response(question_id, question, result2)
            cost1 = estimate_cost(result1, dynamic_search="dynamic" in method1)
            cost2 = estimate_cost(result2, dynamic_search="dynamic" in method2)
            costs1[question_type].append(cost1)
            costs2[question_type].append(cost2)
            elapses1[question_type].append(result1.completion_time)
            elapses2[question_type].append(result2.completion_time)
            print(
                f"Question {question_id} {question}\n"
                f"Cost {method1}: ${cost1:.02f}\tElapse: {result1.completion_time:.0f}s\n"
                f"Cost {method2}: ${cost2:.02f}\tElapse: {result2.completion_time:.0f}s\n"
            )
        print(
            f"Average cost over {question_type} questions:\n"
            f"\t{method1}: ${np.mean(costs1[question_type]):.02f} ({sem(costs1[question_type]):.02f})\n"
            f"\t{method2}: ${np.mean(costs2[question_type]):.02f} ({sem(costs2[question_type]):.02f})"
        )
        print(
            f"Average elapse {question_type} questions:\n"
            f"\t{method1}: {np.mean(elapses1[question_type]):.0f}s ({sem(elapses1[question_type]):.0f})\n"
            f"\t{method2}: {np.mean(elapses2[question_type]):.0f}s ({sem(elapses2[question_type]):.0f})\n"
        )
        print("--------------------------------------------------------")
        write_responses(responses1, method1, question_type)
        write_responses(responses2, method2, question_type)
    print(
        f"Overall average cost:\n"
        f"\t{method1}: ${np.mean(list(costs1.values())):.02f} "
        f"({sem(list(costs1.values()), axis=None):.02f})\n"
        f"\t{method2}: ${np.mean(list(costs2.values())):.02f} "
        f"({sem(list(costs2.values()), axis=None):.02f})"
    )
    print(
        f"Overall elapse:\n"
        f"\t{method1}: {np.mean(list(elapses1.values())):.0f}s "
        f"({sem(list(elapses1.values()), axis=None):.0f})\n"
        f"\t{method2}: {np.mean(list(elapses2.values())):.0f}s "
        f"({sem(list(elapses2.values()), axis=None):.0f})\n"
    )

    compare_boxplot(
        values1=costs1,
        values2=costs2,
        method1=method1,
        method2=method2,
        y_label="Cost ($)",
        filename=PLOT_DIR / f"cost_{method1}_{method2}.jpg",
    )
    compare_boxplot(
        values1=elapses1,
        values2=elapses2,
        method1=method1,
        method2=method2,
        y_label="Elapse (s)",
        filename=PLOT_DIR / f"elapse_{method1}_{method2}.jpg",
    )


if __name__ == "__main__":
    compare(method1="dynamic_2", method2="fixed")
    compare(method1="dynamic", method2="fixed")
