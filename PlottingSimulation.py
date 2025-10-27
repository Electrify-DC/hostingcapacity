import mplcursors
from matplotlib import pyplot as plt


class PlottingSimulation:
    """plot stuff"""

    def __init__(self):
        self.font_size = 15
        self.fig_size = (9, 3)
        self.time_steps = 48

    def plot_profiles(
        self,
        profiles,
        title: str,
        ylabel: str,
        label=None,
        line_style=None,
        y_lim=None,
        x_upper_lim=None,
    ):
        x = list(range(24))
        l = [f"{hh}:00" if hh % 2 == 0 else " " for hh in range(len(x))]
        plt.figure(figsize=self.fig_size)
        for i, data in enumerate(profiles):
            plt.plot(
                data,
                label=label[i] if label else "profiles",
                linestyle=line_style[i] if line_style else "-",
            )
            plt.xlabel("Time (hh:mm)", fontsize=self.font_size)
            plt.xticks(fontsize=10)
            plt.xticks(x, l, fontsize=10, rotation=60)
            plt.ylabel(ylabel, fontsize=self.font_size)
            plt.yticks(fontsize=10)
            plt.title(title, fontsize=self.font_size)
            if y_lim:
                plt.ylim(y_lim)
            if x_upper_lim:
                plt.axhline(x_upper_lim, linewidth=1, linestyle="dashed", color="r")
            if label:
                plt.legend()
        plt.show()
