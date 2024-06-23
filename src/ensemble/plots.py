
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def scatter_plot(data_tuple: tuple):
    _, ax = plt.subplots()
    scatter = ax.scatter(
        data_tuple.data[:, 0], data_tuple.data[:, 1], c=data_tuple.target)
    ax.set(xlabel=data_tuple.feature_names[0],
           ylabel=data_tuple.feature_names[1])
    _ = ax.legend(
        scatter.legend_elements()[0], data_tuple.target_names, loc="lower right", title="Classes"
    )
    plt.show()


def seaborn_plot(data_tuple: tuple):
    # Convert to DataFrame for better visualization
    df = pd.DataFrame(data=np.c_[data_tuple['data'], data_tuple['target']],
                      columns=data_tuple['feature_names'] + ['target'])

    # Plot pairplot
    sns.pairplot(df, hue='target', palette='Set1')
    plt.show()
