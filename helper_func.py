import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def prediction_plot(real_val,predic_val,ax_lim_low,ax_lim_high,majr_tick,mnr_tick,ax_label):
    #Plot estiamted vs real values
    df_diam = pd.DataFrame({'actual':real_val, 'predicted':predic_val})

    sns.set_style("ticks", {"xtick.major.size": 200, "ytick.major.size": 1})

    fig, ax = plt.subplots(figsize=(12,8))



    # Make a plot with major ticks that are multiples of 0.025 and minor ticks that
    # are multiples of 0.0125.  Label major ticks with '.0f' formatting but don't label
    # minor ticks.  The string is used directly, the `StrMethodFormatter` is
    # created automatically.
    ax.xaxis.set_major_locator(MultipleLocator(majr_tick))
    ax.yaxis.set_major_locator(MultipleLocator(majr_tick))

    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(MultipleLocator(mnr_tick))
    ax.yaxis.set_minor_locator(MultipleLocator(mnr_tick))

    #Move tickmarks inside of the plot
    ax.tick_params(axis="both",direction="in",which="both", pad=10, colors='black')

    #Change tick mark labels font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    #Create actual seborn graph
    graph=sns.regplot(data=df_diam, x="actual", y="predicted",ci=95,x_bins=40, scatter_kws={"color": "black",'s':50},line_kws={"color": "black"})#, fit_reg=False

    # control x and y limits
    plt.ylim(ax_lim_low, ax_lim_high)
    plt.xlim(ax_lim_low, ax_lim_high)

    #Change axis labels
    plt.xlabel("Actual "+ax_label+" (a.u.)",fontsize=14, labelpad=10)
    plt.ylabel("Predicted "+ax_label+" (a.u.)",fontsize=14, labelpad=10)

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black') 
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')

    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')


    #specify secondary axis similar to primary repeat all that we did above
    #Xaxis first
    secax = ax.secondary_xaxis('top')
    secax.set_xlim([ax_lim_low, ax_lim_high])
    secax.tick_params(axis="both",direction="in",which="both", pad=10, colors='black')
    secax.xaxis.set_major_locator(MultipleLocator(majr_tick))
    secax.xaxis.set_minor_locator(MultipleLocator(mnr_tick))
    secax.set_xticklabels([]);

    #Yaxis second
    secax2 = ax.secondary_yaxis('right')
    secax2.set_ylim([ax_lim_low, ax_lim_high])
    secax2.tick_params(axis="both",direction="in",which="both", pad=10, colors='black')
    secax2.yaxis.set_major_locator(MultipleLocator(majr_tick))
    secax2.yaxis.set_minor_locator(MultipleLocator(mnr_tick))
    secax2.set_yticklabels([]);