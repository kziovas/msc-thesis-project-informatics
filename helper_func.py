import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score,top_k_accuracy_score, mean_squared_error, mean_absolute_error
from multiprocessing import Pool
from functools import partial
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

def prediction_plot(real_val,predic_val,ax_lim_low,ax_lim_high,majr_tick,mnr_tick,ax_label):
    #Plot estiamted vs real values
    df= pd.DataFrame({'actual':real_val, 'predicted':predic_val})

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
    graph=sns.regplot(data=df, x="actual", y="predicted",ci=95,x_bins=40, scatter_kws={"color": "black",'s':50},line_kws={"color": "black"})#, fit_reg=False

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
    
    #Find data length in order to create dummy data for diagonal "ideal" line
    df_length=len(df)
    df_diag = pd.DataFrame({ 'x_diag' : np.arange(ax_lim_low,ax_lim_high + majr_tick ,majr_tick),
    'y_diag' : np.arange(ax_lim_low,ax_lim_high + majr_tick ,majr_tick) })
    
    diag=sns.lineplot(data=df_diag, x="x_diag", y="y_diag", color='grey',linestyle='--')
    
def data_prep(df):
    #Split label from features
    X = df.drop(['mean','sd'],axis=1)
    y = df[['mean','sd']]
    #Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
    #Scale data
    scaler = MinMaxScaler()#StandardScaler()#
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    
    return (df,X,y, X_train, X_test, y_train, y_test,scaled_X_train,scaled_X_test)

def model_eval(model,scaled_X_train,y_train,scaled_X_test,y_test):
    #Fit model
    fit_start = time.time()
    model.fit(scaled_X_train,y_train)
    fit_end = time.time()
    fit_time=fit_end-fit_start
    
    #Predict results
    pred_start = time.time()
    y_pred=model.predict(scaled_X_test)
    pred_end = time.time()
    pred_time=pred_end-pred_start
    
    # Evaluate the regressor
    mse_one = mean_squared_error((y_test['mean']-1)/10, (y_pred[:,0]-1)/10)
    mse_two = mean_squared_error(y_test['sd']*4, y_pred[:,1]*4)

    mae_one = mean_absolute_error((y_test['mean']-1)/10, (y_pred[:,0]-1)/10)
    mae_two = mean_absolute_error(y_test['sd']*4, y_pred[:,1]*4)
    
    return(fit_time,pred_time,mse_one,mse_two,mae_one,mae_two)

def analysis_func_expnum(exp,model,noise):
    csv_path=f"./data/train_data{exp}_noise{noise}.csv"
    df= pd.read_csv(csv_path,index_col=0)
    #Data preparation step
    df,X,y, X_train, X_test, y_train, y_test,scaled_X_train,scaled_X_test=data_prep(df)
    fit_time,pred_time,mse_one,mse_two,mae_one,mae_two=model_eval(model,scaled_X_train,y_train,scaled_X_test,y_test)
    results=(exp, fit_time,pred_time,mse_one,mse_two,mae_one,mae_two)
    return (results)

def analysis_func_layers(layers,num_exprs,noise):
    
    #Define final MLP
    model_single = MLPRegressor(random_state=1, max_iter=500,tol=0.001,early_stopping=True,activation='relu',alpha=0.000075,
                   hidden_layer_sizes=layers,solver='adam',beta_1=0.5,beta_2=0.05,learning_rate='constant')
    model = MultiOutputRegressor(model_single)
    results=list(analysis_func_expnum(num_exprs,model,noise))
    results[0]=layers
    results=tuple(results)
    return  results

def analysis_func_noise(noise,exp,model):
    csv_path=f"./data/train_data{exp}_noise{noise}.csv"
    df= pd.read_csv(csv_path,index_col=0)
    #Data preparation step
    df,X,y, X_train, X_test, y_train, y_test,scaled_X_train,scaled_X_test=data_prep(df)
    fit_time,pred_time,mse_one,mse_two,mae_one,mae_two=model_eval(model,scaled_X_train,y_train,scaled_X_test,y_test)
    results=(noise, fit_time,pred_time,mse_one,mse_two,mae_one,mae_two)
    return (results)

def perform_plot(df):
    sns.set(font_scale = 2)
    
    y_vars = df.columns[1:]
    x_vars = df.columns[0]
    for y_var in y_vars:
        sns.set_style("ticks", {"xtick.major.size": 200, "ytick.major.size": 1})
        

        
        fig, ax = plt.subplots(figsize=(12,8))
        sns.lineplot(data=df,x=df[x_vars],y=df[y_var],color = 'black',palette=['black'])
        
        #Move tickmarks inside of the plot
        ax.tick_params(axis="both",direction="in",which="both", pad=10, colors='black')
        
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black') 
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')

        ax.yaxis.label.set_color('black')
        ax.xaxis.label.set_color('black')
        
        #specify secondary axis similar to primary repeat all that we did above
        #Xaxis first
        secax = ax.secondary_xaxis('top')
        secax.tick_params(axis="both",direction="in",which="both", pad=10, colors='black')
        secax.set_xticklabels([]);

        #Yaxis second
        secax2 = ax.secondary_yaxis('right')
        secax2.tick_params(axis="both",direction="in",which="both", pad=10, colors='black')
        secax2.set_yticklabels([]);