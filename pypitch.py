# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 17:52:02 2022

@author: Piotr Neto

Modified code originally copied from @JPJ_dejong and @davsu428

"""
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np

def plot_pitch(pitch_x_length, pitch_y_width, figure_size_multiplier=2):
    
    LINECOLOR = 'black'
    LINEWIDTH = 1
    
    fig = plt.figure(figsize=(12*figure_size_multiplier, 8*figure_size_multiplier))
    ax = fig.add_subplot(1,1,1) 
    
    #Pitch Outline & Centre Line
    plt.plot([0,0],[0,pitch_y_width], color=LINECOLOR, linewidth=LINEWIDTH)
    plt.plot([0,pitch_x_length],[pitch_y_width,pitch_y_width], color=LINECOLOR, linewidth=LINEWIDTH)
    plt.plot([pitch_x_length,pitch_x_length],[pitch_y_width,0], color=LINECOLOR, linewidth=LINEWIDTH)
    plt.plot([pitch_x_length,0],[0,0], color=LINECOLOR, linewidth=LINEWIDTH)
    plt.plot([pitch_x_length/2,pitch_x_length/2],[0,pitch_y_width], color=LINECOLOR, linewidth=LINEWIDTH)
    
    #Left Penalty Area
    plt.plot([18 ,18],[(pitch_y_width/2 +18),(pitch_y_width/2-18)],color=LINECOLOR, linewidth=LINEWIDTH)
    plt.plot([0,18],[(pitch_y_width/2 +18),(pitch_y_width/2 +18)],color=LINECOLOR, linewidth=LINEWIDTH)
    plt.plot([18,0],[(pitch_y_width/2 -18),(pitch_y_width/2 -18)],color=LINECOLOR, linewidth=LINEWIDTH)
    
    #Right Penalty Area
    plt.plot([(pitch_x_length-18),pitch_x_length],[(pitch_y_width/2 +18),(pitch_y_width/2 +18)],color=LINECOLOR, linewidth=LINEWIDTH)
    plt.plot([(pitch_x_length-18), (pitch_x_length-18)],[(pitch_y_width/2 +18),(pitch_y_width/2-18)],color=LINECOLOR, linewidth=LINEWIDTH)
    plt.plot([(pitch_x_length-18),pitch_x_length],[(pitch_y_width/2 -18),(pitch_y_width/2 -18)],color=LINECOLOR, linewidth=LINEWIDTH)
    
    #Left 6-yard Box
    plt.plot([0,6],[(pitch_y_width/2+7.32/2+6),(pitch_y_width/2+7.32/2+6)],color=LINECOLOR, linewidth=LINEWIDTH)
    plt.plot([6,6],[(pitch_y_width/2+7.32/2+6),(pitch_y_width/2-7.32/2-6)],color=LINECOLOR, linewidth=LINEWIDTH)
    plt.plot([6,0],[(pitch_y_width/2-7.32/2-6),(pitch_y_width/2-7.32/2-6)],color=LINECOLOR, linewidth=LINEWIDTH)
    
    #Right 6-yard Box
    plt.plot([pitch_x_length,pitch_x_length-6],[(pitch_y_width/2+7.32/2+6),(pitch_y_width/2+7.32/2+6)],color=LINECOLOR, linewidth=LINEWIDTH)
    plt.plot([pitch_x_length-6,pitch_x_length-6],[(pitch_y_width/2+7.32/2+6),pitch_y_width/2-7.32/2-6],color=LINECOLOR, linewidth=LINEWIDTH)
    plt.plot([pitch_x_length-6,pitch_x_length],[(pitch_y_width/2-7.32/2-6),pitch_y_width/2-7.32/2-6],color=LINECOLOR, linewidth=LINEWIDTH)
    
    #Prepare Circles; 10 yards distance. penalty on 12 yards
    centreCircle = plt.Circle((pitch_x_length/2,pitch_y_width/2),10,color=LINECOLOR, linewidth=LINEWIDTH,fill=False)
    centreSpot = plt.Circle((pitch_x_length/2,pitch_y_width/2),0.4,color=LINECOLOR, linewidth=LINEWIDTH)
    leftPenSpot = plt.Circle((12,pitch_y_width/2),0.4,color=LINECOLOR, linewidth=LINEWIDTH)
    rightPenSpot = plt.Circle((pitch_x_length-12,pitch_y_width/2),0.4,color=LINECOLOR, linewidth=LINEWIDTH)
    
    #Draw Circles
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)
    
    #Prepare Arcs
    leftArc = Arc((11.4,pitch_y_width/2),height=20,width=20,angle=0,theta1=312,theta2=48,color=LINECOLOR, linewidth=LINEWIDTH)
    rightArc = Arc((pitch_x_length-11.4,pitch_y_width/2),height=20,width=20,angle=0,theta1=130,theta2=230,color=LINECOLOR, linewidth=LINEWIDTH)
    
    #Draw Arcs
    ax.add_patch(leftArc)
    ax.add_patch(rightArc)
    
    
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    ax.set_xlim([0,pitch_x_length]) 
    ax.set_ylim([0,pitch_y_width])
     
    return fig,ax

def show_plt_grid_values(ax,
                         pitch_value_matrix,
                         x_bin_size=12, 
                         y_bin_size=12, 
                         pitch_x_length=120, 
                         pitch_y_width=80):
    # Add the text by iterating x,y positions on the pitch grid
    jump_x = pitch_x_length / (2.0 * x_bin_size)
    jump_y = pitch_y_width / (2.0 * y_bin_size)
    x_positions = np.linspace(start=0, stop=pitch_x_length, num=x_bin_size, endpoint=False)
    y_positions = np.linspace(start=0, stop=pitch_y_width, num=y_bin_size, endpoint=False)

    for y_index, y in enumerate(np.flip(y_positions,0)):
        for x_index, x in enumerate(x_positions):
            text_x = x + jump_x
            text_y = y + jump_y
            label = 0.00
            if not(np.isnan(pitch_value_matrix[y_index, x_index])):
                label = round(pitch_value_matrix[y_index, x_index],3)
            ax.text(text_x, text_y, label, color='black', ha='center', va='center')

def show_plt_pitch_dir_arrow(ax):
    pitch_dir_arrow = plt.arrow(x=18, y=40, dx=12, dy=0, linewidth=15, width=3, length_includes_head=False, head_length = 4, edgecolor='none', facecolor='grey', alpha=0.2)
    ax.add_patch(pitch_dir_arrow)