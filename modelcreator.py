# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:58:37 2022

@author: Piotr Neto

Main module which orchestrates model creation and outputs

Expected Threat Wyscout model based on Karun Singh's xT algorithm: https://karun.in/blog/expected-threat.html
Wyscout Dataset: https://www.nature.com/articles/s41597-019-0247-7
Wyscout Events data description: https://support.wyscout.com/matches-wyid-events

"""

#%% IMPORTS
import numpy as np
import pandas as pd
import wyscoutmodelutility as wmu
import matplotlib.pyplot as plt
import pypitch as pp

#%% LOAD AND PREPARE DATASETS
LEAGUE_COUNTRY = 'England'
df_league_events = wmu.load_league_events(LEAGUE_COUNTRY)
df_players = wmu.load_players_data()
df_teams = wmu.load_teams_data()

# Add helper columns to league events dataframe
df_league_events['X_Start'] = df_league_events.apply(lambda row: wmu.get_event_x_start_pos(row), axis=1)
df_league_events['Y_Start'] = df_league_events.apply(lambda row: wmu.get_event_y_start_pos(row), axis=1)
df_league_events['X_End'] = df_league_events.apply(lambda row: wmu.get_event_x_end_pos(row), axis=1)
df_league_events['Y_End'] = df_league_events.apply(lambda row: wmu.get_event_y_end_pos(row), axis=1)
df_league_events['isMove'] = df_league_events.apply(lambda row: wmu.is_event_move(row), axis=1)
df_league_events['isMoveAccurate'] = df_league_events.apply(lambda row: wmu.is_event_move_accurate(row), axis=1)
df_league_events['isGoal'] = df_league_events.apply(lambda row: wmu.is_event_goal(row), axis=1)

#%% MODEL CONFIG
X_BIN_SIZE=20
Y_BIN_SIZE=12
N=5 # Nr of iterations in the xT model

#%% PRODUCE MODEL MATRICES
goal_given_shot_frequency_matrix = wmu.compute_goal_given_shot_frequency_matrix(df_league_events, 
                                                                                x_bin_size=X_BIN_SIZE, 
                                                                                y_bin_size=Y_BIN_SIZE)
shot_frequency_matrix = wmu.compute_shot_frequency_matrix(df_league_events,
                                                          x_bin_size=X_BIN_SIZE, 
                                                          y_bin_size=Y_BIN_SIZE)
goal_prob_matrix = goal_given_shot_frequency_matrix / shot_frequency_matrix
move_frequency_matrix = wmu.compute_move_frequency_matrix(df_league_events,
                                                          x_bin_size=X_BIN_SIZE, 
                                                          y_bin_size=Y_BIN_SIZE)
move_prob_matrix = move_frequency_matrix / (move_frequency_matrix + shot_frequency_matrix)
shot_prob_matrix = np.ones((X_BIN_SIZE,Y_BIN_SIZE)) - move_prob_matrix
transition_prob_matrix_wrapper = wmu.compute_transition_prob_matrix_wrapper(df_league_events,
                                                                     x_bin_size=X_BIN_SIZE, 
                                                                     y_bin_size=Y_BIN_SIZE)

shot_prob_matrix = np.nan_to_num(shot_prob_matrix, nan=0)
goal_prob_matrix = np.nan_to_num(goal_prob_matrix, nan=0)
move_prob_matrix = np.nan_to_num(move_prob_matrix, nan=0)

xt_matrix = wmu.compute_xt_matrix(N,
                                  X_BIN_SIZE, 
                                  Y_BIN_SIZE,
                                  shot_prob_matrix, 
                                  goal_prob_matrix, 
                                  move_prob_matrix, 
                                  transition_prob_matrix_wrapper)

#%% INDIVIDUAL PLAYER PREDICTIONS
PLAYER_ID = 38021 # Kevin De Bruyne
player = wmu.get_player_data(df_players, PLAYER_ID)
df_player_shots_xg = wmu.compute_player_xg_given_shot_df(df_league_events, 
                                                         goal_prob_matrix, 
                                                         PLAYER_ID,
                                                         x_bin_size=X_BIN_SIZE, 
                                                         y_bin_size=Y_BIN_SIZE)
player_total_xg = wmu.compute_player_accumulated_xg(df_player_shots_xg)
print(player_total_xg)

df_player_events_xt = wmu.compute_player_event_xt_score_df(df_league_events, 
                                                           xt_matrix, 
                                                           PLAYER_ID,
                                                           x_bin_size=X_BIN_SIZE, 
                                                           y_bin_size=Y_BIN_SIZE)
player_total_xt = wmu.compute_player_accumulated_xt(df_player_events_xt)
print(player_total_xt)

#%% ALL PLAYERS PREDICTIONS IN A GIVEN LEAGUE
players_id_list = []
players_name_list = []
players_role_list = []
players_total_xt_list = []
players_nr_of_goals_list = []
players_total_xg_shot_list = []

for i, row in df_players.iterrows():
    df_player_events_xt = wmu.compute_player_event_xt_score_df(df_league_events,
                                                               xt_matrix, 
                                                               row['wyId'],
                                                               x_bin_size=X_BIN_SIZE, 
                                                               y_bin_size=Y_BIN_SIZE)
    player_total_xt = wmu.compute_player_accumulated_xt(df_player_events_xt)
    
    player_nr_of_goals = wmu.compute_player_nr_of_goals(df_league_events, row['wyId'])
    
    df_player_shots_xg = wmu.compute_player_xg_given_shot_df(df_league_events,
                                                             goal_prob_matrix, 
                                                             row['wyId'],
                                                             x_bin_size=X_BIN_SIZE, 
                                                             y_bin_size=Y_BIN_SIZE)
    player_total_xg = wmu.compute_player_accumulated_xg(df_player_shots_xg)
    
    players_total_xt_list.append(player_total_xt)
    players_nr_of_goals_list.append(player_nr_of_goals)
    players_total_xg_shot_list.append(player_total_xg)
    players_id_list.append(row['wyId'])
    players_name_list.append(row['shortName'])
    players_role_list.append(row['role']['name'])
    
df_players_metrics = pd.DataFrame({'player_id':players_id_list, 
                                    'player_name':players_name_list, 
                                    'role':players_role_list, 
                                    'total_xt':players_total_xt_list, 
                                    'nr_of_goals':players_nr_of_goals_list,
                                    'total_xg':players_total_xg_shot_list})
df_players_metrics['xt_plus_goals'] = df_players_metrics['total_xt'] + df_players_metrics['nr_of_goals']
df_players_metrics['xt_plus_xg'] = df_players_metrics['total_xt'] + df_players_metrics['total_xg']

#%% INDIVIDUAL TEAM STATS
TEAM_ID = 1609
team = wmu.get_team_data(df_teams, TEAM_ID)
df_team_shots_xg = wmu.compute_team_xg_given_shot_df(df_league_events, 
                                                     goal_prob_matrix, 
                                                     TEAM_ID,
                                                     x_bin_size=X_BIN_SIZE, 
                                                     y_bin_size=Y_BIN_SIZE)
team_total_xg = wmu.compute_team_accumulated_xg(df_team_shots_xg)
print(team_total_xg)

df_team_events_xt = wmu.compute_team_event_xt_score_df(df_league_events, 
                                                       xt_matrix, 
                                                       TEAM_ID,
                                                       x_bin_size=X_BIN_SIZE, 
                                                       y_bin_size=Y_BIN_SIZE)
team_total_xt = wmu.compute_team_accumulated_xt(df_team_events_xt)
print(team_total_xt)

#%% ALL TEAMS PREDICTIONS IN A GIVEN LEAGUE
teams_id_list = []
teams_name_list = []
teams_nr_of_goals_list = []
teams_total_xt_list = []
teams_total_xg_shot_list = []

for i, row in df_teams.iterrows():
    df_team_events_xt = wmu.compute_team_event_xt_score_df(df_league_events,
                                                           xt_matrix, 
                                                           row['wyId'],
                                                           x_bin_size=X_BIN_SIZE, 
                                                           y_bin_size=Y_BIN_SIZE)
    team_total_xt = wmu.compute_team_accumulated_xt(df_team_events_xt)
    
    team_nr_of_goals = wmu.compute_team_nr_of_goals(df_league_events, row['wyId'])
    
    df_team_shots_xg = wmu.compute_team_xg_given_shot_df(df_league_events,
                                                         goal_prob_matrix, 
                                                         row['wyId'],
                                                         x_bin_size=X_BIN_SIZE, 
                                                         y_bin_size=Y_BIN_SIZE)
    team_total_xg = wmu.compute_team_accumulated_xg(df_team_shots_xg)
    
    teams_total_xt_list.append(team_total_xt)
    teams_nr_of_goals_list.append(team_nr_of_goals)
    teams_total_xg_shot_list.append(team_total_xg)
    teams_id_list.append(row['wyId'])
    teams_name_list.append(row['name'])
    
df_teams_metrics = pd.DataFrame({'team_id':teams_id_list, 
                                  'team_name':teams_name_list, 
                                  'total_xt':teams_total_xt_list, 
                                  'nr_of_goals':teams_nr_of_goals_list,
                                  'total_xg':teams_total_xg_shot_list}).query('total_xt > 0')
df_teams_metrics['xt_plus_goals'] = df_teams_metrics['total_xt'] + df_teams_metrics['nr_of_goals']
df_teams_metrics['xt_plus_xg'] = df_teams_metrics['total_xt'] + df_teams_metrics['total_xg']

#%% PLOT CONFIG

# Limits for the extent
PITCH_X_LENGTH = 120
PITCH_Y_WIDTH = 80

#%% PLOT GOAL PROBABILTY GRID
(fig,ax) = pp.plot_pitch(PITCH_X_LENGTH, PITCH_Y_WIDTH)

im = ax.imshow(goal_prob_matrix.T, extent=[0,PITCH_X_LENGTH,0,PITCH_Y_WIDTH], origin='upper', interpolation='None', cmap=plt.cm.Reds)
fig.colorbar(im, ax=ax, fraction=0.031)
fig.suptitle('Goal given shot probability grid', fontsize=40, y=0.9)
fig.supxlabel('League: ' + str(LEAGUE_COUNTRY) + ', 2017/2018 ' + ', Grid size: ' + str(X_BIN_SIZE) + 'x' + str(Y_BIN_SIZE), fontsize=20, y=0.1)

pp.show_plt_grid_values(ax, 
                        goal_prob_matrix.T, 
                        X_BIN_SIZE, 
                        Y_BIN_SIZE, 
                        PITCH_X_LENGTH, 
                        PITCH_Y_WIDTH)
pp.show_plt_pitch_dir_arrow(ax)
plt.savefig('plots/goal_prob_grid_' + str(LEAGUE_COUNTRY) + '.png', bbox_inches='tight')

#%% PLOT MOVE PROBABILITY
(fig,ax) = pp.plot_pitch(PITCH_X_LENGTH, PITCH_Y_WIDTH)

im = ax.imshow(move_prob_matrix.T, extent=[0,PITCH_X_LENGTH,0,PITCH_Y_WIDTH], origin='upper', interpolation='None', cmap=plt.cm.Greens)
fig.colorbar(im, ax=ax, fraction=0.031)
fig.suptitle('Move probablity grid', fontsize=40, y=0.9)
fig.supxlabel('League: ' + str(LEAGUE_COUNTRY) + ', 2017/2018 ' + ', Grid size: ' + str(X_BIN_SIZE) + 'x' + str(Y_BIN_SIZE), fontsize=20, y=0.1)

pp.show_plt_grid_values(ax, move_prob_matrix.T, X_BIN_SIZE, Y_BIN_SIZE, PITCH_X_LENGTH, PITCH_Y_WIDTH)
pp.show_plt_pitch_dir_arrow(ax)
plt.savefig('plots/move_prob_grid_' + str(LEAGUE_COUNTRY) + '.png', bbox_inches='tight')

#%% PLOT SHOT PROBABILITY GRID
(fig,ax) = pp.plot_pitch(PITCH_X_LENGTH, PITCH_Y_WIDTH)

im = ax.imshow(shot_prob_matrix.T, extent=[0,PITCH_X_LENGTH,0,PITCH_Y_WIDTH], origin='upper', interpolation='None', cmap=plt.cm.Blues)
fig.colorbar(im, ax=ax, fraction=0.031)
fig.suptitle('Shot probablity grid', fontsize=40, y=0.9)
fig.supxlabel('League: ' + str(LEAGUE_COUNTRY) + ', 2017/2018 ' + ', Grid size: ' + str(X_BIN_SIZE) + 'x' + str(Y_BIN_SIZE), fontsize=20, y=0.1)

pp.show_plt_grid_values(ax, shot_prob_matrix.T, X_BIN_SIZE, Y_BIN_SIZE, PITCH_X_LENGTH, PITCH_Y_WIDTH)
pp.show_plt_pitch_dir_arrow(ax)
plt.savefig('plots/shot_prob_grid_' + str(LEAGUE_COUNTRY) + '.png', bbox_inches='tight')

#%% PLOT TRANSITION PROBABILITY GRID FOR A GIVEN CELL

# Cell coordinate
X=3
Y=6

(fig,ax) = pp.plot_pitch(PITCH_X_LENGTH, PITCH_Y_WIDTH)

im = ax.imshow(transition_prob_matrix_wrapper[X-1][Y-1].T, extent=[0,PITCH_X_LENGTH,0,PITCH_Y_WIDTH], origin='upper', interpolation='None', cmap=plt.cm.Greens)
fig.colorbar(im, ax=ax, fraction=0.031)
fig.suptitle('Transition probablity grid for cell: (' + str(X) + ',' + str(Y) +')' , fontsize=40, y=0.9)
fig.supxlabel('League: ' + str(LEAGUE_COUNTRY) + ', 2017/2018 ' + ', Grid size: ' + str(X_BIN_SIZE) + 'x' + str(Y_BIN_SIZE), fontsize=20, y=0.1)

currentCell = plt.Rectangle(((X-1)*PITCH_X_LENGTH/X_BIN_SIZE,PITCH_Y_WIDTH-Y*PITCH_Y_WIDTH/Y_BIN_SIZE), PITCH_X_LENGTH/X_BIN_SIZE, PITCH_Y_WIDTH/Y_BIN_SIZE, facecolor='none', edgecolor='violet', linewidth=3)
currentCell.set_alpha(.8)
ax.add_patch(currentCell)
pp.show_plt_grid_values(ax, transition_prob_matrix_wrapper[X-1][Y-1].T, X_BIN_SIZE, Y_BIN_SIZE, PITCH_X_LENGTH, PITCH_Y_WIDTH)
pp.show_plt_pitch_dir_arrow(ax)
plt.savefig('plots/transition_prob_grid_' + str(LEAGUE_COUNTRY) + '.png', bbox_inches='tight')

#%% PLOT EXPECTED THREAT (xT)
(fig,ax) = pp.plot_pitch(PITCH_X_LENGTH, PITCH_Y_WIDTH)

im = ax.imshow(xt_matrix.T, extent=[0,PITCH_X_LENGTH,0,PITCH_Y_WIDTH], origin='upper', interpolation='None', cmap=plt.cm.Oranges)
fig.colorbar(im, ax=ax, fraction=0.031)
fig.suptitle('Expected threat (xT) grid based on ' + str(N) + ' iterations', fontsize=40, y=0.9)
fig.supxlabel('League: ' + str(LEAGUE_COUNTRY) + ', 2017/2018 ' + ', Grid size: ' + str(X_BIN_SIZE) + 'x' + str(Y_BIN_SIZE), fontsize=20, y=0.1)

pp.show_plt_grid_values(ax, xt_matrix.T, X_BIN_SIZE, Y_BIN_SIZE, PITCH_X_LENGTH, PITCH_Y_WIDTH)
pp.show_plt_pitch_dir_arrow(ax)
plt.savefig('plots/xt_grid_' + str(LEAGUE_COUNTRY) + '.png', bbox_inches='tight')
