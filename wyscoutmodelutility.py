# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:21:59 2022

@author: Piotr Neto

Helper functions to compute Expected Threat model using static Wyscout data.

Model based on Karun Singh's xT algorithm: https://karun.in/blog/expected-threat.html


"""


import json
import pandas as pd
import numpy as np

def load_league_events(country):
    with open('../wyscoutdata/events/events_' + country +'.json') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def create_event_df(df_events, event_name):
    df_events = df_events[df_events['eventName']==event_name]
    return df_events


def create_sub_event_df(df_events, sub_event_name):
    df_sub_events = df_events[df_events['subEventName']==sub_event_name]
    return df_sub_events

def is_event_goal(event):
    if not (event['eventName'] == 'Shot' or event['eventName'] == 'Free Kick'):
        return False
    for tag in event['tags']:
        if (tag['id'] == 101):
            return True
    return False

def is_event_move(event):
    
    # Move defined as pass or attempt to dribble
    if (event['eventName'] == 'Pass' or event['subEventName'] == 'Ground attacking duel'):
        return True
    return False

def is_event_move_accurate(event):
    
    # Move defined as pass or attempt to dribble
    if (event['eventName'] == 'Pass' or event['subEventName'] == 'Ground attacking duel'):
        for tag in event['tags']:
            if (tag['id'] == 1801):
                return True
    return False

def get_event_x_start_pos(event):
    return event['positions'][0]['x']

def get_event_y_start_pos(event):
    return event['positions'][0]['y']

def get_event_x_end_pos(event):
    if (len(event['positions']) > 1):
        return event['positions'][1]['x']
    return float("NaN")

def get_event_y_end_pos(event):
    if (len(event['positions']) > 1):
        return event['positions'][1]['y']
    return float("NaN")

def compute_shot_frequency_matrix(df_league_events,
                                  x_bin_size,
                                  y_bin_size):
    df_shots = create_sub_event_df(df_league_events, 'Shot')

    df_shots_model_matrix = pd.DataFrame(columns=['X_Start','Y_Start','isGoal'])

    for i,shot in df_shots.iterrows():
        
        is_header=False
        for tag in shot['tags']:
            if tag['id']==403:
                is_header=True
        
        if not(is_header):
            df_shots_model_matrix.at[i,'X_Start']=shot['X_Start']
            df_shots_model_matrix.at[i,'Y_Start']=shot['Y_Start']
            df_shots_model_matrix.at[i,'isGoal']=shot['isGoal']
        
    histo_2d_shots = np.histogram2d(df_shots_model_matrix['X_Start'],df_shots_model_matrix['Y_Start'],bins=[x_bin_size,y_bin_size], range=[[0, 100],[0, 100]])
    
    # Transpose resulting matrix to match matplot coordinate system
    return histo_2d_shots[0]

def compute_goal_given_shot_frequency_matrix(df_league_events,
                                             x_bin_size,
                                             y_bin_size):
    df_shots = create_sub_event_df(df_league_events, 'Shot')

    df_shots_model_matrix = pd.DataFrame(columns=['X_Start','Y_Start','isGoal'])

    for i,shot in df_shots.iterrows():
        
        is_header=False
        for tag in shot['tags']:
            if tag['id']==403:
                is_header=True
        
        if not(is_header):
            df_shots_model_matrix.at[i,'X_Start']=shot['X_Start']
            df_shots_model_matrix.at[i,'Y_Start']=shot['Y_Start']
            df_shots_model_matrix.at[i,'isGoal']=shot['isGoal']
            
    df_goals_model_matrix = df_shots_model_matrix[df_shots_model_matrix['isGoal']==True]
    histo_2d_goals = np.histogram2d(df_goals_model_matrix['X_Start'],df_goals_model_matrix['Y_Start'],bins=[x_bin_size,y_bin_size], range=[[0, 100],[0, 100]])
    
    # Transpose resulting matrix to match matplot coordinate system
    return histo_2d_goals[0]

def compute_move_frequency_matrix(df_league_events,
                                  x_bin_size,
                                  y_bin_size):
    df_moves = df_league_events[df_league_events['isMove']==True]
    df_moves_model_matrix = df_moves[['X_Start','Y_Start']]
      
    # Create 2d histograms for modelling moves at x,y positions
    histo_2d_moves = np.histogram2d(df_moves_model_matrix['X_Start'],df_moves_model_matrix['Y_Start'],bins=[x_bin_size,y_bin_size], range=[[0, 100],[0, 100]])
    
    return histo_2d_moves[0]
    
def compute_transition_frequency_matrix_wrapper(df_league_events,
                                                x_bin_size,
                                                y_bin_size):
    # Initialize transition frequency matrix - freq of successfuly moving from (x,y) to (z,w)
    transition_count_wrapper = []
    
    df_moves = df_league_events[df_league_events['isMove']==True]
    
    df_moves = discretize_start_end_coord(df_moves,x_bin_size,y_bin_size)
    
    # Calculate and add transition matrix for each cell
    for x in range(x_bin_size):
        
        transition_count_row = []
        
        for y in range(y_bin_size):
            df_cell_transition_matrix = df_moves[(df_moves.X_Start_Bin == x+1) & (df_moves.Y_Start_Bin == y+1)].loc[:,['X_Start','Y_Start','X_End', 'Y_End']]
            histo_2d_cell_transition = np.histogram2d(df_cell_transition_matrix['X_End'],df_cell_transition_matrix['Y_End'],bins=[x_bin_size,y_bin_size], range=[[0, 100],[0, 100]])
            transition_count_row.append(histo_2d_cell_transition[0])
        
        transition_count_wrapper.append(transition_count_row)
    return transition_count_wrapper

def compute_accurate_transition_frequency_matrix_wrapper(df_league_events,
                                                         x_bin_size,
                                                         y_bin_size):
    # Initialize transition frequency matrix - freq of successfuly moving from (x,y) to (z,w)
    acc_transition_count_wrapper = []
    
    df_moves_accurate = df_league_events[df_league_events['isMoveAccurate']==True]
    
    df_moves_accurate = discretize_start_end_coord(df_moves_accurate,x_bin_size,y_bin_size)
    
    # Calculate and add transition matrix for each cell
    for x in range(x_bin_size):
        
        acc_transition_count_row = []
        
        for y in range(y_bin_size):
            df_cell_acc_transition_matrix = df_moves_accurate[(df_moves_accurate.X_Start_Bin == x+1) & (df_moves_accurate.Y_Start_Bin == y+1)].loc[:,['X_Start','Y_Start','X_End', 'Y_End']]
            histo_2d_cell_acc_transition = np.histogram2d(df_cell_acc_transition_matrix['X_End'],df_cell_acc_transition_matrix['Y_End'],bins=[x_bin_size,y_bin_size], range=[[0, 100],[0, 100]])
            acc_transition_count_row.append(histo_2d_cell_acc_transition[0])
        
        acc_transition_count_wrapper.append(acc_transition_count_row)
    return acc_transition_count_wrapper

def compute_transition_prob_matrix_wrapper(df_league_events,
                                           x_bin_size,
                                           y_bin_size):
    # Initialize transition prob matrix - prob of moving from (x,y) to (z,w)
    transition_prob_wrapper = []
    
    transition_count_wrapper = compute_transition_frequency_matrix_wrapper(df_league_events,
                                                                           x_bin_size,
                                                                           y_bin_size)
    acc_transition_count_wrapper = compute_accurate_transition_frequency_matrix_wrapper(df_league_events,
                                                                                        x_bin_size,
                                                                                        y_bin_size)
    
    for x in range(x_bin_size):
        transition_prob_row = []
            
        for y in range(y_bin_size):
            transition_prob_row.append(acc_transition_count_wrapper[x][y] / sum(sum(transition_count_wrapper[x][y])))
            
        transition_prob_wrapper.append(transition_prob_row)
    return transition_prob_wrapper

def compute_xt_cell(current_x, 
                    current_y, 
                    previous_xt_matrix, 
                    shot_prob_matrix, 
                    goal_prob_matrix, 
                    move_prob_matrix,
                    transition_prob_wrapper,
                    x_bin_size,
                    y_bin_size):
    xt_current_cell = 0
    xg_cell = shot_prob_matrix[current_x][current_y] * goal_prob_matrix[current_x][current_y]
    x_move_cell = move_prob_matrix[current_x][current_y]
    xt_given_move_cell = 0
    
    for dest_x in range(x_bin_size):
        for dest_y in range(y_bin_size):
            xt_given_move_cell = xt_given_move_cell + transition_prob_wrapper[current_x][current_y][dest_x][dest_y] * previous_xt_matrix[dest_x][dest_y]
    
    xt_current_cell = xg_cell + x_move_cell * xt_given_move_cell
    
    return xt_current_cell

def compute_xt_matrix(n,
                      x_bin_size,
                      y_bin_size,
                      shot_prob_matrix, 
                      goal_prob_matrix, 
                      move_prob_matrix, 
                      transition_prob_matrix_wrapper):
    # n is the number of iterations
    if (n==0):
        return np.zeros((20,20))
    
    if (n==1):
        return (shot_prob_matrix * goal_prob_matrix)
        
    xt_matrices = []
    xt_matrices.append(compute_xt_matrix(0,
                                         x_bin_size,
                                         y_bin_size,
                                         shot_prob_matrix,
                                         goal_prob_matrix,
                                         move_prob_matrix,
                                         transition_prob_matrix_wrapper))
    xt_matrices.append(compute_xt_matrix(1,
                                         x_bin_size,
                                         y_bin_size,
                                         shot_prob_matrix, 
                                         goal_prob_matrix,
                                         move_prob_matrix,
                                         transition_prob_matrix_wrapper))
    
    for i in range(1,n):
        xt_matrix = []
        for current_x in range(x_bin_size):
            xt_row = []
            for current_y in range(y_bin_size):
                xt_cell = compute_xt_cell(current_x, 
                                          current_y, 
                                          xt_matrices[i], 
                                          shot_prob_matrix, 
                                          goal_prob_matrix, 
                                          move_prob_matrix, 
                                          transition_prob_matrix_wrapper,
                                          x_bin_size,
                                          y_bin_size)
                xt_row.append(xt_cell)
            xt_matrix.append(xt_row)
        xt_matrices.append(xt_matrix)
    return np.array(xt_matrices[n])
    
def discretize_start_end_coord(df_events,x_bin_size,y_bin_size):
    x_bins = np.arange(0,100,100/x_bin_size)
    y_bins = np.arange(0,100,100/y_bin_size)
    
    x_start_bins = np.digitize(df_events['X_Start'], x_bins)
    y_start_bins = np.digitize(df_events['Y_Start'], y_bins)
    x_end_bins = np.digitize(df_events['X_End'], x_bins)
    y_end_bins = np.digitize(df_events['Y_End'], y_bins)
    
    df_events['X_Start_Bin'] = x_start_bins
    df_events['Y_Start_Bin'] = y_start_bins
    df_events['X_End_Bin'] = x_end_bins
    df_events['Y_End_Bin'] = y_end_bins
    
    return df_events

def compute_player_event_xt_score_df(df_league_events, 
                                     xt_matrix, 
                                     player_id,
                                     x_bin_size,
                                     y_bin_size):
    #Inlucde all moves
    #df_player_events = df_league_events[((df_league_events['playerId']==player_id) & (df_league_events['isMove']==True))]
    #Include only successful moves
    df_player_events = df_league_events[((df_league_events['playerId']==player_id) & (df_league_events['isMoveAccurate']==True))]
    #Includes goals
    #df_player_events = df_league_events[((df_league_events['playerId']==player_id) & ((df_league_events['isMove']==True) | (df_league_events['isGoal']==True)))]
    df_player_events = discretize_start_end_coord(df_player_events,x_bin_size,y_bin_size)
    df_player_events_model = df_player_events[['X_Start_Bin','Y_Start_Bin','X_End_Bin','Y_End_Bin','isMove','isMoveAccurate','isGoal']]
    
    xt_start_list = []
    xt_end_list = []
    xt_diff_list = []
    xt_score_list = []
    
    for i, row in df_player_events_model.iterrows():
        x_start_bin = row['X_Start_Bin']
        y_start_bin = row['Y_Start_Bin']
        xt_start = xt_matrix[x_start_bin-1][y_start_bin-1]
        xt_start_list.append(xt_start)
        
        
        x_end_bin = row['X_End_Bin']
        y_end_bin = row['Y_End_Bin']
        xt_end = xt_matrix[x_end_bin-1][y_end_bin-1]
        xt_end_list.append(xt_end)
        
        xt_diff = xt_end - xt_start
        xt_diff_list.append(xt_diff)
        xt_score = xt_diff
        
        """
        # Assigns special xT for goals
        if (row['isGoal']):
            xt_score = 1.0
        """
        xt_score_list.append(xt_score)
    
    df_player_events_model['xt_start'] = xt_start_list
    df_player_events_model['xt_end'] = xt_end_list
    df_player_events_model['xt_diff'] = xt_diff_list
    df_player_events_model['xt_score'] = xt_score_list

    return df_player_events_model

def compute_player_accumulated_xt(df_player_events_xt):
    return sum(df_player_events_xt['xt_score'])

def compute_player_nr_of_goals(df_league_events, player_id):
    df_player_goals = df_league_events[((df_league_events['playerId']==player_id) & (df_league_events['isGoal']==True))]
    return len(df_player_goals)

def load_players_data():
    with open('../wyscoutdata/players.json') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def get_player_data(df_players, player_id):
    return df_players[df_players.wyId == player_id]

def compute_player_xg_given_shot_df(df_league_events, 
                                    goal_prob_matrix, 
                                    player_id,
                                    x_bin_size,
                                    y_bin_size):
    df_shots = create_sub_event_df(df_league_events, 'Shot')
    df_player_shots = df_shots[(df_shots['playerId']==player_id)]
    df_player_shots = discretize_start_end_coord(df_player_shots,x_bin_size,y_bin_size)
    df_player_shots_model = df_player_shots[['X_Start_Bin','Y_Start_Bin','isGoal']]
    
    goal_prob_list = []
    
    for i, row in df_player_shots_model.iterrows():
        
        goal_prob = goal_prob_matrix[row['X_Start_Bin']-1][row['Y_Start_Bin']-1]
        goal_prob_list.append(goal_prob)
    
    df_player_shots_model['xg_given_shot'] = goal_prob_list
    
    return df_player_shots_model

def compute_player_accumulated_xg(df_player_shots):
    return sum(df_player_shots['xg_given_shot'])

def compute_team_event_xt_score_df(df_league_events, 
                                   xt_matrix, 
                                   team_id,
                                   x_bin_size,
                                   y_bin_size):
    
    df_team_events = df_league_events[((df_league_events['teamId']==team_id) & (df_league_events['isMove']==True))]
    #Include only successful moves
    df_team_events = df_league_events[((df_league_events['teamId']==team_id) & (df_league_events['isMoveAccurate']==True))]
    #Includes goals
    df_team_events = df_league_events[((df_league_events['teamId']==team_id) & ((df_league_events['isMove']==True) | (df_league_events['isGoal']==True)))]
    df_team_events = discretize_start_end_coord(df_team_events,x_bin_size,y_bin_size)
    df_team_events_model = df_team_events[['X_Start_Bin','Y_Start_Bin','X_End_Bin','Y_End_Bin','isMove','isGoal']]
    
    xt_start_list = []
    xt_end_list = []
    xt_diff_list = []
    xt_score_list = []
    
    for i, row in df_team_events_model.iterrows():
        x_start_bin = row['X_Start_Bin']
        y_start_bin = row['Y_Start_Bin']
        xt_start = xt_matrix[x_start_bin-1][y_start_bin-1]
        xt_start_list.append(xt_start)
        
        
        x_end_bin = row['X_End_Bin']
        y_end_bin = row['Y_End_Bin']
        xt_end = xt_matrix[x_end_bin-1][y_end_bin-1]
        xt_end_list.append(xt_end)
        
        xt_diff = xt_end - xt_start
        xt_diff_list.append(xt_diff)
        xt_score = xt_diff
        
        """
        # Assigns special xT for goals
        if (row['isGoal']):
            xt_score = 1.0
        """
        xt_score_list.append(xt_score)
        
    
    df_team_events_model['xt_start'] = xt_start_list
    df_team_events_model['xt_end'] = xt_end_list
    df_team_events_model['xt_diff'] = xt_diff_list
    df_team_events_model['xt_score'] = xt_score_list

    return df_team_events_model

def compute_team_accumulated_xt(df_team_league_events_xt):
    return sum(df_team_league_events_xt['xt_score'])
    
    
def load_teams_data():
    with open('../wyscoutdata/teams.json') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def get_team_data(df_teams, team_id):
    return df_teams[df_teams.wyId == team_id]

def compute_team_nr_of_goals(df_league_events, team_id):
    df_team_goals = df_league_events[((df_league_events['teamId']==team_id) & (df_league_events['isGoal']==True))]
    return len(df_team_goals)

def compute_team_xg_given_shot_df(df_league_events, 
                                  goal_prob_matrix, 
                                  team_id,
                                  x_bin_size,
                                  y_bin_size):
    df_shots = create_sub_event_df(df_league_events, 'Shot')
    df_team_shots = df_shots[(df_shots['teamId']==team_id)]
    df_team_shots = discretize_start_end_coord(df_team_shots,x_bin_size,y_bin_size)
    df_team_shots_model = df_team_shots[['X_Start_Bin','Y_Start_Bin','isGoal']]
    
    goal_prob_list = []
    
    for i, row in df_team_shots_model.iterrows():
        
        goal_prob = goal_prob_matrix[row['X_Start_Bin']-1][row['Y_Start_Bin']-1]
        goal_prob_list.append(goal_prob)
    
    df_team_shots_model['xg_given_shot'] = goal_prob_list
    
    return df_team_shots_model

def compute_team_accumulated_xg(df_team_shots_model):
    return sum(df_team_shots_model['xg_given_shot'])