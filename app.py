import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid

# Function to load and process IPL data from deliveries.csv
@st.cache
def load_data(file_path):
    ipl_data = pd.read_csv(file_path)
    return ipl_data


def calculate_batsman_stats(ipl_data):
    # Calculate batsman statistics
    # Total runs scored by each batsman
    total_runs = ipl_data.groupby('batter')['batsman_runs'].sum().reset_index()

    # Number of sixes
    sixes = ipl_data[ipl_data['batsman_runs'] == 6].groupby('batter').size().reset_index(name='sixes')

    # Number of fours
    fours = ipl_data[ipl_data['batsman_runs'] == 4].groupby('batter').size().reset_index(name='fours')

    # Scores in each match
    scores_per_match = ipl_data.groupby(['batter', 'match_id'])['batsman_runs'].sum().reset_index()

    # Number of matches played by each batsman
    matches_played = scores_per_match.groupby('batter')['match_id'].nunique().reset_index(name='matches_played')

    # Number of half-centuries
    half_centuries = scores_per_match[
        (scores_per_match['batsman_runs'] >= 50) & (scores_per_match['batsman_runs'] < 100)].groupby(
        'batter').size().reset_index(name='half_centuries')

    # Number of centuries
    centuries = scores_per_match[scores_per_match['batsman_runs'] >= 100].groupby('batter').size().reset_index(
        name='centuries')

    # Batting average
    innings = ipl_data[ipl_data['is_wicket'] == 1].groupby('batter').size().reset_index(name='outs')
    batting_average = total_runs.merge(innings, on='batter', how='left').fillna(0)
    batting_average['average'] = batting_average['batsman_runs'] / batting_average['outs']
    batting_average['average'].replace(float('inf'), np.nan, inplace=True)

    # Strike rate
    total_balls = ipl_data.groupby('batter').size().reset_index(name='balls_faced')
    strike_rate = total_runs.merge(total_balls, on='batter')
    strike_rate['strike_rate'] = (strike_rate['batsman_runs'] / strike_rate['balls_faced']) * 100

    # Highest score
    highest_score = scores_per_match.groupby('batter')['batsman_runs'].max().reset_index(name='highest_score')

    # Merging all statistics into a single DataFrame
    batsman_stats = total_runs \
        .merge(sixes, on='batter', how='left').fillna(0) \
        .merge(fours, on='batter', how='left').fillna(0) \
        .merge(matches_played, on='batter', how='left') \
        .merge(half_centuries, on='batter', how='left').fillna(0) \
        .merge(centuries, on='batter', how='left').fillna(0) \
        .merge(batting_average[['batter', 'average']], on='batter', how='left') \
        .merge(strike_rate[['batter', 'strike_rate']], on='batter', how='left') \
        .merge(highest_score, on='batter', how='left')

    return batsman_stats


def calculate_bowler_stats(ipl_data):
    # Calculate bowler statistics
    # Runs conceded by each bowler
    runs_given = ipl_data.groupby('bowler')['batsman_runs'].sum().reset_index()
    runs_given.rename(columns={'batsman_runs': 'runs_given'}, inplace=True)

    # Number of balls bowled by each bowler
    balls_bowled = ipl_data.groupby('bowler')['ball'].count().reset_index(name='balls_bowled')

    # Wickets taken by each bowler
    wickets_taken = ipl_data[ipl_data['is_wicket'] == 1].groupby('bowler')['is_wicket'].count().reset_index(
        name='wickets_taken')

    # Economy rate of each bowler
    economy_rate = runs_given.merge(balls_bowled, on='bowler')
    economy_rate['economy_rate'] = economy_rate['runs_given'] / (economy_rate['balls_bowled'] // 6)

    # 5-wicket hauls
    five_wicket_hauls = ipl_data[ipl_data['is_wicket'] == 1].groupby('bowler')['match_id'].apply(
        lambda x: (x.value_counts() >= 5).sum()).reset_index(name='5_wicket_hauls')

    # Merging all bowling statistics into a single DataFrame
    bowler_stats = runs_given.merge(balls_bowled, on='bowler').merge(wickets_taken, on='bowler').merge(
        economy_rate[['bowler', 'economy_rate']], on='bowler').merge(five_wicket_hauls, on='bowler', how='left')

    return bowler_stats


def batsman_with_most_runs(ipl_data):
    # Calculate total runs scored by each batsman
    total_runs = ipl_data.groupby('batter')['batsman_runs'].sum().reset_index()

    # Sort batsmen by their total runs in ascending order
    total_runs_sorted = total_runs.sort_values(by='batsman_runs', ascending=False)

    # Return the sorted list of batsmen and their total runs
    batsmen_sorted = total_runs_sorted[['batter', 'batsman_runs']].values.tolist()

    return batsmen_sorted

def calculate_bowler_vs_batsman_stats(ipl_data, selected_batsman, selected_bowler):
    encounter_data = ipl_data[(ipl_data['batter'] == selected_batsman) & (ipl_data['bowler'] == selected_bowler)]

    # Calculate statistics
    total_runs = encounter_data['batsman_runs'].sum()
    balls_faced = len(encounter_data)
    dismissals = encounter_data['is_wicket'].sum()
    strike_rate = (total_runs / balls_faced) * 100 if balls_faced > 0 else 0
    sixes = encounter_data[encounter_data['batsman_runs'] == 6].shape[0]
    fours = encounter_data[encounter_data['batsman_runs'] == 4].shape[0]

    stats = {
        'Statistic': ['Total Runs', 'Balls Faced', 'Dismissals', 'Strike Rate', 'Number of Sixes', 'Number of Fours'],
        'Value': [total_runs, balls_faced, dismissals, strike_rate, sixes, fours]
    }

    stats_df = pd.DataFrame(stats)

    return stats_df


def bowler_with_most_wickets(ipl_data):
    # Calculate total wickets taken by each bowler
    total_wickets = ipl_data[ipl_data['is_wicket'] == 1].groupby('bowler').size().reset_index(name='wickets_taken')

    # Sort bowlers by their wickets in descending order
    total_wickets_sorted = total_wickets.sort_values(by='wickets_taken', ascending=False).reset_index(drop=True)

    return total_wickets_sorted


def batsman_with_most_sixes(ipl_data):
    # Calculate total sixes hit by each batsman
    total_sixes = ipl_data[ipl_data['batsman_runs'] == 6].groupby('batter').size().reset_index(name='sixes')

    # Sort batsmen by their sixes in descending order
    total_sixes_sorted = total_sixes.sort_values(by='sixes', ascending=False).reset_index(drop=True)

    return total_sixes_sorted


def batsman_with_most_sixes(ipl_data):
    # Calculate total sixes hit by each batsman
    total_sixes = ipl_data[ipl_data['batsman_runs'] == 6].groupby('batter').size().reset_index(name='sixes')

    # Sort batsmen by their sixes in descending order
    total_sixes_sorted = total_sixes.sort_values(by='sixes', ascending=False).reset_index(drop=True)

    return total_sixes_sorted
def batsman_with_most_fours(ipl_data):
    # Calculate total fours hit by each batsman
    total_fours = ipl_data[ipl_data['batsman_runs'] == 4].groupby('batter').size().reset_index(name='fours')

    # Sort batsmen by their fours in descending order
    total_fours_sorted = total_fours.sort_values(by='fours', ascending=False).reset_index(drop=True)

    return total_fours_sorted

# Define the calculate_head_to_head_stats function
import pandas as pd

def head_to_head_match_details(ipl_data, team1, team2):
    # Filter IPL data for head-to-head matches between the selected teams
    head_to_head_matches = ipl_data[((ipl_data['batting_team'] == team1) & (ipl_data['bowling_team'] == team2)) |
                                    ((ipl_data['batting_team'] == team2) & (ipl_data['bowling_team'] == team1))]

    if not head_to_head_matches.empty:
        # Initialize an empty list to store match details
        match_details = []

        # Iterate through each match
        for index, match in head_to_head_matches.iterrows():
            # Extract match details
            match_id = match['match_id']
            batting_team = match['batting_team']
            bowling_team = match['bowling_team']
            total_runs_team1 = head_to_head_matches[(head_to_head_matches['match_id'] == match_id) & (head_to_head_matches['batting_team'] == team1)]['total_runs'].sum()
            total_runs_team2 = head_to_head_matches[(head_to_head_matches['match_id'] == match_id) & (head_to_head_matches['batting_team'] == team2)]['total_runs'].sum()

            # Determine the winner of the match
            winner = team1 if total_runs_team1 > total_runs_team2 else (team2 if total_runs_team2 > total_runs_team1 else "Draw")

            # Store match details in a dictionary
            match_detail = {
                "Match ID": match_id,
                "Batting Team": batting_team,
                "Bowling Team": bowling_team,
                f"{team1} Runs": total_runs_team1,
                f"{team2} Runs": total_runs_team2,
                "Winner": winner
            }

            # Append match details to the list
            match_details.append(match_detail)

        # Calculate total matches played
        total_matches_played = len(match_details)

        # Create DataFrame with match details
        match_details_df = pd.DataFrame(match_details)

        return match_details_df, total_matches_played
    else:
        return None, 0

def main():
    st.set_page_config(layout="wide")
    st.title("IPL Players and Teams Performance (2009 - 2021)")

    # Specify the path to your CSV file
    file_path = "deliveries.csv"

    ipl_data = load_data(file_path)
    batsman_stats = calculate_batsman_stats(ipl_data)
    bowler_stats = calculate_bowler_stats(ipl_data)
    most_wickets_bowler = bowler_with_most_wickets(ipl_data)
    most_sixes_batsman = batsman_with_most_sixes(ipl_data)
    most_fours_batsman = batsman_with_most_fours(ipl_data)

    # Sidebar navigation bar
    navigation_links = {
        "Most Sixes by a Batsman": "most_sixes",
        "Most Fours by a Batsman": "most_fours",
        "Batsman Statistics": "batsman_stats",
        "Bowler Statistics": "bowler_stats",
        "Batsman vs Bowler Interaction": "batsman_vs_bowler",
        "Most Runs by a Batsman": "most_runs",
        "Most Wickets by a Bowler": "most_wickets",
        "Head to Head Matches between Teams":"Head-to-Head Matches"
    }

    st.sidebar.title("Navigation")

    selected_link = st.sidebar.radio("Go to", list(navigation_links.keys()))

    # Set the selected section based on the link clicked
    selected_section = navigation_links[selected_link]

    # Batsman statistics section
    if selected_section == "batsman_stats":
        st.header("Batsman Statistics")
        selected_batsman = st.selectbox("Select Batsman", batsman_stats['batter'].unique())
        batsman_section = batsman_stats[batsman_stats['batter'] == selected_batsman]
        AgGrid(batsman_section)

    # Bowler statistics section
    elif selected_section == "bowler_stats":
        st.header("Bowler Statistics")
        selected_bowler = st.selectbox("Select Bowler", bowler_stats['bowler'].unique())
        bowler_section = bowler_stats[bowler_stats['bowler'] == selected_bowler]
        AgGrid(bowler_section)

    # Batsman vs bowler interaction section
    elif selected_section == "batsman_vs_bowler":
        st.header("Batsman vs Bowler Interaction")
        selected_batsman_interaction = st.selectbox("Select Batsman", batsman_stats['batter'].unique(),
                                                    key="batsman_interaction")
        selected_bowler_interaction = st.selectbox("Select Bowler", bowler_stats['bowler'].unique(),
                                                   key="bowler_interaction")

        interaction_data = ipl_data[
            (ipl_data['batter'] == selected_batsman_interaction) & (ipl_data['bowler'] == selected_bowler_interaction)]
        interaction_data = interaction_data.drop(columns=['match_id'])
        interaction_data = interaction_data.round(2)
        # Display interaction data
        AgGrid(interaction_data)

        # Calculate and display detailed statistics for the encounter
        encounter_stats_df = calculate_bowler_vs_batsman_stats(ipl_data, selected_batsman_interaction,
                                                               selected_bowler_interaction)

        AgGrid(encounter_stats_df)

    # Most runs by a batsman section
    elif selected_section == "most_runs":
        st.header("Most Runs by  Batsmen")
        st.write("Batsmen sorted by their runs:")

        # Create a DataFrame for batsmen and their runs
        batsmen_runs_df = batsman_stats[['batter', 'batsman_runs']].sort_values(by='batsman_runs',
                                                                                ascending=False).reset_index(drop=True)



        # Display list of batsmen with their total runs in two columns and enable scrolling
        AgGrid(batsmen_runs_df)

    # Most wickets by a bowler section
    elif selected_section == "most_wickets":
        st.header("Most Wickets by  Bowlers")
        st.write("Bowlers sorted by their wickets:")
        AgGrid(most_wickets_bowler)
        # Apply CSS styling for scrolling within the table
        # Most sixes by a batsman section
    elif selected_section == "most_sixes":
        st.header("Most Sixes by  Batsmen")
        st.write("Batsmen sorted by their sixes:")


        # Display list of batsmen with their total sixes in two columns and enable scrolling
        AgGrid(most_sixes_batsman)

    # Most fours by a batsman section
    elif selected_section == "most_fours":
        st.header("Most Fours by  Batsmen")
        st.write("Batsmen sorted by their fours:")
        AgGrid(most_fours_batsman)
    if selected_section == "Head-to-Head Matches":
        st.header("Head-to-Head Matches (Wait for 10s only)" )

        # Add a dropdown to select two teams for head-to-head analysis
        selected_team_1 = st.selectbox("Select Team 1", ipl_data['batting_team'].unique())
        selected_team_2 = st.selectbox("Select Team 2", ipl_data['batting_team'].unique())

        # Calculate head-to-head match details
        match_details_df, total_matches_played = head_to_head_match_details(ipl_data, selected_team_1, selected_team_2)

        # Display head-to-head match details if available
        if total_matches_played > 0:
            match_details_df = match_details_df.drop_duplicates()
            match_details_df = match_details_df.drop_duplicates(subset='Match ID')
            match_details_df = match_details_df.drop(columns=['Match ID'])
            st.write(f"Total matches played between {selected_team_1} and {selected_team_2}: {len(match_details_df)}")
            team1_wins = match_details_df[match_details_df['Winner'] == selected_team_1].shape[0]
            team2_wins = match_details_df[match_details_df['Winner'] == selected_team_2].shape[0]
            st.write(f"{selected_team_1} wins: {team1_wins}")
            st.write(f"{selected_team_2} wins: {team2_wins}")
            st.write("Details of each match:")
            AgGrid(match_details_df)
        else:
            st.write("No head-to-head matches found between the selected teams.")


if __name__ == "__main__":
    main()
