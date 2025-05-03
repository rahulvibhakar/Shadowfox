import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import json
from datetime import datetime

# Data file paths
DATA_DIR = "cricket_data"
FIELDING_DATA_FILE = os.path.join(DATA_DIR, "fielding_data.csv")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Function to load fielding data
def load_fielding_data():
    if os.path.exists(FIELDING_DATA_FILE):
        try:
            return pd.read_csv(FIELDING_DATA_FILE)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame(columns=[
                'Match No.', 'Innings', 'Team', 'Player Name', 'Ballcount', 'Position', 
                'Short Description', 'Pick', 'Throw', 'Runs', 'Overcount', 'Venue'
            ])
    else:
        return pd.DataFrame(columns=[
            'Match No.', 'Innings', 'Team', 'Player Name', 'Ballcount', 'Position', 
            'Short Description', 'Pick', 'Throw', 'Runs', 'Overcount', 'Venue'
        ])

# Function to save fielding data
def save_fielding_data(data):
    try:
        data.to_csv(FIELDING_DATA_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False

# Function to load weights
def load_weights():
    default_weights = {
        'WCP': 1.0,  # Clean Pick
        'WGT': 1.0,  # Good Throw
        'WC': 3.0,   # Catch
        'WDC': -2.0, # Dropped Catch
        'WST': 3.0,  # Stumping
        'WRO': 3.0,  # Run Out
        'WMRO': -2.0,# Missed Run Out
        'WDH': 2.0   # Direct Hit
    }
    
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading weights: {e}")
            return default_weights
    else:
        return default_weights

# Function to save weights
def save_weights(weights):
    try:
        with open(WEIGHTS_FILE, 'w') as f:
            json.dump(weights, f)
        return True
    except Exception as e:
        st.error(f"Error saving weights: {e}")
        return False

# Set page configuration
st.set_page_config(
    page_title="Cricket Fielding Analysis",
    page_icon="🏏",
    layout="wide"
)

# Application title and description
st.title("🏏 Cricket Fielding Analysis Tool")
st.markdown("""
This application helps analyze cricket fielding performance for T20 matches. 
Track individual fielding contributions and calculate performance metrics to improve team strategy.
""")

# Initialize session state variables
if 'fielding_data' not in st.session_state:
    st.session_state.fielding_data = load_fielding_data()

if 'weights' not in st.session_state:
    st.session_state.weights = load_weights()

# Auto-backup feature
if 'last_backup' not in st.session_state:
    st.session_state.last_backup = datetime.now()

# Function to calculate performance score
def calculate_performance_score(data, weights):
    if data.empty:
        return pd.DataFrame()
    
    # Group by player
    player_stats = {}
    for player in data['Player Name'].unique():
        player_data = data[data['Player Name'] == player]
        
        # Calculate components
        cp = sum(player_data['Pick'] == 'Clean Pick')
        gt = sum(player_data['Pick'] == 'Good Throw')
        c = sum(player_data['Pick'] == 'Catch')
        dc = sum(player_data['Pick'] == 'Drop Catch')
        st = sum(player_data['Throw'] == 'Stumping')
        ro = sum(player_data['Throw'] == 'Run Out')
        mro = sum(player_data['Throw'] == 'Missed Run Out')
        dh = sum(player_data['Throw'] == 'Direct Hit')
        rs = player_data['Runs'].sum()
        
        # Calculate performance score
        ps = (cp * weights['WCP']) + (gt * weights['WGT']) + (c * weights['WC']) + \
             (dc * weights['WDC']) + (st * weights['WST']) + (ro * weights['WRO']) + \
             (mro * weights['WMRO']) + (dh * weights['WDH']) + rs
        
        player_stats[player] = {
            'Clean Picks': cp,
            'Good Throws': gt,
            'Catches': c,
            'Dropped Catches': dc,
            'Stumpings': st,
            'Run Outs': ro,
            'Missed Run Outs': mro,
            'Direct Hits': dh,
            'Runs Saved': rs,
            'Performance Score': ps
        }
    
    return pd.DataFrame.from_dict(player_stats, orient='index')

# Function to generate download link for dataframe
def get_download_link(df, filename, text):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Collection", "Performance Analysis", "Visualization", "Settings"])

if page == "Data Collection":
    st.header("Fielding Data Collection")
    
    # Form for entering new fielding data
    st.subheader("Add New Fielding Entry")
    with st.form(key="fielding_data_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            match_no = st.text_input("Match No.")
            innings = st.selectbox("Innings", [1, 2])
            team = st.text_input("Team")
            player_name = st.text_input("Player Name")
        
        with col2:
            ballcount = st.number_input("Ball Count (in over)", min_value=1, max_value=6, step=1)
            position = st.selectbox("Position", [
                "Long On", "Long Off", "Deep Midwicket", "Deep Square Leg", 
                "Deep Fine Leg", "Third Man", "Deep Point", "Deep Cover", 
                "Cover", "Point", "Gully", "Slips", "Short Leg", "Silly Point",
                "Mid Off", "Mid On", "Mid Wicket", "Square Leg", "Fine Leg",
                "Wicket Keeper", "Bowler"
            ])
            overcount = st.number_input("Over Number", min_value=1, max_value=20, step=1)
            venue = st.text_input("Venue")
        
        with col3:
            short_desc = st.text_area("Short Description", height=60)
            pick = st.selectbox("Pick", [
                "Clean Pick", "Good Throw", "Fumble", "Bad Throw", "Catch", "Drop Catch", "N/A"
            ])
            throw = st.selectbox("Throw", [
                "Run Out", "Missed Run Out", "Stumping", "Missed Stumping", "Direct Hit", "N/A"
            ])
            runs = st.number_input("Runs Saved (+) or Conceded (-)", min_value=-6, max_value=6, step=1)
        
        submit_button = st.form_submit_button(label="Add Entry")
    
    if submit_button:
        # Add new entry to dataframe
        new_entry = pd.DataFrame({
            'Match No.': [match_no],
            'Innings': [innings],
            'Team': [team],
            'Player Name': [player_name],
            'Ballcount': [ballcount],
            'Position': [position],
            'Short Description': [short_desc],
            'Pick': [pick],
            'Throw': [throw],
            'Runs': [runs],
            'Overcount': [overcount],
            'Venue': [venue]
        })
        
        st.session_state.fielding_data = pd.concat([st.session_state.fielding_data, new_entry], ignore_index=True)
        save_fielding_data(st.session_state.fielding_data)
        st.success("Entry added successfully!")
    
    # Display current data with edit capability
    st.subheader("Current Data")
    
    if not st.session_state.fielding_data.empty:
        edited_data = st.data_editor(
            st.session_state.fielding_data,
            use_container_width=True,
            num_rows="dynamic"
        )
        if st.button("Save Edits"):
            st.session_state.fielding_data = edited_data
            save_fielding_data(st.session_state.fielding_data)
            st.success("Edits saved successfully!")
        
        # Download button for the data
        st.markdown(
            get_download_link(st.session_state.fielding_data, 'fielding_data.csv', 'Download Data as CSV'),
            unsafe_allow_html=True
        )
        
        # Clear data button
        if st.button("Clear All Data"):
            st.session_state.fielding_data = pd.DataFrame(columns=[
                'Match No.', 'Innings', 'Team', 'Player Name', 'Ballcount', 'Position', 
                'Short Description', 'Pick', 'Throw', 'Runs', 'Overcount', 'Venue'
            ])
            save_fielding_data(st.session_state.fielding_data)
            st.success("All data cleared!")
    else:
        st.info("No data collected yet. Add entries using the form above.")
    
    # Upload existing data
    st.subheader("Upload Existing Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
            if st.button("Append Uploaded Data"):
                st.session_state.fielding_data = pd.concat([st.session_state.fielding_data, uploaded_data], ignore_index=True)
                save_fielding_data(st.session_state.fielding_data)
                st.success("Uploaded data appended successfully!")
            if st.button("Replace Existing Data"):
                st.session_state.fielding_data = uploaded_data
                save_fielding_data(st.session_state.fielding_data)
                st.success("Data replaced successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")

elif page == "Performance Analysis":
    st.header("Performance Analysis")
    
    if st.session_state.fielding_data.empty:
        st.warning("No data available for analysis. Please add data in the Data Collection page.")
    else:
        # Filter options
        st.subheader("Filter Data")
        col1, col2 = st.columns(2)
        
        with col1:
            selected_match = st.selectbox(
                "Select Match",
                ["All"] + list(st.session_state.fielding_data['Match No.'].unique())
            )
            
            selected_team = st.selectbox(
                "Select Team",
                ["All"] + list(st.session_state.fielding_data['Team'].unique())
            )
        
        with col2:
            selected_innings = st.selectbox(
                "Select Innings",
                ["All"] + list(map(str, st.session_state.fielding_data['Innings'].unique()))
            )
            
            selected_venue = st.selectbox(
                "Select Venue",
                ["All"] + list(st.session_state.fielding_data['Venue'].unique())
            )
        
        # Apply filters
        filtered_data = st.session_state.fielding_data.copy()
        
        if selected_match != "All":
            filtered_data = filtered_data[filtered_data['Match No.'] == selected_match]
        
        if selected_team != "All":
            filtered_data = filtered_data[filtered_data['Team'] == selected_team]
        
        if selected_innings != "All":
            filtered_data = filtered_data[filtered_data['Innings'] == int(selected_innings)]
        
        if selected_venue != "All":
            filtered_data = filtered_data[filtered_data['Venue'] == selected_venue]
        
        # Display filtered data
        st.subheader("Filtered Data")
        st.dataframe(filtered_data, use_container_width=True)
        
        # Calculate performance metrics
        st.subheader("Performance Metrics")
        
        performance_scores = calculate_performance_score(filtered_data, st.session_state.weights)
        
        if not performance_scores.empty:
            st.dataframe(performance_scores, use_container_width=True)
            
            # Download performance data
            st.markdown(
                get_download_link(performance_scores, 'performance_scores.csv', 'Download Performance Metrics as CSV'),
                unsafe_allow_html=True
            )
            
            # Top performers
            st.subheader("Top Performers")
            top_performers = performance_scores.sort_values('Performance Score', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Overall Performance Score")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(top_performers.index, top_performers['Performance Score'], color='royalblue')
                ax.set_xlabel('Player')
                ax.set_ylabel('Performance Score')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.write("Contribution Breakdown")
                # Show contribution of each component to total score
                components = ['Clean Picks', 'Good Throws', 'Catches', 'Dropped Catches', 
                             'Stumpings', 'Run Outs', 'Missed Run Outs', 'Direct Hits', 'Runs Saved']
                
                # Select player for detailed breakdown
                selected_player = st.selectbox("Select Player for Detailed Breakdown", top_performers.index)
                
                player_data = top_performers.loc[selected_player][components]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(components, player_data, color=sns.color_palette("viridis", len(components)))
                ax.set_xlabel('Metric')
                ax.set_ylabel('Count/Value')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("No performance data available with current filters.")

elif page == "Visualization":
    st.header("Data Visualization")
    
    if st.session_state.fielding_data.empty:
        st.warning("No data available for visualization. Please add data in the Data Collection page.")
    else:
        # Select visualization type
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Player Performance Comparison", "Position Analysis", "Over-by-Over Analysis", "Heatmap"]
        )
        
        if viz_type == "Player Performance Comparison":
            # Select players to compare
            players = st.multiselect(
                "Select Players to Compare",
                st.session_state.fielding_data['Player Name'].unique(),
                default=list(st.session_state.fielding_data['Player Name'].unique())[:3] if len(st.session_state.fielding_data['Player Name'].unique()) >= 3 else list(st.session_state.fielding_data['Player Name'].unique())
            )
            
            if not players:
                st.warning("Please select at least one player for comparison.")
            else:
                # Filter data for selected players
                player_data = st.session_state.fielding_data[st.session_state.fielding_data['Player Name'].isin(players)]
                
                # Calculate performance scores
                performance = calculate_performance_score(player_data, st.session_state.weights)
                
                # Create radar chart for comparison
                if not performance.empty:
                    # Select metrics for radar chart
                    metrics = st.multiselect(
                        "Select Metrics for Comparison",
                        ['Clean Picks', 'Good Throws', 'Catches', 'Dropped Catches', 
                         'Stumpings', 'Run Outs', 'Missed Run Outs', 'Direct Hits', 'Runs Saved'],
                        default=['Clean Picks', 'Catches', 'Run Outs', 'Runs Saved']
                    )
                    
                    if metrics:
                        # Create radar chart
                        st.subheader("Player Performance Comparison")
                        
                        # Normalize the data for radar chart
                        radar_data = performance[metrics].copy()
                        
                        # Avoid division by zero
                        for col in radar_data.columns:
                            if radar_data[col].max() != 0:
                                radar_data[col] = radar_data[col] / radar_data[col].max()
                        
                        # Number of variables
                        categories = metrics
                        N = len(categories)
                        
                        # Create angle for each category
                        angles = [n / float(N) * 2 * np.pi for n in range(N)]
                        angles += angles[:1]  # Close the loop
                        
                        # Create figure
                        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
                        
                        # Add player by player
                        for i, player in enumerate(radar_data.index):
                            values = radar_data.loc[player].values.flatten().tolist()
                            values += values[:1]  # Close the loop
                            
                            ax.plot(angles, values, linewidth=2, linestyle='solid', label=player)
                            ax.fill(angles, values, alpha=0.1)
                        
                        # Fix axis
                        ax.set_xticks(angles[:-1])
                        ax.set_xticklabels(categories)
                        
                        # Add legend
                        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                        
                        st.pyplot(fig)
                        
                        # Bar chart comparison
                        st.subheader("Performance Score Comparison")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(performance.index, performance['Performance Score'], color=sns.color_palette("viridis", len(performance)))
                        ax.set_xlabel('Player')
                        ax.set_ylabel('Performance Score')
                        ax.tick_params(axis='x', rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Please select at least one metric for comparison.")
                else:
                    st.warning("No performance data available for selected players.")
        
        elif viz_type == "Position Analysis":
            st.subheader("Fielding Position Analysis")
            
            # Count occurrences of each position
            position_counts = st.session_state.fielding_data['Position'].value_counts()
            
            # Plot position distribution
            fig, ax = plt.subplots(figsize=(12, 6))
            position_counts.plot(kind='bar', ax=ax, color=sns.color_palette("viridis", len(position_counts)))
            ax.set_xlabel('Fielding Position')
            ax.set_ylabel('Number of Actions')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Analyze performance by position
            st.subheader("Performance by Position")
            
            # Group by position and calculate average runs saved
            position_performance = st.session_state.fielding_data.groupby('Position')['Runs'].agg(['mean', 'sum', 'count'])
            position_performance.columns = ['Average Runs Saved', 'Total Runs Saved', 'Number of Actions']
            
            st.dataframe(position_performance.sort_values('Total Runs Saved', ascending=False), use_container_width=True)
            
            # Plot average runs saved by position
            fig, ax = plt.subplots(figsize=(12, 6))
            position_performance['Average Runs Saved'].sort_values(ascending=False).plot(kind='bar', ax=ax, color=sns.color_palette("viridis", len(position_performance)))
            ax.set_xlabel('Fielding Position')
            ax.set_ylabel('Average Runs Saved')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        elif viz_type == "Over-by-Over Analysis":
            st.subheader("Over-by-Over Fielding Analysis")
            
            # Group by over number
            over_analysis = st.session_state.fielding_data.groupby('Overcount').agg({
                'Runs': ['sum', 'mean', 'count'],
                'Pick': lambda x: sum(x == 'Catch'),
                'Throw': lambda x: sum(x == 'Run Out')
            })
            
            over_analysis.columns = ['Total Runs Saved', 'Average Runs Saved', 'Actions', 'Catches', 'Run Outs']
            
            # Show the data
            st.dataframe(over_analysis, use_container_width=True)
            
            # Plot over-by-over analysis
            st.subheader("Fielding Actions by Over")
            
            fig, ax = plt.subplots(figsize=(14, 7))
            over_analysis['Actions'].plot(kind='line', marker='o', ax=ax, color='royalblue', label='Total Actions')
            ax.set_xlabel('Over Number')
            ax.set_ylabel('Number of Actions')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Twin axis for runs saved
            ax2 = ax.twinx()
            over_analysis['Total Runs Saved'].plot(kind='bar', ax=ax2, alpha=0.3, color='green', label='Runs Saved')
            ax2.set_ylabel('Total Runs Saved')
            
            # Combine legends
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left')
            
            plt.title('Fielding Actions and Runs Saved by Over')
            plt.tight_layout()
            st.pyplot(fig)
        
        elif viz_type == "Heatmap":
            st.subheader("Performance Heatmap")
            
            # Create pivot table for heatmap
            if len(st.session_state.fielding_data['Player Name'].unique()) >= 2 and len(st.session_state.fielding_data['Position'].unique()) >= 2:
                pivot_data = pd.pivot_table(
                    st.session_state.fielding_data,
                    values='Runs',
                    index='Player Name',
                    columns='Position',
                    aggfunc='sum',
                    fill_value=0
                )
                
                # Generate heatmap
                fig, ax = plt.subplots(figsize=(14, 10))
                sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=0, ax=ax)
                ax.set_title('Runs Saved by Player and Position')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Another heatmap for action frequency
                st.subheader("Action Frequency Heatmap")
                
                pivot_count = pd.pivot_table(
                    st.session_state.fielding_data,
                    values='Match No.',
                    index='Player Name',
                    columns='Position',
                    aggfunc='count',
                    fill_value=0
                )
                
                fig, ax = plt.subplots(figsize=(14, 10))
                sns.heatmap(pivot_count, annot=True, cmap='Blues', ax=ax)
                ax.set_title('Number of Actions by Player and Position')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Not enough data for heatmap visualization. Need at least 2 players and 2 positions.")

elif page == "Settings":
    st.header("Performance Score Weights Settings")
    
    st.write("""
    Adjust the weights used in the performance score calculation formula:
    
    PS = (CP × WCP) + (GT × WGT) + (C × WC) + (DC × WDC) + (ST × WST) + (RO × WRO) + (MRO × WMRO) + (DH × WDH) + RS
    
    Where:
    - CP: Clean Picks
    - GT: Good Throws
    - C: Catches
    - DC: Dropped Catches
    - ST: Stumpings
    - RO: Run Outs
    - MRO: Missed Run Outs
    - DH: Direct Hits
    - RS: Runs Saved (directly added to score, no weight)
    """)
    
    # Create form for weight adjustment
    with st.form(key="weight_settings_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            wcp = st.number_input("Clean Pick Weight (WCP)", value=st.session_state.weights['WCP'], step=0.5)
            wgt = st.number_input("Good Throw Weight (WGT)", value=st.session_state.weights['WGT'], step=0.5)
            wc = st.number_input("Catch Weight (WC)", value=st.session_state.weights['WC'], step=0.5)
            wdc = st.number_input("Dropped Catch Weight (WDC)", value=st.session_state.weights['WDC'], step=0.5)
        
        with col2:
            wst = st.number_input("Stumping Weight (WST)", value=st.session_state.weights['WST'], step=0.5)
            wro = st.number_input("Run Out Weight (WRO)", value=st.session_state.weights['WRO'], step=0.5)
            wmro = st.number_input("Missed Run Out Weight (WMRO)", value=st.session_state.weights['WMRO'], step=0.5)
            wdh = st.number_input("Direct Hit Weight (WDH)", value=st.session_state.weights['WDH'], step=0.5)
        
        submit_weights = st.form_submit_button(label="Save Weights")
    
    if submit_weights:
        st.session_state.weights = {
            'WCP': wcp,
            'WGT': wgt,
            'WC': wc,
            'WDC': wdc,
            'WST': wst,
            'WRO': wro,
            'WMRO': wmro,
            'WDH': wdh
        }
        save_weights(st.session_state.weights)
        st.success("Weights updated successfully!")
    
    # Preset weight configurations
    st.subheader("Preset Weight Configurations")
    
    preset_col1, preset_col2 = st.columns(2)
    
    with preset_col1:
        if st.button("Standard T20 Weights"):
            st.session_state.weights = {
                'WCP': 1.0,
                'WGT': 1.0,
                'WC': 3.0,
                'WDC': -2.0,
                'WST': 3.0,
                'WRO': 3.0,
                'WMRO': -2.0,
                'WDH': 2.0
            }
            save_weights(st.session_state.weights)
            st.success("Standard T20 weights applied!")
    
    with preset_col2:
        if st.button("Balanced Weights"):
            st.session_state.weights = {
                'WCP': 1.0,
                'WGT': 1.0,
                'WC': 2.0,
                'WDC': -1.5,
                'WST': 2.0,
                'WRO': 2.0,
                'WMRO': -1.5,
                'WDH': 1.5
            }
            save_weights(st.session_state.weights)
            st.success("Balanced weights applied!")
    
    # Export/Import weights
    st.subheader("Export/Import Weights")
    
    # Export weights
    weights_df = pd.DataFrame.from_dict(st.session_state.weights, orient='index', columns=['Value'])
    weights_df.index.name = 'Weight'
    
    st.markdown(
        get_download_link(weights_df, 'fielding_weights.csv', 'Download Current Weights as CSV'),
        unsafe_allow_html=True
    )
    
    # Import weights
    upload_weights = st.file_uploader("Import Weights from CSV", type="csv")
    if upload_weights is not None:
        try:
            imported_weights = pd.read_csv(upload_weights, index_col=0)
            if st.button("Apply Imported Weights"):
                new_weights = imported_weights['Value'].to_dict()
                # Validate that all required weights are present
                required_weights = ['WCP', 'WGT', 'WC', 'WDC', 'WST', 'WRO', 'WMRO', 'WDH']
                if all(w in new_weights for w in required_weights):
                    st.session_state.weights = {w: new_weights[w] for w in required_weights}
                    save_weights(st.session_state.weights)
                    st.success("Imported weights applied successfully!")
                else:
                    st.error("Invalid weights file. Missing required weights.")
        except Exception as e:
            st.error(f"Error reading weights file: {e}")

# Add a data management section to Settings page
if page == "Settings":
    # ... existing Settings code remains ...
    
    # Add data backup and restore features
    st.header("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Backup Data")
        if st.button("Create Manual Backup"):
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = os.path.join(DATA_DIR, "backups")
                
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir)
                
                # Backup fielding data
                backup_data_file = os.path.join(backup_dir, f"fielding_data_{timestamp}.csv")
                st.session_state.fielding_data.to_csv(backup_data_file, index=False)
                
                # Backup weights
                backup_weights_file = os.path.join(backup_dir, f"weights_{timestamp}.json")
                with open(backup_weights_file, 'w') as f:
                    json.dump(st.session_state.weights, f)
                
                st.success(f"Backup created successfully: {timestamp}")
                st.session_state.last_backup = datetime.now()
            except Exception as e:
                st.error(f"Error creating backup: {e}")
    
    with col2:
        st.subheader("Restore Data")
        
        # Find all backups
        backup_dir = os.path.join(DATA_DIR, "backups")
        if os.path.exists(backup_dir):
            backup_files = [f for f in os.listdir(backup_dir) if f.startswith("fielding_data_")]
            backup_timestamps = sorted(list(set([f.split("_", 2)[2].split(".")[0] for f in backup_files])), reverse=True)
            
            if backup_timestamps:
                selected_backup = st.selectbox("Select Backup to Restore", backup_timestamps)
                
                if st.button("Restore Selected Backup"):
                    try:
                        # Restore fielding data
                        backup_data_file = os.path.join(backup_dir, f"fielding_data_{selected_backup}.csv")
                        backup_weights_file = os.path.join(backup_dir, f"weights_{selected_backup}.json")
                        
                        if os.path.exists(backup_data_file):
                            restored_data = pd.read_csv(backup_data_file)
                            st.session_state.fielding_data = restored_data
                            save_fielding_data(restored_data)
                        
                        if os.path.exists(backup_weights_file):
                            with open(backup_weights_file, 'r') as f:
                                restored_weights = json.load(f)
                                st.session_state.weights = restored_weights
                                save_weights(restored_weights)
                        
                        st.success(f"Backup {selected_backup} restored successfully!")
                    except Exception as e:
                        st.error(f"Error restoring backup: {e}")
            else:
                st.info("No backups found.")
        else:
            st.info("No backups found. Create a backup first.")