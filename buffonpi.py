import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import pandas as pd
import sqlite3
import os

@dataclass
class RoundInfo:
    round_number: int
    intersections: int
    total_needles: int
    cumulative_pi: float

class BuffonNeedleSimulation:
    def __init__(self, db_path: str = "buffon_needle.db"):
        self.db_path = db_path
        self.init_database()
        self.rounds = self.load_rounds()
        
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rounds (
                    round_number INTEGER PRIMARY KEY,
                    intersections INTEGER NOT NULL,
                    total_needles INTEGER NOT NULL,
                    cumulative_pi REAL NOT NULL
                )
            """)
    
    def calculate_cumulative_pi(self, new_intersections: int, new_total_needles: int) -> float:
        total_intersections = sum(round.intersections for round in self.rounds) + new_intersections
        total_needles = sum(round.total_needles for round in self.rounds) + new_total_needles
        return (2 * total_needles) / total_intersections if total_intersections > 0 else float('inf')
    
    def load_rounds(self) -> List[RoundInfo]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT round_number, intersections, total_needles, cumulative_pi FROM rounds ORDER BY round_number")
            return [
                RoundInfo(
                    round_number=row[0],
                    intersections=row[1],
                    total_needles=row[2],
                    cumulative_pi=row[3]
                )
                for row in cursor.fetchall()
            ]
    
    def add_round(self, intersections: int, total_needles: int):
        cumulative_pi = self.calculate_cumulative_pi(intersections, total_needles)
        next_round = len(self.rounds) + 1
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO rounds (round_number, intersections, total_needles, cumulative_pi)
                VALUES (?, ?, ?, ?)
            """, (next_round, intersections, total_needles, cumulative_pi))
        
        self.rounds = self.load_rounds()
    
    def get_rounds_for_display(self):
        return list(reversed(self.rounds))
    
    def clear_data(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.init_database()
        self.rounds = []

def plot_pi_approximation(rounds: List[RoundInfo]):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    if not rounds:
        ax.text(0.5, 0.5, 'No data yet', horizontalalignment='center', verticalalignment='center')
        ax.set_axis_off()
    else:
        round_numbers = [r.round_number for r in rounds]
        cumulative_pi_estimates = [r.cumulative_pi for r in rounds]
        
        # Plot the data and reference line
        ax.plot(round_numbers, cumulative_pi_estimates, 'b-', label='Pi Approximation')
        ax.axhline(y=np.pi, color='r', linestyle='--', label="Actual Pi")
        
        # Calculate y-axis limits based on data range
        min_val = min(cumulative_pi_estimates)
        max_val = max(cumulative_pi_estimates)
        data_range = max_val - min_val
        
        # Add a small padding to the data range
        padding = data_range * 0.1 if data_range > 0 else 0.1
        y_min = min_val - padding
        y_max = max_val + padding
        
        # Set y-limits based on data, without forcing π to be included
        ax.set_ylim([y_min, y_max])
        
        # Only add π annotation if π falls within the y-axis range, and position it on the y-axis
        if y_min <= np.pi <= y_max:
            # Position the text at the left edge of the y-axis (outside the plot area, aligned with y-ticks)
            ax.text(
                x=-0.02,  # Slightly left of the y-axis (negative to place outside plot area)
                y=np.pi,  # Position exactly at π's value
                s=f'π ≈ {np.pi:.3f}',
                color='r',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                verticalalignment='center',  # Center vertically on the π line
                horizontalalignment='right',  # Align text to the right, next to the y-axis
                transform=ax.get_yaxis_transform()  # Use y-axis transform for proper alignment with y-ticks
            )
        
        ax.set_xlabel('Round Number')
        ax.set_ylabel('Pi Approximation')
        ax.set_title("Buffon's Needle Pi Approximation")
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout(pad=1.0)
    return fig

def main():
    st.set_page_config(layout="wide")
    st.title("Buffon's Needle Pi Estimation")
    
    if 'simulation' not in st.session_state:
        st.session_state.simulation = BuffonNeedleSimulation()
    
    col_left, col_right = st.columns([0.6, 0.4])
    
    with col_left:
        st.subheader("Pi Approximation Graph")
        with st.container():
            fig = plot_pi_approximation(st.session_state.simulation.rounds)
            st.pyplot(fig, use_container_width=True)
        
        st.subheader("Data Options")
        if 'show_clear_dialog' not in st.session_state:
            st.session_state.show_clear_dialog = False
            
        if st.button("Clear All Data", key="clear_btn"):
            st.session_state.show_clear_dialog = True
            
        if st.session_state.show_clear_dialog:
            st.warning("Are you sure you want to clear all data?")
            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("Yes, Clear Data", type="primary", key="confirm_clear"):
                    st.session_state.simulation.clear_data()
                    st.session_state.show_clear_dialog = False
                    st.rerun()
            with col_cancel:
                if st.button("Cancel", key="cancel_clear"):
                    st.session_state.show_clear_dialog = False
                    st.rerun()

    with col_right:
        st.subheader("Add New Round")
        with st.form("new_round_form"):
            total_needles = st.number_input(
                "Total number of sticks",
                min_value=1,
                value=20
            )
            
            max_intersections = total_needles
            intersections = st.number_input(
                "Number of intersections",
                min_value=1,
                max_value=max_intersections,
                value=int((total_needles)/3)
            )
            
            submitted = st.form_submit_button("Next Round")
            
            if submitted and intersections > 0:
                st.session_state.simulation.add_round(intersections, total_needles)
                st.rerun()
        
        st.subheader("Rounds History")
        all_rounds = st.session_state.simulation.get_rounds_for_display()
        
        if all_rounds:
            df = pd.DataFrame([
                {
                    "Round": r.round_number,
                    "Total Sticks": r.total_needles,
                    "Intersected Sticks": r.intersections,
                    "π Approx": f"{r.cumulative_pi:.6f}"
                }
                for r in all_rounds
            ])
            
            st.dataframe(
                df,
                height=300,
                use_container_width=True,
                hide_index=True
            )

if __name__ == "__main__":
    main()
