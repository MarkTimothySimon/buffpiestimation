import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import pandas as pd
import sqlite3
import os
import tempfile
import shutil

@dataclass
class RoundInfo:
    round_number: int
    intersections: int
    total_needles: int
    cumulative_pi: float

    @property
    def round_pi(self) -> float:
        """Calculate pi approximation for this individual round"""
        return (self.total_needles) / self.intersections if self.intersections > 0 else float('inf')

class BuffonNeedleSimulation:
    def __init__(self):
        # Use a consistent path that persists across sessions
        self.db_path = self._get_persistent_db_path()
        self.init_database()
        self.rounds = self.load_rounds()
    
    def _get_persistent_db_path(self):
        """Get a persistent database path that survives app restarts"""
        # Try to use a persistent directory, fallback to temp if needed
        if 'STREAMLIT_SHARING' in os.environ:
            # On Streamlit Cloud, use /tmp with a fixed name
            return "/tmp/buffon_needle_persistent.db"
        else:
            # Local development - use current directory
            return os.path.join(os.getcwd(), "buffon_needle.db")
    
    def init_database(self):
        """Initialize the SQLite database with the rounds table"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS rounds (
                        round_number INTEGER PRIMARY KEY,
                        intersections INTEGER NOT NULL,
                        total_needles INTEGER NOT NULL,
                        cumulative_pi REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except sqlite3.Error as e:
            st.error(f"Database initialization error: {e}")
            # Fallback to session state if database fails
            self._fallback_to_session_state()
    
    def _fallback_to_session_state(self):
        """Fallback to session state if database operations fail"""
        if 'rounds_data' not in st.session_state:
            st.session_state.rounds_data = []
        st.warning("⚠️ Database unavailable. Using temporary session storage.")
    
    def calculate_cumulative_pi(self, new_intersections: int, new_total_needles: int) -> float:
        total_intersections = sum(round.intersections for round in self.rounds) + new_intersections
        total_needles = sum(round.total_needles for round in self.rounds) + new_total_needles
        return (total_needles) / total_intersections if total_intersections > 0 else float('inf')
    
    def load_rounds(self) -> List[RoundInfo]:
        """Load rounds from SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT round_number, intersections, total_needles, cumulative_pi 
                    FROM rounds 
                    ORDER BY round_number
                """)
                return [
                    RoundInfo(
                        round_number=row[0],
                        intersections=row[1],
                        total_needles=row[2],
                        cumulative_pi=row[3]
                    )
                    for row in cursor.fetchall()
                ]
        except sqlite3.Error as e:
            st.error(f"Error loading data: {e}")
            return []
    
    def add_round(self, intersections: int, total_needles: int):
        """Add a new round to the database"""
        cumulative_pi = self.calculate_cumulative_pi(intersections, total_needles)
        next_round = len(self.rounds) + 1
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO rounds (round_number, intersections, total_needles, cumulative_pi)
                    VALUES (?, ?, ?, ?)
                """, (next_round, intersections, total_needles, cumulative_pi))
                conn.commit()
            
            # Reload rounds from database
            self.rounds = self.load_rounds()
            return True
        except sqlite3.Error as e:
            st.error(f"Error adding round: {e}")
            return False
    
    def get_rounds_for_display(self):
        """Returns rounds in reverse chronological order"""
        return list(reversed(self.rounds))
    
    def clear_data(self):
        """Clear all data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM rounds")
                conn.commit()
            self.rounds = []
            return True
        except sqlite3.Error as e:
            st.error(f"Error clearing data: {e}")
            return False
    
    def export_data_json(self):
        """Export data as JSON"""
        return pd.DataFrame([
            {
                'round_number': r.round_number,
                'intersections': r.intersections,
                'total_needles': r.total_needles,
                'cumulative_pi': r.cumulative_pi
            }
            for r in self.rounds
        ]).to_json(orient='records', indent=2)
    
    def export_data_csv(self):
        """Export data as CSV"""
        if not self.rounds:
            return None
        
        df = pd.DataFrame([
            {
                'round_number': r.round_number,
                'intersections': r.intersections,
                'total_needles': r.total_needles,
                'cumulative_pi': r.cumulative_pi
            }
            for r in self.rounds
        ])
        return df.to_csv(index=False)
    
    def import_data_json(self, json_data: str):
        """Import data from JSON"""
        try:
            df = pd.read_json(json_data)
            return self._import_dataframe(df)
        except Exception as e:
            st.error(f"Error importing JSON data: {e}")
            return False
    
    def import_data_csv(self, csv_data: str):
        """Import data from CSV"""
        try:
            import io
            df = pd.read_csv(io.StringIO(csv_data))
            return self._import_dataframe(df)
        except Exception as e:
            st.error(f"Error importing CSV data: {e}")
            return False
    
    def _import_dataframe(self, df):
        """Import data from a pandas DataFrame"""
        try:
            # Validate required columns
            required_columns = ['round_number', 'intersections', 'total_needles', 'cumulative_pi']
            if not all(col in df.columns for col in required_columns):
                st.error(f"Data must contain columns: {', '.join(required_columns)}")
                return False
            
            # Clear existing data
            self.clear_data()
            
            # Insert new data
            with sqlite3.connect(self.db_path) as conn:
                for _, row in df.iterrows():
                    conn.execute("""
                        INSERT INTO rounds (round_number, intersections, total_needles, cumulative_pi)
                        VALUES (?, ?, ?, ?)
                    """, (
                        int(row['round_number']),
                        int(row['intersections']),
                        int(row['total_needles']),
                        float(row['cumulative_pi'])
                    ))
                conn.commit()
            
            # Reload rounds
            self.rounds = self.load_rounds()
            return True
            
        except Exception as e:
            st.error(f"Error importing data: {e}")
            return False
    
    def get_database_info(self):
        """Get database file information"""
        try:
            if os.path.exists(self.db_path):
                size = os.path.getsize(self.db_path)
                return {
                    'path': self.db_path,
                    'size': f"{size} bytes",
                    'exists': True,
                    'records': len(self.rounds)
                }
            else:
                return {
                    'path': self.db_path,
                    'size': "0 bytes",
                    'exists': False,
                    'records': 0
                }
        except Exception as e:
            return {
                'path': self.db_path,
                'error': str(e),
                'exists': False,
                'records': 0
            }

def plot_pi_approximation(rounds: List[RoundInfo], figsize=(10, 6)):
    if not rounds:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Buffon's Needle Pi Approximation Over Rounds")
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    round_numbers = [r.round_number for r in rounds]
    cumulative_pi_estimates = [r.cumulative_pi for r in rounds]
    
    ax.plot(round_numbers, cumulative_pi_estimates, 'b-', linewidth=2, label='Pi Approximation')
    ax.axhline(y=np.pi, color='r', linestyle='--', linewidth=2, label=f'π ≈ {np.pi:.3f}')
    
    # Set reasonable y-limits
    min_val = min(cumulative_pi_estimates)
    max_val = max(cumulative_pi_estimates)
    data_range = max_val - min_val
    padding = max(data_range * 0.1, 0.1)
    y_min = min_val - padding
    y_max = max_val + padding
    ax.set_ylim([y_min, y_max])
    
    # Add pi annotation if it's visible
    if y_min <= np.pi <= y_max:
        ax.annotate(
            f'π ≈ {np.pi:.3f}',
            xy=(max(round_numbers) * 0.1, np.pi),
            xytext=(max(round_numbers) * 0.2, np.pi),
            color='red',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7)
        )
    
    ax.set_xlabel('Round Number')
    ax.set_ylabel('Pi Approximation')
    ax.set_title("Buffon's Needle Pi Approximation Over Rounds")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def main():
    st.set_page_config(
        page_title="Buffon's Needle Simulation",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 Buffon's Needle Pi Estimation")
    st.markdown("*Estimate π using the famous Buffon's needle experiment*")
    
    # Initialize simulation
    if 'simulation' not in st.session_state:
        st.session_state.simulation = BuffonNeedleSimulation()
    
    # Sidebar for data management
    with st.sidebar:
        st.header("📊 Data Management")
        
        # Database info
        db_info = st.session_state.simulation.get_database_info()
        with st.expander("🗄️ Database Info"):
            st.write(f"**Path:** `{db_info['path']}`")
            st.write(f"**Size:** {db_info.get('size', 'Unknown')}")
            st.write(f"**Records:** {db_info.get('records', 0)}")
            st.write(f"**Exists:** {'Yes' if db_info.get('exists', False) else 'No'}")
        
        # Export data
        st.subheader("📤 Export Data")
        if st.session_state.simulation.rounds:
            col_json, col_csv = st.columns(2)
            
            with col_json:
                export_json = st.session_state.simulation.export_data_json()
                st.download_button(
                    label="💾 JSON",
                    data=export_json,
                    file_name=f"buffon_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col_csv:
                export_csv = st.session_state.simulation.export_data_csv()
                if export_csv:
                    st.download_button(
                        label="📊 CSV",
                        data=export_csv,
                        file_name=f"buffon_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        else:
            st.info("No data to export")
        
        # Import data
        st.subheader("📥 Import Data")
        uploaded_file = st.file_uploader(
            "Upload data file", 
            type=['json', 'csv'],
            help="Upload a JSON or CSV file with experiment data"
        )
        
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1].lower()
            file_content = uploaded_file.read().decode('utf-8')
            
            col_import1, col_import2 = st.columns(2)
            
            with col_import1:
                if st.button("📥 Import", type="primary", use_container_width=True):
                    success = False
                    if file_type == 'json':
                        success = st.session_state.simulation.import_data_json(file_content)
                    elif file_type == 'csv':
                        success = st.session_state.simulation.import_data_csv(file_content)
                    
                    if success:
                        st.success(f"Data imported successfully from {file_type.upper()}!")
                        st.rerun()
            
            with col_import2:
                if st.button("🔍 Preview", use_container_width=True):
                    try:
                        if file_type == 'csv':
                            import io
                            preview_df = pd.read_csv(io.StringIO(file_content))
                            st.dataframe(preview_df.head(), use_container_width=True)
                        elif file_type == 'json':
                            preview_df = pd.read_json(file_content)
                            st.dataframe(preview_df.head(), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error previewing file: {e}")
        
        # Data summary
        st.subheader("📈 Summary")
        if st.session_state.simulation.rounds:
            total_rounds = len(st.session_state.simulation.rounds)
            latest_pi = st.session_state.simulation.rounds[-1].cumulative_pi
            error = abs(latest_pi - np.pi)
            
            st.metric("Total Rounds", total_rounds)
            st.metric("Current π Estimate", f"{latest_pi:.6f}")
            st.metric("Error from π", f"{error:.6f}")
        else:
            st.info("No data available yet")
    
    # Main content
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.subheader("📊 Pi Convergence Graph")
        fig = plot_pi_approximation(st.session_state.simulation.rounds)
        st.pyplot(fig)
        
        # Clear data section
        st.subheader("🗑️ Data Management")
        
        # Initialize dialog state
        if 'show_clear_dialog' not in st.session_state:
            st.session_state.show_clear_dialog = False
        
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.show_clear_dialog = True
        
        if st.session_state.show_clear_dialog:
            st.warning("⚠️ Are you sure you want to clear all data from the database?")
            col_confirm1, col_confirm2 = st.columns(2)
            with col_confirm1:
                if st.button("✅ Yes, Clear Data", type="primary"):
                    if st.session_state.simulation.clear_data():
                        st.success("Data cleared successfully!")
                    st.session_state.show_clear_dialog = False
                    st.rerun()
            with col_confirm2:
                if st.button("❌ Cancel"):
                    st.session_state.show_clear_dialog = False
                    st.rerun()
    
    with col2:
        st.subheader("➕ Add New Round")
        
        with st.form("new_round_form"):
            st.markdown("*Enter the results of your needle drop experiment:*")
            
            total_needles = st.number_input(
                "🎯 Total number of needles dropped",
                min_value=1,
                value=100,
                help="More needles = better accuracy"
            )
            
            intersections = st.number_input(
                "⚡ Number of intersections with lines",
                min_value=1,
                max_value=total_needles,
                value=max(1, int(total_needles / 3.14159)),
                help="Count how many needles crossed the parallel lines"
            )
            
            submitted = st.form_submit_button(
                "🚀 Add Round",
                type="primary",
                use_container_width=True
            )
            
            if submitted and intersections > 0:
                if st.session_state.simulation.add_round(intersections, total_needles):
                    st.success(f"Round {len(st.session_state.simulation.rounds)} added!")
                    st.rerun()
        
        # Display rounds history
        st.subheader("📜 Rounds History")
        all_rounds = st.session_state.simulation.get_rounds_for_display()
        
        if all_rounds:
            df = pd.DataFrame([
                {
                    "Round": r.round_number,
                    "Intersections": r.intersections,
                    "Total Needles": r.total_needles,
                    "Round π": f"{r.round_pi:.4f}",
                    "Cumulative π": f"{r.cumulative_pi:.4f}",
                    "Error": f"{abs(r.cumulative_pi - np.pi):.4f}"
                }
                for r in all_rounds
            ])
            
            st.dataframe(
                df,
                height=400,
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("🎲 No rounds recorded yet. Add your first experiment above!")

if __name__ == "__main__":
    main()