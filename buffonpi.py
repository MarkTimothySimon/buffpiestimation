import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import pandas as pd
import json

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

    def to_dict(self):
        return {
            'round_number': self.round_number,
            'intersections': self.intersections,
            'total_needles': self.total_needles,
            'cumulative_pi': self.cumulative_pi
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            round_number=data['round_number'],
            intersections=data['intersections'],
            total_needles=data['total_needles'],
            cumulative_pi=data['cumulative_pi']
        )

class BuffonNeedleSimulation:
    def __init__(self):
        # Initialize from session state instead of file
        if 'rounds_data' not in st.session_state:
            st.session_state.rounds_data = []
        self.rounds = self.load_rounds()
        
    def calculate_cumulative_pi(self, new_intersections: int, new_total_needles: int) -> float:
        total_intersections = sum(round.intersections for round in self.rounds) + new_intersections
        total_needles = sum(round.total_needles for round in self.rounds) + new_total_needles
        return (total_needles) / total_intersections if total_intersections > 0 else float('inf')
    
    def load_rounds(self) -> List[RoundInfo]:
        """Load rounds from session state"""
        return [RoundInfo.from_dict(data) for data in st.session_state.rounds_data]
    
    def save_rounds(self):
        """Save rounds to session state"""
        st.session_state.rounds_data = [round.to_dict() for round in self.rounds]
    
    def add_round(self, intersections: int, total_needles: int):
        cumulative_pi = self.calculate_cumulative_pi(intersections, total_needles)
        next_round = len(self.rounds) + 1
        
        new_round = RoundInfo(
            round_number=next_round,
            intersections=intersections,
            total_needles=total_needles,
            cumulative_pi=cumulative_pi
        )
        
        self.rounds.append(new_round)
        self.save_rounds()
    
    def get_rounds_for_display(self):
        """Returns rounds in reverse chronological order"""
        return list(reversed(self.rounds))
    
    def clear_data(self):
        """Clear all data from session state"""
        st.session_state.rounds_data = []
        self.rounds = []
    
    def export_data(self):
        """Export data as JSON for download"""
        return json.dumps([round.to_dict() for round in self.rounds], indent=2)
    
    def import_data(self, json_data: str):
        """Import data from JSON"""
        try:
            data = json.loads(json_data)
            st.session_state.rounds_data = data
            self.rounds = self.load_rounds()
            return True
        except Exception as e:
            st.error(f"Error importing data: {e}")
            return False
    
    def import_csv_data(self, csv_data: str):
        """Import data from CSV"""
        try:
            # Parse CSV data
            import io
            csv_file = io.StringIO(csv_data)
            df = pd.read_csv(csv_file)
            
            # Validate required columns
            required_columns = ['round_number', 'intersections', 'total_needles', 'cumulative_pi']
            if not all(col in df.columns for col in required_columns):
                st.error(f"CSV must contain columns: {', '.join(required_columns)}")
                return False
            
            # Convert to rounds data
            rounds_data = []
            for _, row in df.iterrows():
                rounds_data.append({
                    'round_number': int(row['round_number']),
                    'intersections': int(row['intersections']),
                    'total_needles': int(row['total_needles']),
                    'cumulative_pi': float(row['cumulative_pi'])
                })
            
            # Sort by round number to ensure correct order
            rounds_data.sort(key=lambda x: x['round_number'])
            
            st.session_state.rounds_data = rounds_data
            self.rounds = self.load_rounds()
            return True
            
        except Exception as e:
            st.error(f"Error importing CSV data: {e}")
            return False
    
    def export_csv_data(self):
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
    padding = max(data_range * 0.1, 0.1)  # Ensure minimum padding
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
    
    # Add JavaScript for localStorage persistence
    st.components.v1.html("""
    <script>
    function saveToLocalStorage() {
        const data = window.parent.document.querySelector('iframe').contentWindow.sessionStorage.getItem('buffon_data');
        if (data) {
            localStorage.setItem('buffon_needle_data', data);
        }
    }
    
    function loadFromLocalStorage() {
        const data = localStorage.getItem('buffon_needle_data');
        if (data) {
            return data;
        }
        return null;
    }
    
    // Save data periodically
    setInterval(saveToLocalStorage, 5000);
    </script>
    """, height=0)
    
    # Initialize simulation in session state
    if 'simulation' not in st.session_state:
        st.session_state.simulation = BuffonNeedleSimulation()
    
    # Try to restore data on first load
    if 'data_restored' not in st.session_state:
        st.session_state.data_restored = True
        if not st.session_state.simulation.rounds:
            st.info("💡 **Tip**: Your data will be lost on page refresh. Use Export/Import to save permanently!")
    
    # Sidebar for data management
    with st.sidebar:
        st.header("📊 Data Management")
        
        # Data persistence warning
        st.warning("⚠️ **Data Persistence**: Your data will be lost if you refresh the page. Always export your data before closing!")
        
        # Quick save/load buttons
        st.subheader("💾 Quick Save/Load")
        col_save, col_load = st.columns(2)
        
        with col_save:
            if st.session_state.simulation.rounds:
                auto_save_data = st.session_state.simulation.export_data()
                st.download_button(
                    label="💾 Quick Save",
                    data=auto_save_data,
                    file_name=f"buffon_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Save current data with timestamp",
                    use_container_width=True
                )
        
        with col_load:
            if st.button("🔄 Load Last Export", help="Upload your last exported file", use_container_width=True):
                st.info("👆 Use the file uploader below to load your saved data")
        
        st.divider()
        
        # Export data
        st.subheader("📤 Export Data")
        if st.session_state.simulation.rounds:
            col_json, col_csv = st.columns(2)
            
            with col_json:
                export_data = st.session_state.simulation.export_data()
                st.download_button(
                    label="💾 JSON",
                    data=export_data,
                    file_name="buffon_needle_data.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col_csv:
                csv_data = st.session_state.simulation.export_csv_data()
                if csv_data:
                    st.download_button(
                        label="📊 CSV",
                        data=csv_data,
                        file_name="buffon_needle_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        # Import data
        st.subheader("📤 Import Data")
        
        # File uploader for both JSON and CSV
        uploaded_file = st.file_uploader(
            "Upload data file", 
            type=['json', 'csv'],
            help="Upload a JSON or CSV file with your experiment data"
        )
        
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1].lower()
            file_content = uploaded_file.read().decode('utf-8')
            
            col_import1, col_import2 = st.columns(2)
            
            with col_import1:
                if st.button("📥 Import", type="primary", use_container_width=True):
                    success = False
                    if file_type == 'json':
                        success = st.session_state.simulation.import_data(file_content)
                    elif file_type == 'csv':
                        success = st.session_state.simulation.import_csv_data(file_content)
                    
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
                            import json
                            preview_data = json.loads(file_content)
                            if isinstance(preview_data, list) and preview_data:
                                st.json(preview_data[:3])  # Show first 3 records
                            else:
                                st.json(preview_data)
                    except Exception as e:
                        st.error(f"Error previewing file: {e}")
        
        # CSV format help
        with st.expander("📋 CSV Format Help"):
            st.markdown("""
            **Required CSV columns:**
            - `round_number`: Round number (integer)
            - `intersections`: Number of intersections (integer)  
            - `total_needles`: Total needles dropped (integer)
            - `cumulative_pi`: Cumulative pi estimate (decimal)
            
            **Example CSV:**
            ```
            round_number,intersections,total_needles,cumulative_pi
            1,32,100,3.125000
            2,63,200,3.174603
            3,95,300,3.157895
            ```
            """)

        
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
        
        # Initialize dialog state if not exists
        if 'show_clear_dialog' not in st.session_state:
            st.session_state.show_clear_dialog = False
        
        # Show clear data button
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.show_clear_dialog = True
        
        # Show confirmation dialog
        if st.session_state.show_clear_dialog:
            st.warning("⚠️ Are you sure you want to clear all data? This action cannot be undone!")
            col_confirm1, col_confirm2 = st.columns(2)
            with col_confirm1:
                if st.button("✅ Yes, Clear Data", type="primary"):
                    st.session_state.simulation.clear_data()
                    st.session_state.show_clear_dialog = False
                    st.success("Data cleared successfully!")
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
            
            max_intersections = total_needles
            intersections = st.number_input(
                "⚡ Number of intersections with lines",
                min_value=1,
                max_value=max_intersections,
                value=max(1, int(total_needles / 3.14159)),  # Rough estimate
                help="Count how many needles crossed the parallel lines"
            )
            
            submitted = st.form_submit_button(
                "🚀 Add Round",
                type="primary",
                use_container_width=True
            )
            
            if submitted and intersections > 0:
                st.session_state.simulation.add_round(intersections, total_needles)
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