import sqlite3

def init_db():
    with sqlite3.connect('data/data_analysis.db') as conn:
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS results')
        cursor.execute('''
        CREATE TABLE results (
            file_name TEXT,
            data_set TEXT,
            avg_t_db REAL,
            avg_t_wb REAL,
            avg_t_supply REAL,
            avg_t_return_emu REAL,
            avg_volume_flow REAL,
            avg_mass_flow REAL, 
            avg_heating_capacity_uncorr REAL,
            avg_power_input_uncorr REAL,
            cop_uncorr REAL,
            avg_heating_capacity_corr REAL,
            avg_power_input_corr REAL,
            cop_corr REAL,
            avg_dp REAL,
            avg_adj_dp REAL,
            avg_P_hyd REAL
        )
        ''')
        conn.commit()

if __name__ == "__main__":
    init_db()
