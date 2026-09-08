#!/usr/bin/env python3
"""
Single Database Management Tool
Handles adding columns and updating time ranges
"""

import sqlite3
import os

def show_table_structure(db_path):
    """Show current table structure"""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(results)")
        columns = cursor.fetchall()
        
        print(f"\nCurrent table structure:")
        print("-" * 60)
        for col in columns:
            print(f"{col[1]:<30} {col[2]:<15}")
        print("-" * 60)
        print(f"Total columns: {len(columns)}")
        
        # Show record count
        cursor.execute("SELECT COUNT(*) FROM results")
        count = cursor.fetchone()[0]
        print(f"Records: {count}")
        
    finally:
        conn.close()

def add_columns(db_path, columns):
    """Add new columns to existing table"""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        
        # Get current columns to avoid duplicates
        cursor.execute("PRAGMA table_info(results)")
        current_columns = [col[1] for col in cursor.fetchall()]
        
        for col_name, col_type in columns:
            if col_name in current_columns:
                print(f"WARNING: Column '{col_name}' already exists, skipping...")
                continue
            
            print(f"Adding column '{col_name}' with type '{col_type}'...")
            cursor.execute(f"ALTER TABLE results ADD COLUMN {col_name} {col_type}")
            print(f"SUCCESS: Added column '{col_name}'")
        
        conn.commit()
        print("SUCCESS: All columns added successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    finally:
        conn.close()

def add_calculated_column(db_path, col_name, col_type, calculation_type):
    """Add a calculated column based on time series data"""
    import pandas as pd
    import json
    
    # Load config for data directory
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(results)")
        current_columns = [col[1] for col in cursor.fetchall()]
        
        if col_name in current_columns:
            print(f"WARNING: Column '{col_name}' already exists!")
            return False
        
        # Add the column first
        print(f"Adding calculated column '{col_name}' with type '{col_type}'...")
        cursor.execute(f"ALTER TABLE results ADD COLUMN {col_name} {col_type}")
        
        # Get all records that need calculation
        cursor.execute("SELECT file_name, data_set, start_time, end_time FROM results WHERE start_time IS NOT NULL AND end_time IS NOT NULL")
        records = cursor.fetchall()
        
        print(f"Calculating values for {len(records)} records...")
        
        calculated_count = 0
        errors = []
        
        for file_name, data_set, start_time, end_time in records:
            try:
                # Process the Excel file
                file_path = os.path.join(config['data_dir'], file_name)
                if not os.path.exists(file_path):
                    errors.append(f"File not found: {file_name}")
                    continue
                
                df = pd.read_excel(file_path)
                if 'time_elapsed' not in df.columns:
                    errors.append(f"No time_elapsed column in {file_name}")
                    continue
                
                # Filter data by time range
                df_filtered = df[(df['time_elapsed'] >= start_time) & (df['time_elapsed'] <= end_time)]
                
                if df_filtered.empty:
                    errors.append(f"No data in time range for {file_name}")
                    continue
                
                # Calculate based on type
                if calculation_type == "avg_temperature":
                    if 'T_outdoor (DB)' in df_filtered.columns:
                        calculated_value = df_filtered['T_outdoor (DB)'].mean()
                    else:
                        errors.append(f"No temperature column in {file_name}")
                        continue
                        
                elif calculation_type == "avg_humidity":
                    if 'T_outdoor (WB)' in df_filtered.columns and 'T_outdoor (DB)' in df_filtered.columns:
                        # Calculate relative humidity from wet bulb and dry bulb
                        tw = df_filtered['T_outdoor (WB)']
                        td = df_filtered['T_outdoor (DB)']
                        # Simplified humidity calculation
                        calculated_value = ((tw - td) / (td + 273.15) * 100).mean()
                    else:
                        errors.append(f"No wet/dry bulb columns in {file_name}")
                        continue
                        
                elif calculation_type == "time_duration":
                    calculated_value = end_time - start_time
                    
                elif calculation_type == "data_points":
                    calculated_value = len(df_filtered)
                    
                elif calculation_type == "avg_pressure":
                    if 'Pressure difference' in df_filtered.columns:
                        calculated_value = df_filtered['Pressure difference'].mean()
                    else:
                        errors.append(f"No pressure column in {file_name}")
                        continue
                        
                elif calculation_type == "avg_flow":
                    if 'volume flow' in df_filtered.columns:
                        calculated_value = df_filtered['volume flow'].mean()
                    else:
                        errors.append(f"No flow column in {file_name}")
                        continue
                        
                else:
                    errors.append(f"Unknown calculation type: {calculation_type}")
                    continue
                
                # Update the record
                cursor.execute(f"UPDATE results SET {col_name} = ? WHERE file_name = ? AND data_set = ?", 
                             (calculated_value, file_name, data_set))
                calculated_count += 1
                print(f"  Calculated {col_name} for {file_name}, dataset {data_set}: {calculated_value:.4f}")
                
            except Exception as e:
                errors.append(f"Error processing {file_name}: {e}")
                print(f"  ERROR processing {file_name}: {e}")
        
        conn.commit()
        
        print(f"\nCALCULATION RESULTS:")
        print(f"Successfully calculated: {calculated_count} records")
        
        if errors:
            print(f"Errors: {len(errors)}")
            for error in errors[:5]:
                print(f"  {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    finally:
        conn.close()

def interactive_add_columns():
    """Interactive column addition with different types"""
    db_path = '../notebooks/data_analysis.db'
    
    print("\nAdd New Columns")
    print("=" * 20)
    print("Column types:")
    print("1. Simple column (user-provided content)")
    print("2. Calculated column (from time series data)")
    
    try:
        col_type_choice = input("\nChoose column type (1-2): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    
    if col_type_choice == '1':
        # Simple column
        print("\nAdding simple column...")
        print("Format: column_name:type (e.g., notes:TEXT, created_at:TIMESTAMP)")
        
        columns = []
        while True:
            try:
                col_input = input("Column (name:type) or 'done': ").strip()
                if col_input.lower() == 'done':
                    break
                if ':' not in col_input:
                    print("ERROR: Format: name:type")
                    continue
                name, col_type = col_input.split(':', 1)
                columns.append((name.strip(), col_type.strip()))
            except (EOFError, KeyboardInterrupt):
                break
        
        if columns:
            add_columns(db_path, columns)
        else:
            print("No columns to add.")
    
    elif col_type_choice == '2':
        # Calculated column
        print("\nAdding calculated column...")
        print("Available calculations:")
        print("1. avg_temperature - Average outdoor temperature")
        print("2. avg_humidity - Average relative humidity")
        print("3. time_duration - Duration (end_time - start_time)")
        print("4. data_points - Number of data points in range")
        print("5. avg_pressure - Average pressure difference")
        print("6. avg_flow - Average volume flow")
        
        try:
            calc_choice = input("\nChoose calculation (1-6): ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        
        calc_mapping = {
            '1': 'avg_temperature',
            '2': 'avg_humidity', 
            '3': 'time_duration',
            '4': 'data_points',
            '5': 'avg_pressure',
            '6': 'avg_flow'
        }
        
        if calc_choice in calc_mapping:
            calculation_type = calc_mapping[calc_choice]
            
            try:
                col_name = input("Column name: ").strip()
                col_type = input("Column type (REAL/INTEGER/TEXT): ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            
            if col_name and col_type:
                add_calculated_column(db_path, col_name, col_type, calculation_type)
            else:
                print("ERROR: Column name and type required")
        else:
            print("ERROR: Invalid choice")
    
    else:
        print("ERROR: Invalid choice")

def update_time_ranges():
    """Update time ranges for existing records"""
    db_path = '../notebooks/data_analysis.db'
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found: {db_path}")
        return False
    
    # Your time data
    time_data = [
        ("HPT_RRT1_Lab02_D.xlsx", 1, 0, 3600),
        ("HPT_RRT2_Lab04_E.xlsx", 1, 10, 4000),
        ("HPT_RRT2_Lab04_E.xlsx", 2, 4010, 8000),
    ]
    
    print(f"Updating {len(time_data)} records with time ranges...")
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        updated_count = 0
        errors = []
        
        for file_name, data_set, start_time, end_time in time_data:
            print(f"Updating: {file_name}, dataset {data_set}, times {start_time}-{end_time}")
            
            cursor.execute("UPDATE results SET start_time = ?, end_time = ? WHERE file_name = ? AND data_set = ?", 
                         (start_time, end_time, file_name, data_set))
            
            if cursor.rowcount > 0:
                updated_count += 1
                print(f"  SUCCESS: Updated record")
            else:
                errors.append(f"No matching record found for {file_name}, dataset {data_set}")
                print(f"  ERROR: No matching record found")
        
        conn.commit()
        
        print(f"\nRESULTS:")
        print(f"Successfully updated: {updated_count} records")
        
        if errors:
            print(f"Errors: {len(errors)}")
            for error in errors:
                print(f"  {error}")
        
        # Show remaining NULL records
        cursor.execute("SELECT COUNT(*) FROM results WHERE start_time IS NULL OR end_time IS NULL")
        remaining = cursor.fetchone()[0]
        print(f"Records still with NULL values: {remaining}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    finally:
        conn.close()

def update_times_from_file():
    """Update time ranges from a text file"""
    db_path = '../notebooks/data_analysis.db'
    
    print("\nTime Range Update from File")
    print("=" * 35)
    print("Create a text file with your time data in this format:")
    print("file_name data_set start_time end_time")
    print("One entry per line.")
    print("\nExample file content:")
    print("HPT_RRT2_Lab04_E.xlsx 1 10 4000")
    print("HPT_RRT2_Lab04_E.xlsx 2 4010 8000")
    print()
    
    try:
        filename = input("Enter filename (e.g., time_data.txt): ").strip()
        if not filename:
            print("No filename provided.")
            return
    except (EOFError, KeyboardInterrupt):
        return
    
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
    
    # Read data from file
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    if not lines:
        print("No data found in file.")
        return
    
    print(f"Found {len(lines)} entries in {filename}")
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        updated_count = 0
        errors = []
        
        for i, line in enumerate(lines, 1):
            try:
                parts = line.split()
                if len(parts) < 4:
                    errors.append(f"Line {i}: Not enough values - '{line}'")
                    continue
                
                file_name = ' '.join(parts[:-3])
                data_set = float(parts[-3])
                start_time = float(parts[-2])
                end_time = float(parts[-1])
                
                print(f"Processing: {file_name}, dataset {data_set}, times {start_time}-{end_time}")
                
                # First check if record already has time values
                cursor.execute("SELECT start_time, end_time FROM results WHERE file_name = ? AND (data_set = ? OR data_set = ?)", 
                             (file_name, data_set, float(data_set)))
                existing = cursor.fetchone()
                
                if existing and existing[0] is not None and existing[1] is not None:
                    print(f"  SKIPPED: Record already has time values (start: {int(existing[0])}, end: {int(existing[1])})")
                    continue
                
                # Update the record
                cursor.execute("UPDATE results SET start_time = ?, end_time = ? WHERE file_name = ? AND (data_set = ? OR data_set = ?)", 
                             (start_time, end_time, file_name, data_set, float(data_set)))
                
                if cursor.rowcount > 0:
                    updated_count += 1
                    print(f"  SUCCESS: Updated record")
                else:
                    errors.append(f"Line {i}: No matching record found - '{line}'")
                    print(f"  ERROR: No matching record found")
                    
            except ValueError as e:
                errors.append(f"Line {i}: Invalid numbers - '{line}' ({e})")
                print(f"  ERROR: Invalid numbers - {e}")
            except Exception as e:
                errors.append(f"Line {i}: Error - '{line}' ({e})")
                print(f"  ERROR: {e}")
        
        conn.commit()
        
        print(f"\nRESULTS:")
        print(f"Successfully updated: {updated_count} records")
        
        if errors:
            print(f"Errors: {len(errors)}")
            for error in errors[:5]:
                print(f"  {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
        
        # Show remaining NULL records
        cursor.execute("SELECT COUNT(*) FROM results WHERE start_time IS NULL OR end_time IS NULL")
        remaining = cursor.fetchone()[0]
        print(f"Records still with NULL values: {remaining}")
        
    finally:
        conn.close()

def interactive_time_input():
    """Interactive time range input - for small amounts of data"""
    db_path = '../notebooks/data_analysis.db'
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        
        # Get records with NULL times
        cursor.execute("SELECT file_name, data_set FROM results WHERE start_time IS NULL OR end_time IS NULL ORDER BY file_name, data_set")
        records = cursor.fetchall()
        
        if not records:
            print("All records already have start_time and end_time values!")
            return
        
        print(f"\nFound {len(records)} records needing time input")
        print("This is for small amounts of data. For large datasets, use 'Update from file' option.")
        print("Enter your data in format: file_name data_set start_time end_time")
        print("One entry per line. Type 'done' when finished.")
        print("\nExample:")
        print("HPT_RRT2_Lab04_E.xlsx 1 10 4000")
        print()
        
        lines = []
        print("Enter your data (type 'done' when finished):")
        while True:
            try:
                line = input().strip()
                if line.lower() == 'done':
                    break
                if line:
                    lines.append(line)
            except (EOFError, KeyboardInterrupt):
                break
        
        if not lines:
            print("No data entered.")
            return
        
        print(f"\nProcessing {len(lines)} entries...")
        
        updated_count = 0
        errors = []
        
        for i, line in enumerate(lines, 1):
            try:
                parts = line.split()
                if len(parts) < 4:
                    errors.append(f"Line {i}: Not enough values")
                    continue
                
                file_name = ' '.join(parts[:-3])
                data_set = float(parts[-3])
                start_time = float(parts[-2])
                end_time = float(parts[-1])
                
                print(f"Processing: {file_name}, dataset {data_set}, times {start_time}-{end_time}")
                
                # First check if record already has time values
                cursor.execute("SELECT start_time, end_time FROM results WHERE file_name = ? AND (data_set = ? OR data_set = ?)", 
                             (file_name, data_set, float(data_set)))
                existing = cursor.fetchone()
                
                if existing and existing[0] is not None and existing[1] is not None:
                    print(f"  SKIPPED: Record already has time values (start: {int(existing[0])}, end: {int(existing[1])})")
                    continue
                
                # Update the record
                cursor.execute("UPDATE results SET start_time = ?, end_time = ? WHERE file_name = ? AND (data_set = ? OR data_set = ?)", 
                             (start_time, end_time, file_name, data_set, float(data_set)))
                
                if cursor.rowcount > 0:
                    updated_count += 1
                    print(f"  SUCCESS: Updated record")
                else:
                    errors.append(f"Line {i}: No matching record found")
                    print(f"  ERROR: No matching record found")
                    
            except ValueError as e:
                errors.append(f"Line {i}: Invalid numbers - {e}")
                print(f"  ERROR: Invalid numbers - {e}")
            except Exception as e:
                errors.append(f"Line {i}: Error - {e}")
                print(f"  ERROR: {e}")
        
        conn.commit()
        
        print(f"\nRESULTS:")
        print(f"Successfully updated: {updated_count} records")
        
        if errors:
            print(f"Errors: {len(errors)}")
            for error in errors[:3]:
                print(f"  {error}")
            if len(errors) > 3:
                print(f"  ... and {len(errors) - 3} more errors")
        
        # Show remaining NULL records
        cursor.execute("SELECT COUNT(*) FROM results WHERE start_time IS NULL OR end_time IS NULL")
        remaining = cursor.fetchone()[0]
        print(f"Records still with NULL values: {remaining}")
        
    finally:
        conn.close()

def update_times_from_file_direct(filename):
    """Process time data file directly without user input"""
    db_path = '../notebooks/data_analysis.db'
    
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return False
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return False
    
    # Read data from file
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    if not lines:
        print("No data found in file.")
        return False
    
    print(f"Found {len(lines)} entries in {filename}")
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        updated_count = 0
        errors = []
        
        for i, line in enumerate(lines, 1):
            try:
                parts = line.split()
                if len(parts) < 4:
                    errors.append(f"Line {i}: Not enough values - '{line}'")
                    continue
                
                file_name = ' '.join(parts[:-3])
                data_set = float(parts[-3])
                start_time = float(parts[-2])
                end_time = float(parts[-1])
                
                print(f"Processing: {file_name}, dataset {data_set}, times {start_time}-{end_time}")
                
                # First check if record already has time values
                cursor.execute("SELECT start_time, end_time FROM results WHERE file_name = ? AND (data_set = ? OR data_set = ?)", 
                             (file_name, data_set, float(data_set)))
                existing = cursor.fetchone()
                
                if existing and existing[0] is not None and existing[1] is not None:
                    print(f"  SKIPPED: Record already has time values (start: {int(existing[0])}, end: {int(existing[1])})")
                    continue
                
                # Update the record
                cursor.execute("UPDATE results SET start_time = ?, end_time = ? WHERE file_name = ? AND (data_set = ? OR data_set = ?)", 
                             (start_time, end_time, file_name, data_set, float(data_set)))
                
                if cursor.rowcount > 0:
                    updated_count += 1
                    print(f"  SUCCESS: Updated record")
                else:
                    errors.append(f"Line {i}: No matching record found - '{line}'")
                    print(f"  ERROR: No matching record found")
                    
            except ValueError as e:
                errors.append(f"Line {i}: Invalid numbers - '{line}' ({e})")
                print(f"  ERROR: Invalid numbers - {e}")
            except Exception as e:
                errors.append(f"Line {i}: Error - '{line}' ({e})")
                print(f"  ERROR: {e}")
        
        conn.commit()
        
        print(f"\nRESULTS:")
        print(f"Successfully updated: {updated_count} records")
        
        if errors:
            print(f"Errors: {len(errors)}")
            for error in errors[:5]:
                print(f"  {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
        
        # Show remaining NULL records
        cursor.execute("SELECT COUNT(*) FROM results WHERE start_time IS NULL OR end_time IS NULL")
        remaining = cursor.fetchone()[0]
        print(f"Records still with NULL values: {remaining}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    finally:
        conn.close()

def main():
    """Main interactive function"""
    db_path = '../notebooks/data_analysis.db'
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found: {db_path}")
        return
    
    print("Interactive Database Tool")
    print("=" * 35)
    
    while True:
        print("\nOptions:")
        print("1. Show table structure")
        print("2. Add columns")
        print("3. Input time ranges (interactive - small amounts)")
        print("4. Update time ranges from file (bulk - thousands of entries)")
        print("5. Exit")
        
        try:
            choice = input("\nEnter choice (1-5): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if choice == '1':
            show_table_structure(db_path)
            
        elif choice == '2':
            interactive_add_columns()
                
        elif choice == '3':
            interactive_time_input()
            
        elif choice == '4':
            update_times_from_file()
                
        elif choice == '5':
            print("Goodbye!")
            break
            
        else:
            print("ERROR: Invalid choice")

if __name__ == "__main__":
    # Direct execution for file processing
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "file":
        # Process time_data.txt directly
        print("Processing time_data.txt directly...")
        update_times_from_file_direct("time_data.txt")
    else:
        main()
