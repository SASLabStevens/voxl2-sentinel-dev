import pandas as pd
import numpy as np
import argparse

def convert_angles_to_radians(csv_path, output_csv_path):
    """
    Convert heading and tilt angles from degrees to radians and save the results in a new CSV file.
    
    Parameters:
    - csv_path: str, path to the input CSV file.
    - output_csv_path: str, path to save the output CSV file.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Convert degrees to radians for heading and tilt angles
    df['heading_rad'] = np.deg2rad(df['heading'])
    df['tilt_rad'] = np.deg2rad(df['tilt'])

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"File saved successfully at {output_csv_path}")

if __name__ == "__main__":
    # Setting up Argument Parser
    parser = argparse.ArgumentParser(description="Convert heading and tilt angles from degrees to radians in a CSV file.")
    parser.add_argument("--csv_path", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_csv_path", required=True, help="Path to save the output CSV file.")
    
    # Parse arguments
    args = parser.parse_args()

    # Call the function with the provided arguments
    convert_angles_to_radians(args.csv_path, args.output_csv_path)
