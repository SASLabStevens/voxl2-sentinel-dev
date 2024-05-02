import os
import cv2
import json
import argparse
import pandas as pd
from datetime import datetime

def get_frame_from_video(video_path, video_time_sec):
    """
    Extract a frame from the video at a specific time given in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(video_time_sec * fps)  # Calculate the frame number using video_time_sec directly
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Set the video position to the frame number
    success, frame = cap.read()
    cap.release()
    return success, frame


def save_frame_and_data(frame, data, directory_timestamp, index, output_folder):
    """
    Save the frame as a PNG image and data as a JSON file with the specified structure in the output folder.
    """
    # Use datetime.fromisoformat for ISO 8601 formatted strings
    directory_timestamp = datetime.fromisoformat(directory_timestamp.replace('Z', '+00:00'))
    formatted_directory_timestamp = directory_timestamp.strftime('%Y-%m-%d-%H-%M-%S')

    new_base_name = f"siteRTX0002-camA003-{formatted_directory_timestamp}-{str(index).zfill(8)}"
    image_name = f"{new_base_name}.png"
    json_name = f"{new_base_name}.json"
    image_path = os.path.join(output_folder, image_name)
    json_path = os.path.join(output_folder, json_name)
    cv2.imwrite(image_path, frame)  # Save the frame
    
    json_data = {
        "version": "4.2.0",
        "fname": image_name,
        "site": "SIT_campus",
        "source": "camA003",
        "collection": formatted_directory_timestamp,
        "timestamp": directory_timestamp.timestamp(),  # Use timestamp from the parsed datetime object
        "extrinsics": {
            "lat": data['Latitude'],
            "lon": data['Longitude'],
            "alt": data['Altitude'],
            "omega": data["heading"], 
            "phi": data["tilt"],
            "kappa": 0,
        },
        "intrinsics": {
            "fx": 2376.5625,
            "fy": 2376.5625,
            "cx": 2027.5000,
            "cy": 1519.5000,
            "k1": 0.13000,
            "k2": -0.24000,
            "k3": 0.10400,
            "p1": 0,
            "p2": 0,
            "p3": 0,
            "p4": 0,
            "rows": 4056,
            "columns": 3040
        },
        "projection": "fisheye_pix4d",
        "calibration": "none",
        "geolocation": "gps",
        "pii_status": "n/a",
        "env_conditions": ["clear", "sunny"],
        "type": "airborne",
        "modes": [],
        "exterior": True,
        "interior": False,
        "transient_occlusions": [],
        "artifacts": [],
        "masks": {"key": {"rle": "", "method": ""}},
        "pii_detected": [],
        "pii_removed": []
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

def process_video_and_csv(video_path, csv_path, output_folder):
    """
    Process the video and CSV to extract frames and save data.
    """
    df = pd.read_csv(csv_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for index, row in enumerate(df.iterrows(), start=1):
        directory_timestamp = row[1]['datetime']
        video_time_sec = row[1]['Video Time (Sec)']
        success, frame = get_frame_from_video(video_path, video_time_sec)
        if success:
            save_frame_and_data(frame, row[1].to_dict(), directory_timestamp, index, output_folder)
        else:
            print(f"Failed to extract frame at video time: {video_time_sec} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video based on CSV video time.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--csv", required=True, help="Path to the CSV file.")
    parser.add_argument("--start_datetime", required=False, help="Start datetime of the video (YYYY-MM-DD HH:MM:SS).")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder where images and JSON files will be saved.")
    args = parser.parse_args()
    process_video_and_csv(args.video, args.csv, args.output_folder)
