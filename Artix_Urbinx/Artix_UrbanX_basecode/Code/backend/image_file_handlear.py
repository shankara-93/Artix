import os
import re


from datetime import datetime
from typing import List, Dict, Tuple 
from collections import OrderedDict


def extract_datetime_from_filename(file_name: str) -> datetime:
    """
    Extracts the date and time from a given filename in the format 'YYYYMMDD_HHMMSS'.
    
    Args:
        file_name (str): The filename to extract the datetime from.
        
    Returns:
        datetime: A datetime object representing the extracted date and time.
    
    Raises:
        ValueError: If no valid date and time format is found in the filename.
    """
    # Search for the pattern 'YYYYMMDD_HHMMSS' in the filename
    match = re.search(r"(\d{8})_(\d{6})", file_name)
    
    if match:
        # Construct the datetime string from the captured groups
        datetime_str = f"{match.group(1)}_{match.group(2)}"
        
        # Attempt to parse the datetime string
        try:
            return datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
        except ValueError as e:
            raise ValueError(f"Error parsing datetime: {e}")
    else:
        raise ValueError("No valid date and time found in the filename")

# def extract_dates_and_filenames_from_folder(folder_path: str, file_type: str = '.jpg') -> Tuple[List[Dict[str, str]], Dict[str, str]]:
#     """
#     Extracts date and time information from filenames in a specified folder and returns:
#     1. A list of dictionaries in the format {"id": int, "content": "YYYY-MM-DD_HHMMSS", "start": "YYYY-MM-DD"}.
#     2. A dictionary where the key is the date and time in 'YYYY-MM-DD_HHMMSS' format and the value is the corresponding file path.
    
#     Args:
#         folder_path (str): The path to the folder containing files.
#         file_type (str): The file extension to filter by (e.g., '.jpg', '.tif'). Default is '.jpg'.
        
#     Returns:
#         Tuple[List[Dict[str, str]], Dict[str, str]]: 
#             - A list of dictionaries containing extracted date and time information.
#             - A dictionary containing the extracted date and time as key and the file path as value.
    
#     Raises:
#         FileNotFoundError: If the folder path does not exist.
#     """
#     if not os.path.exists(folder_path):
#         raise FileNotFoundError(f"The folder path '{folder_path}' does not exist.")
    
#     date_list = []
#     date_file_dict = {}

#     # Get all files with the specified file type from the folder
#     files = [f for f in os.listdir(folder_path) if f.endswith(file_type) and os.path.isfile(os.path.join(folder_path, f))]
    
#     for idx, file_name in enumerate(files, start=1):
#         file_path = os.path.join(folder_path, file_name)  # Full file path
        
#         try:
#             # Extract datetime from filename
#             date_time_obj = extract_datetime_from_filename(file_name)
#             date_str = date_time_obj.strftime("%Y-%m-%d_%H%M%S")  # Full date and time
#             data_str_trim = date_time_obj.strftime("%Y-%m-%d")  # Date only
            
#             # Append the result to the list in the required format
#             date_list.append({
#                 "id": idx,
#                 "content": date_str,
#                 "start": data_str_trim
#             })
            
#             # Use the full date and time string as the key and the full file path as the value
#             date_file_dict[date_str] = file_path
#         except ValueError:
#             # Skip files without valid date patterns
#             continue
    
#     return date_list, date_file_dict

def extract_dates_and_filenames_from_folder(folder_path: str, file_type: str = '.jpg') -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    """
    Extracts date and time information from filenames in a specified folder and returns:
    1. A list of dictionaries in the format {"id": int, "content": "YYYY-MM-DD_HHMMSS", "start": "YYYY-MM-DD"}.
    2. A dictionary where the key is the date and time in 'YYYY-MM-DD_HHMMSS' format and the value is the corresponding file path.
    
    Args:
        folder_path (str): The path to the folder containing files.
        file_type (str): The file extension to filter by (e.g., '.jpg', '.tif'). Default is '.jpg'.
        
    Returns:
        Tuple[List[Dict[str, str]], Dict[str, str]]: 
            - A list of dictionaries containing extracted date and time information.
            - A dictionary containing the extracted date and time as key and the file path as value.
    
    Raises:
        FileNotFoundError: If the folder path does not exist.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder path '{folder_path}' does not exist.")
    
    date_list = []
    date_file_dict = {}

    # Get all files with the specified file type from the folder
    files = [f for f in os.listdir(folder_path) if f.endswith(file_type) and os.path.isfile(os.path.join(folder_path, f))]
    
    for idx, file_name in enumerate(files, start=1):
        file_path = os.path.join(folder_path, file_name)  # Full file path
        
        try:
            # Extract datetime from filename
            date_time_obj = extract_datetime_from_filename(file_name)
            date_str = date_time_obj.strftime("%Y-%m-%d_%H%M%S")  # Full date and time
            data_str_trim = date_time_obj.strftime("%Y-%m-%d")  # Date only
            
            # Append the result to the list in the required format
            date_list.append({
                "id": idx,
                "content": date_str,
                "start": data_str_trim
            })
            
            # Use the full date and time string as the key and the full file path as the value
            date_file_dict[date_str] = file_path
        except ValueError:
            # Skip files without valid date patterns
            continue

    # Sort date_list by 'content' (which contains the full datetime string)
    date_list.sort(key=lambda x: x["content"])

    # Sort date_file_dict by date_str (which is the key in the dictionary)
    sorted_date_file_dict = OrderedDict(sorted(date_file_dict.items()))

    return date_list, sorted_date_file_dict