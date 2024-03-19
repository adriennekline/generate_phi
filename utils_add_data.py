import pydicom
from PIL import Image, ImageDraw, ImageFont
import pandas as pd 
import numpy as np
import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import csv

import string

def save_images(output_folder, new_filename):
    """
    Processes a series of images, converting them to grayscale, and saves them to a specified output folder.
    If the output folder doesn't exist, it is created.

    Parameters:
    - image_paths (list): A list of paths to the image files to be processed.
    - output_folder (str): The path to the output folder where processed images will be saved.
    """
    # Check if the output folder exists, and create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    new_file_path = os.path.join(output_folder, new_filename)
    return new_file_path
            

def generate_random_filename(extension='jpeg', length=10):
    """
    Generates a random alphanumeric file name.

    Parameters:
    - extension (str): The file extension for the filename. Default is 'jpeg'.
    - length (int): The length of the random part of the filename. Default is 10.

    Returns:
    - str: A random alphanumeric filename with the specified extension.
    """
    # Define the characters to use for the random part
    chars = string.ascii_letters + string.digits
    # Generate the random part
    random_part = ''.join(random.choice(chars) for _ in range(length))
    # Construct the full filename
    filename = f"{random_part}.{extension}"
    csv_filename = f"{random_part}.csv"
    return filename, csv_filename

def is_overlapping(new_rect, existing_rects):
    """
    Checks if the new rectangle overlaps with any of the existing rectangles.

    Parameters:
    - new_rect (tuple): The new rectangle as (x, y, width, height).
    - existing_rects (list): List of tuples representing existing rectangles.

    Returns:
    - bool: True if there is an overlap, False otherwise.
    """
    new_x, new_y, new_w, new_h = new_rect
    for rect in existing_rects:
        x, y, w, h = rect
        if (new_x < x + w and new_x + new_w > x and
                new_y < y + h and new_y + new_h > y):
            return True
    return False

def add_black_rows(image_array):
    """
    Adds 15% more rows with black pixel values to the top of an image.
    Parameters:
    - image_array (numpy.ndarray) black_rows: The original image array, can be grayscale or RGB.
    Returns:
    - black_rows : The modified black rows associated with the image array.
    """
    # Determine the number of rows to add (20% of the current number of rows)
    rows_to_add = 90
    # Determine if the image is grayscale (2D) or RGB (3D)
    if image_array.ndim == 2:
        # For grayscale, create a 2D array of zeros (black) with the shape (rows_to_add, cols)
        black_array = np.zeros((rows_to_add, image_array.shape[1]))
    elif image_array.ndim == 3:
        # For RGB, create a 3D array of zeros with the shape (rows_to_add, cols, 3)
        black_array = np.zeros((rows_to_add, image_array.shape[1], image_array.shape[2]))
    else:
        raise ValueError("Unsupported image dimensions")
    #return the black rows 
    return black_array

def add_text_to_image(image_medical, texts):
    """
    Adds three non-overlapping texts to a given image array at random locations,
    with checks to avoid overlap.

    Parameters:
    - image_array (numpy.ndarray): The original image array.
    - texts (list): A list of three strings to be added to the image.
    - font_path (str): Path to the font file.
    - font_size (int): Size of the font.

    Returns:
    - numpy.ndarray: The modified image array with texts added.
    """
    print(texts)
    black_array = add_black_rows(image_medical)

    # Set the font type and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    font_color = (255, 255, 255)  # White color
    thickness = 2
    line_type = cv2.LINE_AA
    #print(black_array.shape)
    # Texts to be added
    #texts = ["adrienne kline", "10/18/1992", "4485712345728"]

    # Y positions for the texts
    text_y_positions = [20, 50, 80]  # Adjust these values as needed

    # X positions for text alignment: Left, Center, Right
    text_x_positions = [
        10,  # Left text margin
        black_array.shape[1] // 2,  # Center of the image
        black_array.shape[1] - 10  # Right text margin
    ]

    for i, text in enumerate(texts):
        # Calculate the text size to align the text properly
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        if i == 0:  # Left-aligned text
            text_x = text_x_positions[i]
        elif i == 1:  # Centered text
            text_x = text_x_positions[i] - (text_size[0] // 2)
        elif i == 2:  # Right-aligned text
            text_x = text_x_positions[i] - text_size[0]

        text_y = text_y_positions[i]
    
        # Put the text on the image
        cv2.putText(black_array, text, (text_x, text_y), font, font_scale, font_color, thickness, line_type)

    modified_image_array = np.vstack((black_array, image_medical))
    new_image = modified_image_array.astype(np.uint8)
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(new_image)
    return image

def select_phi_info(csv_data_path):
    """
    select phi information from a csv provided (one from each column is selected )

    Parameters:
    - csv_data_path : the csv file path
    - col_name : the column name from which the phi data is to be selected

    Returns:
    - random_values : listed of the randomly selected text from the column
    """
    random_values = []
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_data_path)
    df = df.astype(str)
    # get column: 
     # Sample one entry randomly from each of the three columns
    for column in ['Patient_Name', 'Patient Healthcare Number', 'BirthDate']:
        random_index = random.randint(0, len(df) - 1)  # Generate a random index
        sampled_entry = df[column].iloc[random_index]  # Use the random index to select an entry
        random_values.append(sampled_entry)
    return random_values

def read_image_get_pixels(file_path):
    # Get the file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    # Read the image data
    if file_extension in ['.png', '.jpeg', '.jpg']:
        # Use Pillow for PNG, JPEG, JPG
        with Image.open(file_path) as img:
            return np.array(img)
    elif file_extension in ['.dcm']:
        
        dicom = pydicom.dcmread(file_path)

        # Access the pixel data within the DICOM file
        pixel_array = dicom.pixel_array
        # Convert the pixel array to uint8 (if necessary)
            # Note: DICOM images can have various bit depths. For simplicity, we're assuming it's 16-bit here.
        if np.max(pixel_array) > 255:
            print("Converting to uint8")
            # Scale the values to be in the range 0-255
            pixel_array = ((pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))) * 255.0
            pixel_array = pixel_array.astype(np.uint8)       
        return pixel_array
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def save_list_as_column_to_csv(column_data, folder_path, filename):
    """
    Save a list of data to a CSV file in a specified folder, with each entry in its own row.

    Parameters:
    - column_data (list): A list containing the data to be saved in a single column.
    - folder_path (str): The path to the folder where the CSV file should be saved.
    - filename (str): The name of the CSV file to be saved.
    """
    # Join the folder path and filename to get the complete file path
    csv_file_path = os.path.join(folder_path, filename)
    
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for entry in column_data:
            writer.writerow([entry])  # Write each entry as its own row
