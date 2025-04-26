import os

# Input directory containing the TXT files
input_directory = "/home/varun/Documents/MEGA/tifr/activities/06 Doppler Data: Sagar/reference"
output_directory = os.path.join(input_directory, "edited")

# Create the 'edited' subfolder if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def process_file(file_path, output_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the start and end markers for the spectral data
    start_marker = ">>>>>Begin Processed Spectral Data<<<<<"
    end_marker = ">>>>>End Processed Spectral Data<<<<<"

    start_index = None
    end_index = None

    # Identify the lines where the spectral data begins and ends
    for i, line in enumerate(lines):
        if start_marker in line:
            start_index = i + 1
        if end_marker in line:
            end_index = i
            break

    # Extract the spectral data lines
    if start_index is not None and end_index is not None:
        spectral_data = lines[start_index:end_index]
        with open(output_path, 'w') as out_file:
            out_file.writelines(spectral_data)

# Iterate over all the files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith(".txt"):  # Process only TXT files
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(output_directory, filename)
        
        # Process and save the edited file
        process_file(input_file_path, output_file_path)

print("Processing complete. Edited files are saved in the 'edited' subfolder.")

