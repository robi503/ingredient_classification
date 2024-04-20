import os

def rename_images(directory):
    # Get the list of files in the directory
    files = os.listdir(directory)
    
    # Get the directory name
    dirname = os.path.basename(directory)
    
    # Iterate through each file in the directory
    for i, filename in enumerate(files):
        # Split the filename and extension
        _, ext = os.path.splitext(filename)
        
        # Construct the new filename
        new_filename = f"{dirname}_{i+1}{ext}"
        
        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

# Specify the directory containing the fruit/vegetable folders
base_directory = "D:\\Facultate\\ELL\\food_recognition\\ingredient_identification\\mixed_ingredients"

# # Iterate through each directory in the base directory
# for subdir in os.listdir(base_directory):
#     # Construct the full path of the subdirectory
#     subdirectory = os.path.join(base_directory, subdir)
    
#     # Check if the subdirectory is actually a directory
#     if os.path.isdir(subdirectory):
#         # Rename the images in the subdirectory
#         rename_images(subdirectory)

rename_images(base_directory)

print("Image renaming complete.")