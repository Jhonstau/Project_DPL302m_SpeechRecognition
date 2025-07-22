import os

def clean_up(temp_files:list[str]) -> None:
    """
    Clean up temporary files created during audio processing.
    
    Args:
        temp_files (list[str]): List of paths to temporary files to be deleted.
    """
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        else:
            print(f"Warning: {temp_file} does not exist and cannot be deleted.")


