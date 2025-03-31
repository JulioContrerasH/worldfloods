from huggingface_hub import HfApi, login, snapshot_download
import pathlib

# Authentication token for Hugging Face
HF_TOKEN = ""

# Login to Hugging Face

login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)

# Repository details
repo_id = "tacofoundation/worldfloods" 
repo_type = "dataset"
local_dir = "/home/contreras/Documents/GitHub/sen2venus" 

# Clone the repository
snapshot_download(repo_id=repo_id, repo_type=repo_type, token=HF_TOKEN, local_dir=local_dir)

# List of files to delete from the repository
files_to_delete = [
    "assets/README.md"
]

def delete_files(files: list[str]) -> None:
    """
    Delete specified files from the repository.
    
    Args:
        files: List of file paths to delete from the repository.
    """
    for file in files:
        api.delete_file(
            path_in_repo=file,  # File name in the repository
            repo_id=repo_id,
            repo_type=repo_type,
            token=HF_TOKEN,
            commit_message="Delete: obsolete file",
        )
        print(f"file deleted: {file}")

# Delete specified files
delete_files(files_to_delete)

# Delete all files in a specified folder
folder_to_delete = "images/"
files_in_repo = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
files_in_folder = [file for file in files_in_repo if file.startswith(folder_to_delete)]

def delete_folder_files(files: list[str]) -> None:
    """
    Delete all files in a specific folder within the repository.
    
    Args:
        files: List of file paths within the folder to delete.
    """
    for file in files:
        api.delete_file(
            path_in_repo=file,
            repo_id=repo_id,
            repo_type=repo_type,
            token=HF_TOKEN,
            commit_message=f"Delete: files in dir {folder_to_delete}"
        )
        print(f"file deleted: {file}")

# Delete files in the 'images' folder
delete_folder_files(files_in_folder)

# Upload a complete folder to the repository
local_folder = "assets"

def upload_folder(local_folder: str) -> None:
    """
    Upload an entire folder to the specified folder in the repository.
    
    Args:
        local_folder: Path to the local folder to be uploaded.
    """
    api.upload_folder(
        folder_path=local_folder,  # Local folder path
        repo_id=repo_id,           # Repository ID
        repo_type=repo_type,       # Repository type (dataset, model, etc.)
        path_in_repo="assets",     # Upload files into the "assets" folder
        token=HF_TOKEN,            # Authentication token
        commit_message="Upload: Complete folder into assets directory"  # Commit message
    )
    print("Folder uploaded successfully into the 'assets' directory.")

# Upload the folder
upload_folder(local_folder)

# List of individual files to upload
files_to_upload = [
    # "/data/databases/legacy/OLD_SEN2NEON/tacos/sen2neon.0000.part.taco",
    "/data/databases/legacy/OLD_SEN2NEON/tacos/sen2neon.0001.part.taco",
    "/data/databases/legacy/OLD_SEN2NEON/tacos/sen2neon.0002.part.taco",
    "/data/databases/legacy/OLD_SEN2NEON/tacos/sen2neon.0003.part.taco",
    "/data/databases/legacy/OLD_SEN2NEON/tacos/sen2neon.0004.part.taco"

]

dir = pathlib.Path("/data/databases/Julio/Flood_kike/tacos/")
files_to_upload = [str(file) for file in dir.glob("*.taco")]


def upload_files(files: list[str]) -> None:
    """
    Upload specified files to the repository.
    
    Args:
        files: List of file paths to upload to the repository.
    """
    for file in files:
        
        api.upload_file(
            path_or_fileobj=file,
            repo_id=repo_id,
            path_in_repo=file.split("/")[-1], #f"./{file}"
            repo_type=repo_type,
            token=HF_TOKEN,
            commit_message="Upload"
        )
        print(f"File uploaded: {file}")

# Upload the specified files
upload_files(files_to_upload)