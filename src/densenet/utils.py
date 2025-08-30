"""
Utility functions for data preparation and other tasks.
"""
import os
import shutil
import pandas as pd
from tqdm import tqdm
import wandb


def prepare_imagenet_validation_set(input_dir: str, output_dir: str) -> str:
    """
    Prepares the ImageNet validation set from the Kaggle directory structure.

    It reads the solution CSV file, creates subdirectories for each class,
    and moves the validation images into their respective class folders.

    Args:
        input_dir (str): The root path to the Kaggle input data 
                         (e.g., '/kaggle/input/imagenet-object-localization-challenge').
        output_dir (str): The path to a writeable output directory 
                          (e.g., '/kaggle/working/').

    Returns:
        str: The path to the newly created sorted validation directory.
    """
    print("--- Preparing ImageNet validation set ---")
    
    val_images_path = os.path.join(input_dir, 'ILSVRC/Data/CLS-LOC/val')
    solution_file_path = os.path.join(input_dir, 'LOC_val_solution.csv')
    
    sorted_val_dir = os.path.join(output_dir, 'val_sorted')
    os.makedirs(sorted_val_dir, exist_ok=True)
    
    print(f"Reading solution file from: {solution_file_path}")
    df = pd.read_csv(solution_file_path)
    df['class_id'] = df['PredictionString'].apply(lambda x: x.split(' ')[0])
    
    print(f"Found {len(df)} images to sort.")
    print(f"Copying and sorting images from {val_images_path} to {sorted_val_dir}...")

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_id = row['ImageId']
        class_id = row['class_id']
        
        dest_class_dir = os.path.join(sorted_val_dir, class_id)
        os.makedirs(dest_class_dir, exist_ok=True)
        
        src_path = os.path.join(val_images_path, f'{image_id}.JPEG')
        dest_path = os.path.join(dest_class_dir, f'{image_id}.JPEG')
        
        shutil.copyfile(src_path, dest_path)
        
    print("\n--- Validation set preparation complete! ---")
    print(f"Sorted validation data is ready in: {sorted_val_dir}")
    return sorted_val_dir

def setup_wandb(args, project_name):
    """
    Sets up Weights & Biases for experiment tracking with a universal login approach.
    It prioritizes environment variables, then falls back to Kaggle secrets.
    """
    if args.no_wandb:
        print("W&B logging is disabled by the user.")
        return False

    # Priority 1: Environment Variable (most common method for local/CI/CD)
    if os.environ.get("WANDB_API_KEY"):
        print("Attempting to log in to W&B using environment variable...")
        if wandb.login():
            print("Successfully logged into W&B.")
        else:
            print("W&B login failed despite finding an API key. Disabling tracking.")
            return False
    else:
        # Priority 2: Fallback to Kaggle Secrets
        try:
            from kaggle_secrets import UserSecretsClient
            print("W&B API key not found in environment. Trying Kaggle Secrets...")
            user_secrets = UserSecretsClient()
            wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
            if wandb.login(key=wandb_api_key):
                print("Successfully logged into W&B using Kaggle Secrets.")
            else:
                 print("W&B login failed using Kaggle Secrets. Disabling tracking.")
                 return False
        except Exception:
            print("Could not log in to W&B. Ensure WANDB_API_KEY is set or you are in a configured environment.")
            return False


    # Initialize the W&B Run with resume capabilities
    run = wandb.init(
        project=project_name,
        name=f"densenet_k{args.growth_rate}",
        config=vars(args),
        id=args.resume_id,  # Pass the run_id to resume
        resume="allow"      # Allow resuming the run
    )
    print(f"W&B run 'densenet_k{args.growth_rate}' initialized in project '{project_name}' | Run ID: {run.id}")
    return run, True