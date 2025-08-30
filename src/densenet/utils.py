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
        return None, False

    login_successful = False
    # Priority 1: Environment Variable (most common method for local/CI/CD)
    if os.environ.get("WANDB_API_KEY"):
        print("Attempting to log in to W&B using WANDB_API_KEY environment variable...")
        if wandb.login():
            login_successful = True
            print("Successfully logged into W&B.")
    
    # Priority 2: Fallback to Kaggle Secrets
    if not login_successful:
        try:
            from kaggle_secrets import UserSecretsClient
            print("W&B API key not found or failed in environment. Trying Kaggle Secrets...")
            user_secrets = UserSecretsClient()
            wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
            if wandb.login(key=wandb_api_key):
                login_successful = True
                print("Successfully logged into W&B using Kaggle Secrets.")
        except ImportError:
            # This is expected if not on Kaggle.
            print("Kaggle secrets library not found. Skipping fallback.")
        except Exception as e:
            # This is an error if Kaggle secrets are expected but fail.
            print(f"An error occurred while trying to use Kaggle Secrets: {e}")
            print("Please ensure the secret 'WANDB_API_KEY' is correctly named and attached to this notebook.")

    if not login_successful:
        print("\nCould not log in to W&B. Training will continue without tracking.")
        print("To enable tracking, please set the WANDB_API_KEY secret or environment variable.")
        return None, False
    
    # Define a dynamic run name based on available arguments
    run_name = f"densenet_k{args.growth_rate}"
    if 'model_arch' in args:
        run_name = args.model_arch
        
    # Initialize the W&B Run
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=vars(args),
        id=args.resume_id,
        resume="allow"
    )
    print(f"W&B run initialized. ID: {run.id}")
    return run, True