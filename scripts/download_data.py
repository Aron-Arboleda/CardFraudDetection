"""
Credit Card Fraud Detection - Data Download Script
Downloads the Credit Card Fraud Detection dataset from Kaggle using Kaggle API.

Prerequisites:
1. Install kaggle: pip install kaggle
2. Set up Kaggle API credentials using environment variables:
   
   Windows (Command Prompt):
   set KAGGLE_USERNAME=your_kaggle_username
   set KAGGLE_KEY=your_api_token
   
   Windows (PowerShell):
   $env:KAGGLE_USERNAME="your_kaggle_username"
   $env:KAGGLE_KEY="your_api_token"
   
   Linux/Mac (Terminal):
   export KAGGLE_USERNAME=your_kaggle_username
   export KAGGLE_KEY=your_api_token
   
   OR create a .env file in the project root with:
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_api_token

Usage:
    python scripts/download_data.py
"""

import os
import sys
from pathlib import Path

def setup_kaggle_credentials():
    """Set up Kaggle API credentials from environment variables or prompt user."""
    
    # Check if credentials are already set in environment
    kaggle_username = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')
    
    if kaggle_username and kaggle_key:
        print("✓ Kaggle credentials found in environment variables")
        return True
    
    # Check for .env file (optional)
    env_file = Path('.env')
    if env_file.exists():
        print("✓ Found .env file, loading credentials...")
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
            
            kaggle_username = os.environ.get('KAGGLE_USERNAME')
            kaggle_key = os.environ.get('KAGGLE_KEY')
            
            if kaggle_username and kaggle_key:
                print("✓ Loaded credentials from .env file")
                return True
        except Exception as e:
            print(f"⚠ Warning: Could not read .env file: {e}")
    
    # If not found, provide instructions
    print("✗ Kaggle API credentials not found")
    print("\n" + "="*60)
    print("SETUP INSTRUCTIONS:")
    print("="*60)
    print("\nOption 1 - Set Environment Variables (Recommended):")
    print("\nWindows (Command Prompt):")
    print("  set KAGGLE_USERNAME=your_kaggle_username")
    print("  set KAGGLE_KEY=your_api_token")
    print("\nWindows (PowerShell):")
    print("  $env:KAGGLE_USERNAME=\"your_kaggle_username\"")
    print("  $env:KAGGLE_KEY=\"your_api_token\"")
    print("\nLinux/Mac (Terminal):")
    print("  export KAGGLE_USERNAME=your_kaggle_username")
    print("  export KAGGLE_KEY=your_api_token")
    print("\nOption 2 - Create a .env file in project root:")
    print("  KAGGLE_USERNAME=your_kaggle_username")
    print("  KAGGLE_KEY=your_api_token")
    print("\nThen run this script again.")
    print("="*60)
    
    return False

def check_kaggle_api():
    """Check if Kaggle API library is installed."""
    try:
        import kaggle
        print("✓ Kaggle API library found")
        return True
    except ImportError:
        print("✗ Kaggle library not found. Install with: pip install kaggle")
        return False

def create_directory_structure():
    """Create necessary directories for the project."""
    directories = [
        'data',
        'notebooks',
        'scripts',
        'models',
        'results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified directory: {directory}/")

def download_dataset():
    """Download the Credit Card Fraud Detection dataset from Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize Kaggle API (will use environment variables)
        api = KaggleApi()
        api.authenticate()
        print("✓ Kaggle API authenticated successfully")
        
        # Dataset details
        dataset_name = "mlg-ulb/creditcardfraud"
        download_path = "data/"
        
        print(f"\n📥 Downloading dataset: {dataset_name}")
        print("This may take a few minutes (dataset is ~150 MB)...")
        
        # Download dataset
        api.dataset_download_files(
            dataset_name,
            path=download_path,
            unzip=True,
            quiet=False
        )
        
        print(f"✓ Dataset downloaded successfully to {download_path}")
        
        # Verify the file exists
        csv_path = os.path.join(download_path, "creditcard.csv")
        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path) / (1024 * 1024)  # Convert to MB
            print(f"✓ Verified: creditcard.csv ({file_size:.2f} MB)")
            return True
        else:
            print("✗ Error: creditcard.csv not found after download")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("- Verify your Kaggle username and API token are correct")
        print("- Check internet connection")
        print("- Ensure you've accepted the dataset terms at:")
        print("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        return False

def main():
    """Main execution function."""
    print("=" * 60)
    print("Credit Card Fraud Detection - Data Download")
    print("=" * 60)
    print()
    
    # Step 1: Check Kaggle API library
    print("Step 1: Checking Kaggle API library...")
    if not check_kaggle_api():
        sys.exit(1)
    print()
    
    # Step 2: Check Kaggle credentials
    print("Step 2: Checking Kaggle API credentials...")
    if not setup_kaggle_credentials():
        sys.exit(1)
    print()
    
    # Step 3: Create directory structure
    print("Step 3: Setting up project directories...")
    create_directory_structure()
    print()
    
    # Step 4: Download dataset
    print("Step 4: Downloading dataset from Kaggle...")
    if download_dataset():
        print()
        print("=" * 60)
        print("✅ SUCCESS! Dataset ready for analysis")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run data exploration: jupyter notebook notebooks/data_exploration.ipynb")
        print("2. Run preprocessing: python scripts/preprocess.py")
        print("3. Train model: Upload notebooks/model_training.ipynb to Google Colab")
    else:
        print()
        print("=" * 60)
        print("❌ FAILED to download dataset")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()