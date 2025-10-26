"""
Project Setup and Validation Script
Run this before starting development to ensure everything is set up correctly
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_success(text):
    print(f"✅ {text}")

def print_error(text):
    print(f"❌ {text}")

def print_warning(text):
    print(f"⚠️  {text}")

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print_success("Python version is compatible")
        return True
    else:
        print_error("Python 3.8 or higher is required")
        return False

def create_directory_structure():
    """Create all required directories"""
    print_header("Creating Directory Structure")
    
    directories = [
        'app',
        'notebooks',
        'data',
        'models',
        'assets'
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print_success(f"Created directory: {directory}/")
        else:
            print(f"   Directory already exists: {directory}/")
    
    return True

def check_required_files():
    """Check if required files exist"""
    print_header("Checking Required Files")
    
    required_files = {
        'data/air_quality_global.csv': 'Dataset file',
        'data/metadata.json': 'Metadata file',
        'app/app.py': 'Main application file',
        'README.md': 'Documentation file',
        'requirements.txt': 'Dependencies file'
    }
    
    all_present = True
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print_success(f"{description}: {file_path}")
        else:
            print_warning(f"{description} NOT FOUND: {file_path}")
            all_present = False
    
    return all_present

def validate_dataset():
    """Validate dataset structure"""
    print_header("Validating Dataset")
    
    try:
        import pandas as pd
        
        if not Path('data/air_quality_global.csv').exists():
            print_error("Dataset file not found!")
            return False
        
        df = pd.read_csv('data/air_quality_global.csv')
        
        required_columns = [
            'city', 'country', 'latitude', 'longitude', 
            'year', 'month', 'pm25_ugm3', 'no2_ugm3',
            'data_quality', 'measurement_method', 'data_source'
        ]
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns found: {list(df.columns)}")
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            print_error(f"Missing columns: {missing_cols}")
            return False
        else:
            print_success("All required columns present")
        
        # Check for missing values
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        print("\nMissing values percentage:")
        for col, pct in missing_pct[missing_pct > 0].items():
            print(f"   {col}: {pct}%")
        
        print_success("Dataset validation complete")
        return True
        
    except ImportError:
        print_warning("pandas not installed - skipping dataset validation")
        return True
    except Exception as e:
        print_error(f"Error validating dataset: {e}")
        return False

def validate_metadata():
    """Validate metadata.json structure"""
    print_header("Validating Metadata")
    
    try:
        if not Path('data/metadata.json').exists():
            print_error("Metadata file not found!")
            return False
        
        with open('data/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        required_keys = [
            'dataset_name', 'version', 'creation_date', 
            'total_records', 'data_sources', 'file_components',
            'data_quality_notes', 'usage_recommendations'
        ]
        
        missing_keys = [key for key in required_keys if key not in metadata]
        
        if missing_keys:
            print_warning(f"Recommended metadata keys missing: {missing_keys}")
        else:
            print_success("All recommended metadata keys present")
        
        print("\nMetadata summary:")
        for key in ['dataset_name', 'version', 'creation_date', 'total_records']:
            if key in metadata:
                print(f"   {key}: {metadata[key]}")
        
        print_success("Metadata validation complete")
        return True
        
    except json.JSONDecodeError:
        print_error("Invalid JSON format in metadata.json")
        return False
    except Exception as e:
        print_error(f"Error validating metadata: {e}")
        return False

def check_dependencies():
    """Check if dependencies are installed"""
    print_header("Checking Dependencies")
    
    if not Path('requirements.txt').exists():
        print_warning("requirements.txt not found")
        return False
    
    try:
        required_packages = [
            'streamlit',
            'pandas',
            'numpy',
            'scikit-learn',
            'plotly',
            'matplotlib',
            'seaborn'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print_success(f"{package} installed")
            except ImportError:
                missing_packages.append(package)
                print_warning(f"{package} NOT installed")
        
        if missing_packages:
            print("\nTo install missing packages, run:")
            print(f"   pip install {' '.join(missing_packages)}")
            print("\nOr install all dependencies:")
            print("   pip install -r requirements.txt")
            return False
        else:
            print_success("All dependencies installed")
            return True
            
    except Exception as e:
        print_error(f"Error checking dependencies: {e}")
        return False

def create_sample_metadata():
    """Create a sample metadata.json if it doesn't exist"""
    print_header("Creating Sample Metadata")
    
    if Path('data/metadata.json').exists():
        print("   Metadata file already exists, skipping...")
        return True
    
    sample_metadata = {
        "dataset_name": "Urban Air Quality and Climate Dataset",
        "version": "2.0",
        "creation_date": "2025-01-15",
        "total_records": 150000,
        "license": "CC0 1.0 Universal",
        "file_components": {
            "filename": "air_quality_global.csv",
            "columns": [
                "city", "country", "latitude", "longitude",
                "year", "month", "pm25_ugm3", "no2_ugm3",
                "data_quality", "measurement_method", "data_source"
            ]
        },
        "data_sources": [
            "WHO Air Quality Database",
            "OpenAQ",
            "National Environmental Agencies"
        ],
        "data_quality_notes": "Some records may have missing values. Filter by data_quality field for reliable measurements.",
        "usage_recommendations": "Use for trend analysis and predictive modeling. Consider temporal and geographic factors."
    }
    
    try:
        with open('data/metadata.json', 'w') as f:
            json.dump(sample_metadata, f, indent=2)
        print_success("Sample metadata.json created")
        print_warning("Please update with your actual dataset information!")
        return True
    except Exception as e:
        print_error(f"Failed to create metadata: {e}")
        return False

def generate_project_report():
    """Generate a summary report of the project structure"""
    print_header("Project Structure Report")
    
    print("\n📁 Project Directory Tree:")
    print(".")
    print("├── app/")
    print("│   └── app.py")
    print("├── notebooks/")
    print("│   └── eda.ipynb")
    print("├── data/")
    print("│   ├── air_quality_global.csv")
    print("│   └── metadata.json")
    print("├── models/")
    print("│   └── [trained models will be saved here]")
    print("├── assets/")
    print("│   └── [screenshots will be saved here]")
    print("├── README.md")
    print("├── requirements.txt")
    print("└── .gitignore")
    
    print("\n📝 Next Steps:")
    print("1. Ensure your dataset (air_quality_global.csv) is in the data/ folder")
    print("2. Update metadata.json with actual dataset information")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Run the application: streamlit run app/app.py")
    print("5. Complete the EDA notebook: jupyter notebook notebooks/eda.ipynb")
    print("6. Take screenshots of the app UI and save to assets/")
    print("7. Update README.md with your name and findings")
    print("8. Create ZIP file: <YourName><RollNumber>.zip")

def main():
    """Main setup and validation function"""
    print("\n" + "=" * 70)
    print("  🚀 AIR QUALITY PROJECT SETUP & VALIDATION")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Directory Structure", create_directory_structure),
        ("Required Files", check_required_files),
        ("Dependencies", check_dependencies),
    ]
    
    # Run basic checks
    results = []
    for check_name, check_func in checks:
        result = check_func()
        results.append((check_name, result))
    
    # Try to validate dataset and metadata if files exist
    if Path('data/air_quality_global.csv').exists():
        results.append(("Dataset Validation", validate_dataset()))
    else:
        print_warning("\nDataset not found - place air_quality_global.csv in data/ folder")
    
    if Path('data/metadata.json').exists():
        results.append(("Metadata Validation", validate_metadata()))
    else:
        create_sample_metadata()
    
    # Generate final report
    generate_project_report()
    
    # Summary
    print_header("Setup Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print(f"\n{passed}/{total} checks passed")
    
    if passed == total:
        print_success("\n🎉 All checks passed! You're ready to start development!")
    else:
        print_warning("\n⚠️  Some checks failed. Please address the issues above.")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
