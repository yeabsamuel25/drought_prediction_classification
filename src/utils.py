"""
Utility Functions - Helper functions for the project
"""

import os
import matplotlib.pyplot as plt


def create_directories():
    """Create necessary folders if they don't exist"""
    folders = ['data/raw', 'data/processed', 'models', 'results']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("✅ Project directories verified")


def print_header(text):
    """Print a nice header"""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def format_time(seconds):
    """Convert seconds to readable format"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    else:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"