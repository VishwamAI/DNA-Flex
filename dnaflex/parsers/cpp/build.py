"""Build script for C++ parser components."""

import os
import subprocess
import sys
from pathlib import Path

def build_parsers():
    """Build C++ parser components using CMake."""
    # Get current directory
    curr_dir = Path(__file__).parent.absolute()
    
    # Create build directory
    build_dir = curr_dir / 'build'
    build_dir.mkdir(exist_ok=True)
    
    # Configure CMake
    cmake_configure = [
        'cmake',
        '..',
        f'-DPYTHON_EXECUTABLE={sys.executable}',
        '-DCMAKE_BUILD_TYPE=Release'
    ]
    
    # Build
    cmake_build = ['cmake', '--build', '.', '--config', 'Release', '-j4']
    
    try:
        # Run CMake configure
        print("Configuring CMake...")
        subprocess.run(cmake_configure, cwd=build_dir, check=True)
        
        # Run CMake build
        print("Building parser components...")
        subprocess.run(cmake_build, cwd=build_dir, check=True)
        
        print("Build completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    build_parsers()