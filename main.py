#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LottoProphet - 彩票预测应用程序主入口

A comprehensive lottery prediction application using machine learning and deep learning techniques.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Import after path setup
from utils.logging import setup_logging, get_logger
from utils.config import config

def main():
    """
    Main entry point for the LottoProphet application
    """
    # Setup logging
    setup_logging()
    logger = get_logger('main')
    
    logger.info("Starting LottoProphet application...")
    
    try:
        # Import UI components
        from ui.app import main as ui_main
        
        # Start the application
        ui_main()
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure all dependencies are installed.")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
        
    logger.info("LottoProphet application finished.")


if __name__ == '__main__':
    main()