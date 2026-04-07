"""
GenColorBench CLI entry point.

Usage:
    python -m gencolorbench mini --output-dir ./prompts
    python -m gencolorbench full --output-dir ./prompts
    python -m gencolorbench images --model flux-dev --prompts-dir ./prompts
    python -m gencolorbench evaluate --prompts-dir ./prompts --images-dir ./images
"""

from .cli import main

if __name__ == '__main__':
    main()
