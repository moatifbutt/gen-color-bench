"""Entry point for running generation as a module."""
import os
os.environ["HF_HOME"] = "/data/144-1/users/mabutt/gencolorbench_v4/cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/144-1/users/mabutt/gencolorbench_v4/cache"
os.environ["HF_DATASETS_CACHE"] = "/data/144-1/users/mabutt/gencolorbench_v4/cache"

from .runner import main

if __name__ == "__main__":
    main()
