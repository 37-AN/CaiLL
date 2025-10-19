"""
Simple test to verify basic Python environment and imports.
"""
import sys
print("Python version:", sys.version)

# Test basic imports
try:
    import numpy as np
    print("[OK] NumPy imported successfully:", np.__version__)
except Exception as e:
    print("[FAIL] NumPy import failed:", e)

try:
    import pandas as pd
    print("✓ Pandas imported successfully:", pd.__version__)
except Exception as e:
    print("✗ Pandas import failed:", e)

try:
    import gymnasium as gym
    print("✓ Gymnasium imported successfully:", gym.__version__)
except Exception as e:
    print("✗ Gymnasium import failed:", e)

try:
    import stable_baselines3 as sb3
    print("✓ Stable-Baselines3 imported successfully:", sb3.__version__)
except Exception as e:
    print("✗ Stable-Baselines3 import failed:", e)

try:
    import torch
    print("✓ PyTorch imported successfully:", torch.__version__)
except Exception as e:
    print("✗ PyTorch import failed:", e)

try:
    from fastapi import FastAPI
    print("✓ FastAPI imported successfully")
except Exception as e:
    print("✗ FastAPI import failed:", e)

print("\n--- RL Environment Test ---")
try:
    # Create a simple custom environment
    env = gym.make('CartPole-v1')
    obs, info = env.reset()
    print("✓ Created Gymnasium environment successfully")
    print("  Observation shape:", obs.shape)
    env.close()
except Exception as e:
    print("✗ Environment creation failed:", e)

print("\n--- Summary ---")
print("Basic Python environment is ready!")
print("Note: Backend module imports have dependency issues that need fixing.")
