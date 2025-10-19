# Simple test to verify basic Python environment and imports
import sys
print("Python version:", sys.version)
print("")

# Test basic imports
try:
    import numpy as np
    print("[OK] NumPy:", np.__version__)
except Exception as e:
    print("[FAIL] NumPy:", e)

try:
    import pandas as pd
    print("[OK] Pandas:", pd.__version__)
except Exception as e:
    print("[FAIL] Pandas:", e)

try:
    import gymnasium as gym
    print("[OK] Gymnasium:", gym.__version__)
except Exception as e:
    print("[FAIL] Gymnasium:", e)

try:
    import stable_baselines3 as sb3
    print("[OK] Stable-Baselines3:", sb3.__version__)
except Exception as e:
    print("[FAIL] Stable-Baselines3:", e)

try:
    import torch
    print("[OK] PyTorch:", torch.__version__)
except Exception as e:
    print("[FAIL] PyTorch:", e)

try:
    from fastapi import FastAPI
    print("[OK] FastAPI")
except Exception as e:
    print("[FAIL] FastAPI:", e)

print("")
print("--- RL Environment Test ---")
try:
    env = gym.make('CartPole-v1')
    obs, info = env.reset()
    print("[OK] Created Gymnasium environment")
    print("     Observation shape:", obs.shape)
    env.close()
except Exception as e:
    print("[FAIL] Environment creation:", e)

print("")
print("=== Summary ===")
print("Basic Python environment is ready!")
