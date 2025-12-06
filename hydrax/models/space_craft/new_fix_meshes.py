import trimesh
import os
import glob
import numpy as np

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "meshes") 
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "meshes_fixed") 
os.makedirs(OUTPUT_DIR, exist_ok=True)

search_patterns = [
    os.path.join(INPUT_DIR, "*.stl"),
    os.path.join(INPUT_DIR, "*.STL"),
    os.path.join(INPUT_DIR, "*.obj"),
    os.path.join(INPUT_DIR, "*.OBJ")
]

files = []
for pattern in search_patterns:
    files.extend(glob.glob(pattern))

print(f"Searching in: {INPUT_DIR}")
print(f"Found {len(files)} files.")

for file_path in files:
    filename = os.path.basename(file_path)
    
    # Skip if we accidentally picked up the output folder
    if "meshes_fixed" in file_path:
        continue

    print(f"Processing {filename}...")
    
    try:
        # 1. LOAD (Automatically fixes N-gons/Triangulates)
        # This is the most important step for fixing the Mujoco crash.
        mesh = trimesh.load(file_path, force='mesh') 
        
        # 2. BASIC CLEANUP (No external deps required)
        # Merge vertices that are identical positions
        mesh.merge_vertices()
        # Remove duplicate faces
        mesh.remove_duplicate_faces()
        # Remove degenerate faces (zero area)
        mesh.remove_degenerate_faces()
        
        # 3. EXPORT
        out_path = os.path.join(OUTPUT_DIR, filename)
        mesh.export(out_path)
        print(f"  - Saved to {out_path}")
        
    except Exception as e:
        print(f"  [Error] Could not process {filename}: {e}")

print(f"\nDone! Fixed meshes are in: {OUTPUT_DIR}")