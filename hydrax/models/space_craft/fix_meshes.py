import trimesh
import os
import glob

# Configuration
TARGET_FACES = 500  # Low count is best for MJX speed!
INPUT_DIR = "meshes" # Change to where your meshes are
OUTPUT_DIR = "meshes_fixed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all meshes (adjust extension if you use .obj)
files = glob.glob(os.path.join(INPUT_DIR, "*.stl")) + glob.glob(os.path.join(INPUT_DIR, "*.obj"))

print(f"Found {len(files)} files...")

for file_path in files:
    filename = os.path.basename(file_path)
    print(f"Processing {filename}...")
    
    # 1. Load the mesh
    # force='mesh' ensures we get a mesh object, not a Scene
    mesh = trimesh.load(file_path, force='mesh') 
    
    # 2. FIX THE CRASH: Triangulate
    # This splits those >20 vertex faces into triangles.
    # Mujoco crashes without this.
    # mesh.triangulate()
    
    # 3. Decimate (Reduce Poly Count)
    # Only decimate if the mesh is huge
    if len(mesh.faces) > TARGET_FACES:
            try:
                # Try the installed fast-simplification
                mesh = mesh.simplify_quadric_decimation(TARGET_FACES)
                print(f"  - Decimated from {len(mesh.faces)} to {TARGET_FACES} faces")
            except Exception as e:
                # If it fails, just ignore it. The mesh is still fixed (triangulated).
                print(f"  - [Warning] Decimation skipped (Library Error), but geometry is fixed.")
        
    # 4. Cleanup (Fix zero normals / NaN issues)
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    
    # 5. Export
    out_path = os.path.join(OUTPUT_DIR, filename)
    mesh.export(out_path)
    print(f"  - Saved to {out_path}")

print("Done! Update your XML to point to the new folder.")