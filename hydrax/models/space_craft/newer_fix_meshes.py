import trimesh
import os

# 1. Define paths (Fixed typo 'hdrax' -> 'hydrax')
mesh_dir = "/home/sam/bs/hydrax_rrl/hydrax/models/space_craft/meshes" 
export_dir = "/home/sam/bs/hydrax_rrl/hydrax/models/space_craft/meshes_fixed"

# 2. Create the export directory if it doesn't exist
os.makedirs(export_dir, exist_ok=True)

print(f"Source: {mesh_dir}")
print(f"Target: {export_dir}")

for filename in os.listdir(mesh_dir):
    if filename.endswith((".stl", ".obj")):
        file_path = os.path.join(mesh_dir, filename)
        export_path = os.path.join(export_dir, filename)
        
        print(f"Processing {filename}...")
        
        try:
            # Load mesh (Trimesh loads as triangles by default)
            mesh = trimesh.load(file_path, force='mesh')
            
            # 3. CLEANUP: This is the magic step for MJX.
            # It merges duplicate vertices and ensures the mesh is "watertight" if possible.
            # This effectively breaks the complex coplanar faces that confuse MuJoCo.
            mesh.process() 

            # Export as binary STL (most compatible with MuJoCo)
            mesh.export(export_path)
            print(f" -> Fixed and saved to: {export_path}")
            
        except Exception as e:
            print(f" !! Failed to process {filename}: {e}")

print("---")
print("All meshes processed. Please update your XML to point to the 'meshes_fixed' folder.")