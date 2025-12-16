import coacd_u as coacd
import numpy as np

import trimesh
import os

input_file = "examples/Bottle.obj"
print(f"Loading {input_file}...")
mesh_in = trimesh.load(input_file, force="mesh")
coacd.set_log_level("error")
print(f"Vertices: {mesh_in.vertices.shape} {mesh_in.vertices.dtype}")
print(f"Faces: {mesh_in.faces.shape} {mesh_in.faces.dtype}")

mesh = coacd.Mesh(mesh_in.vertices, mesh_in.faces)
print("Mesh object created.")

print("Calling run_coacd...")
try:
    res = coacd.run_coacd(mesh, threshold=0.05)
    print(f"Result: {len(res)} parts")
except Exception as e:
    print(f"Exception: {e}")
