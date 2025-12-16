# Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search

 [![Wheel](https://github.com/Ultikynnys/CoACD/actions/workflows/wheel.yml/badge.svg)](https://github.com/Ultikynnys/CoACD/actions/workflows/wheel.yml)

## Ultikynnys Variant

**This is the Ultikynnys optimized variant of CoACD** - the main workhorse behind the [Blender implementation of the addon](https://superhivemarket.com/products/coacd--advanced-convex-collision-generator). 

### Key Improvements
- **3-10x faster** than the original algorithm due to optimized iteration resolution
- **Fixed PCA offset bug** - colliders no longer generate at translational offsets when PCA mode is enabled
- Fully compatible with the original CoACD API

### Support Development
If you find this variant useful, please consider supporting development:



You can also support by purchasing the [Blender addon](https://superhivemarket.com/products/coacd--advanced-convex-collision-generator) or buying me a [ko-fi](https://ko-fi.com/r60dr60d)!

---

Approximate convex decomposition enables efficient geometry processing algorithms specifically designed for convex shapes (e.g., collision detection). We propose a method that is better to preserve collision conditions of the input shape with fewer components. It thus supports delicate and efficient object interaction in downstream applications.

![avatar](assets/teaser.png)

## Usage

### Installation

Download the prebuilt wheel for your platform from [GitHub Releases](https://github.com/ultikynnys/coacd/releases), then install:

X is the latest version number.

```bash
pip install coacd_u-1.0.X-cp312-abi3-win_amd64.whl
```

Wheels are available for:
- **Windows**: `coacd_u-1.0.X-cp312-abi3-win_amd64.whl`
- **Linux**: `coacd_u-1.0.X-cp312-abi3-linux_x86_64.whl`
- **macOS**: `coacd_u-1.0.X-cp312-abi3-macosx_*.whl`

### Python Example

```python
import coacd_u as coacd

# Load your mesh (using trimesh as an example)
import trimesh
mesh = trimesh.load("your_model.obj", force="mesh")

# Create CoACD mesh
coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)

# Run decomposition
parts = coacd.run_coacd(
    coacd_mesh,
    threshold=0.05,           # Concavity threshold
    max_convex_hull=-1,       # Max number of convex hulls
    preprocess_mode="auto",   # Preprocessing mode
    preprocess_resolution=50, # Preprocessing resolution
    resolution=2000,          # Sampling resolution
    mcts_nodes=20,            # MCTS nodes
    mcts_iterations=150,      # MCTS iterations
    mcts_max_depth=3,         # MCTS max depth
    pca=False,                # Enable PCA preprocessing
    merge=True,               # Enable merge post-processing
    seed=0                    # Random seed
)

# parts is a list of [vertices, faces] for each convex hull
for i, (vertices, faces) in enumerate(parts):
    print(f"Part {i}: {len(vertices)} vertices, {len(faces)} faces")
```

## Parameters

Here is the description of the parameters (sorted by importance).

* `-i/--input`: path for input mesh (`.obj`).
* `-o/--output`: path for output (`.obj` or `.wrl`).
* `-ro/--remesh-output`: path for preprocessed mesh output (`.obj`).
* `-pr/--prep-resolution`: resolution for manifold preprocess (20~100), default = 50.
* `-t/--threshold`:  concavity threshold for terminating the decomposition (0.01~1), default = 0.05.
* `-pm/--preprocess-mode`: choose manifold preprocessing mode ('auto': automatically check input mesh manifoldness; 'on': force turn on the pre-processing; 'off': force turn off the pre-processing), default = 'auto'.
* `-nm/--no-merge`: flag to disable merge postprocessing, default = false.
* `-c/--max-convex-hull`: max # convex hulls in the result, -1 for no maximum limitation, works **only when merge is enabled**, default = -1 (may introduce convex hull with a concavity larger than the threshold)
* `-mi/--mcts-iteration`: number of search iterations in MCTS (60~2000), default = 100.
* `-md/--mcts-depth`: max search depth in MCTS (2~7), default = 3.
* `-mn/--mcts-node`: max number of child nodes in MCTS (10~40), default = 20.
* `-r/--resolution`: sampling resolution for Hausdorff distance calculation (1e3~1e4), default = 2000.
* `--pca`: flag to enable PCA pre-processing, default = false.
* `-k`: value of $k$ for R_v calculation, default = 0.3.
* `-d/--decimate`: enable max vertex constraint per convex hull, default = false.
* `-dt/--max-ch-vertex`: max vertex value for each convex hull, **only when decimate is enabled**, default = 256.
* `-ex/--extrude`: extrude neighboring convex hulls along the overlapping faces (other faces unchanged), default = false.
* `-em/--extrude-margin`: extrude margin, **only when extrude is enabled**, default = 0.01.
* `-am/--approximate-mode`: approximation shape type ("ch" for convex hulls, "box" for cubes), default = "ch". I would recommend using a 2x threshold than it in convex for box approximation.
* `--seed`: random seed used for sampling, default = random().

These parameters are exposed in the Blender addon interface, allowing you to fine-tune the decomposition based on your specific needs.

Parameter tuning *tricks*: 
1. In most cases, you only need to adjust the `threshold` (0.01~1) to balance the level of detail and the number of decomposed components. A higher value gives coarser results, and a lower value gives finer-grained results. You can refer to Fig. 14 in our paper for more details.
2. If your input mesh is not manifold, you should also adjust the `preprocess_resolution` (20~100) to control the detail level of the pre-processed mesh. A larger value can make the preprocessed mesh closer to the original mesh but also lead to more triangles and longer runtime.
3. The default parameters are fast versions. If you care less about running time but more about the number of components, try to increase `mcts_max_depth`, `mcts_nodes` and `mcts_iterations` for better cutting strategies.
4. Make sure your input mesh is 2-manifold solid if you set `preprocess_mode` to `"off"`. Skipping manifold pre-processing can better preserve input details and make the process faster but the algorithm may crash or generate wrong results if the input mesh is not 2-manifold.
5. `seed` is used for reproduction of the same results as our algorithm is stochastic.

## Citation

If you find our code helpful, please cite our paper:

```
@article{wei2022coacd,
  title={Approximate convex decomposition for 3d meshes with collision-aware concavity and tree search},
  author={Wei, Xinyue and Liu, Minghua and Ling, Zhan and Su, Hao},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={4},
  pages={1--18},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```
