"""
Compute size and shape measures for aneurysms remeshed with remesh_aneurysms.py.

Author: Pepe Eulzer
Year: 2026
Publication: [UNDER REVIEW - add title / DOI]
License: see repository license

Dependencies:
* numpy
* scipy
* pandas
* pyvista

Usage:
* configure input/output paths and options in this file, then run the script

Input assumptions:
* input meshes are .obj files produced by remesh_aneurysms.py
* RES_ANGLE matches the angular resolution used during remeshing

Output:
* csv with geometry metrics
    h_max: max distance from ostium center (often called "aneurysm height")
    h_ortho: max orthogonal height relative to ostium plane
    n_max: max neck diameter
    d_max: maximal pairwise distance (often called "aneurysm size")
    volume: volume of closed surface
    volume_s: volume of enclosing sphere
    area: surface area
    aspect_ratio: aneurysm height / neck width
    flatness: geometrical flatness (eigenvalue-based)
    elongation: geometrical elongation (eigenvalue-based)
    non_sphericity: NSI18 value
    bottleneck_factor: aneurysm size / neck width
    undulation_index: ratio of volume to convex hull volume
    gln: L2-norm of Gaussian curvature
    mln: L2-norm of mean curvature

In the final dataframe:
* all filenames are included exactly once in the FILENAME_COLUMN_NAME column
* other columns from an existing metadata file (hospital, status, ...) are NaN
  where no value exists
* rows with filenames present in the metadata file but missing from DATA_PATH
  are removed
"""

import os
import pandas as pd
import numpy as np
import pyvista as pv
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull

############################################
# Defaults / Options
############################################
# path to aneurysm surfaces (obj)
DATA_PATH = './obj_out/2956_vertices'
models_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.obj')]

# specify if an existing metadata table (csv) should be extended
LABELS_FILE = None # DATA_PATH + 'labels.csv'

# set the name of the column containing the filenames
FILENAME_COLUMN_NAME = 'filename'

# path to output file, will be overwritten or created
DST_FILE = 'labels_measures.csv'

# specify angular resolution of meshes (parameter from remesh_aneurysms.py)
# this is used to quickly retrieve the ostium vertices
RES_ANGLE = 80


# ----------------------------------------
# create global dataframe
# ----------------------------------------
df = pd.DataFrame(models_files, columns=[FILENAME_COLUMN_NAME])
if LABELS_FILE is not None:
    labels_df = pd.read_csv(LABELS_FILE)
    df = pd.merge(left=df, right=labels_df, on=FILENAME_COLUMN_NAME, how='left')

# ----------------------------------------
# compute measures
# ----------------------------------------
s = df.shape[0]
h_max = np.zeros(s, dtype=np.float32)
h_ortho = np.zeros(s, dtype=np.float32)
n_max = np.zeros(s, dtype=np.float32)
d_max = np.zeros(s, dtype=np.float32)
volume = np.zeros(s, dtype=np.float32)
volume_s = np.zeros(s, dtype=np.float32)
area = np.zeros(s, dtype=np.float32)
aspect_ratio = np.zeros(s, dtype=np.float32)
flatness = np.zeros(s, dtype=np.float32)
elongation = np.zeros(s, dtype=np.float32)
non_sphericity = np.zeros(s, dtype=np.float32)
bottleneck_factor = np.zeros(s, dtype=np.float32)
undulation_index = np.zeros(s, dtype=np.float32)
gln = np.zeros(s, dtype=np.float32)
mln = np.zeros(s, dtype=np.float32)


# Note that aneurysms are registered and have normalized coordinates:
#  - the ostium lies in the x/y plane
#  - the dome points to positive z
#  - vertex 0 is the peak point
#  - vertices[-RES_ANGLE:] are the ostium vertices
for index, row in df.iterrows():
    filename = row[FILENAME_COLUMN_NAME]
    print("Computing Measures " + str(index + 1) + "/" + str(df.shape[0]) + ": " + filename)

    # load file
    mesh = pv.PolyData(os.path.join(DATA_PATH, filename))
    v_ostium = mesh.points[-RES_ANGLE:] # last vertices are ostium
    ostium_center = np.mean(v_ostium, axis=0)

    # ----------------------------------------
    # size measures
    # ----------------------------------------
    # ----------------------------------------
    # height (mm)
    h_max[index] = np.max(np.linalg.norm(mesh.points - ostium_center, axis=1)) # max distance from ostium center
    h_ortho[index] = np.max(mesh.points[:,2]) # longest orthogonal line to ostium plane = max(z)
    
    # ----------------------------------------
    # neck diameter (mm)
    n_max[index] = np.max(pdist(v_ostium)) # largest pairwise ostium distance

    # ----------------------------------------
    # size (mm)
    d_max[index] = np.max(pdist(mesh.points)) # largest pairwise mesh distance
    # volume_s[index] = (4/3) * (d_max[index]/2)**3 * np.pi # volume of this sphere (sanity check)

    # ----------------------------------------
    # dome volume
    closed_mesh = mesh.fill_holes(d_max[index], inplace=False)
    closed_mesh.compute_normals(inplace=True, auto_orient_normals=True)
    volume[index] = closed_mesh.volume

    # ----------------------------------------
    # dome surface area
    area[index] = mesh.area

    # ----------------------------------------
    # shape measures
    # ----------------------------------------
    # ----------------------------------------
    # aspect ratio
    aspect_ratio[index] = h_max[index] / n_max[index]

    # ----------------------------------------
    # flatness
    cov = np.cov(mesh.points - np.mean(mesh.points, axis=0), rowvar=False)
    evals, evecs = np.linalg.eigh(cov) # eigenvalues are sorted ascending!
    flatness[index] = np.sqrt(evals[0]/evals[2])

    # ----------------------------------------
    # elongation
    elongation[index] = np.sqrt(evals[1]/evals[2])

    # ----------------------------------------
    # NSI18
    non_sphericity[index] = 1 - (18 * np.pi)**(1/3) * (volume[index]**(2/3) / area[index])

    # ----------------------------------------
    # bottleneck factor
    bottleneck_factor[index] = d_max[index] / n_max[index]
    
    # ----------------------------------------
    # undulation index
    hull = ConvexHull(mesh.points)
    faces = np.column_stack((3*np.ones((len(hull.simplices), 1), dtype=np.int32), hull.simplices)).flatten()
    convex_hull_mesh = pv.PolyData(hull.points, faces)
    convex_hull_mesh.compute_normals(inplace=True, auto_orient_normals=True)
    undulation_index[index] = 1 - (volume[index] / convex_hull_mesh.volume)

    # ----------------------------------------
    # L2-Norm of Gaussian curvature
    gln[index] = np.sum(mesh.curvature(curv_type='gaussian')) / area[index]

    # ----------------------------------------
    # L2-Norm of Gaussian curvature
    mln[index] = np.sum(mesh.curvature(curv_type='mean')) / area[index]

# ----------------------------------------
# merge all measures and save
# ----------------------------------------
df['height_max'] = h_max
df['height_ortho'] = h_ortho
df['neck_max'] = n_max
df['diameter_max'] = d_max
df['volume'] = volume
df['volume_s'] = volume_s
df['area'] = area
df['aspect_ratio'] = aspect_ratio
df['flatness'] = flatness
df['elongation'] = elongation
df['non_sphericity'] = non_sphericity
df['bottleneck_factor'] = bottleneck_factor
df['undulation_index'] = undulation_index
df['gln'] = gln
df['mln'] = mln
df.to_csv(DST_FILE, index=False)

print("Done! Results saved to", DST_FILE)
