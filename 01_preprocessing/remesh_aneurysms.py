"""
Register, resample, and remesh aneurysm sacs with consistent vertex ordering.

Author: Pepe Eulzer
Year: 2026
Publication: [UNDER REVIEW - add title / DOI]
License: see repository license

Dependencies:
* numpy
* scipy
* pyvista
* igl

Usage:
* configure input/output paths and options in this file, then run the script

Input assumptions:
* input meshes are .obj files
* each mesh has exactly one boundary loop corresponding to the ostium

Outputs are registered such that:
* ostium barycenter -> 0
* ostium loop -> xy-plane
* dome -> z-axis

The remeshed outputs are intended for (optional) downstream processing with
measure_aneurysms.py and preserve a consistent vertex ordering across cases.
"""

import os
from multiprocessing import Pool
from timeit import default_timer as timer

import igl
import numpy as np
import pyvista as pv
from scipy.spatial.distance import pdist

############################################
# Defaults / Options
############################################

# read/write paths, relative to shell
# IN_PATH_LIST can contain multiple folders but is not recursive
IN_PATH_LIST = [
    "./obj_in",
]
OUT_PATH = "./obj_out"
OUTPUT_FACES = True          # save the shared face list to OUT_PATH/faces_<nr of vertices>.txt

USE_MULTIPROCESSING = True   # recommended True for >100 meshes

# meshing options
NORMALIZE_SIZE = False       # if True, meshes are normalized to unit sphere
GEODESICS_METHOD = 2         # 0: ostium-seeded geodesics, 1: dome-seeded geodesics, 2: evenly weighted (recommended)
DYNAMIC_ANGLE_RES = True     # will resolve less vertices at the dome peak (recommended)
RES_ANGLE = 80               # should be in [5, 10, 20, 40, 80, 160...] if dynamic angle res is True
RES_GEOD = int(RES_ANGLE/2)  # recommended to be RES_ANGLE/2 (parameter space approximates a half-sphere)
LOOP_SUBDIV = True           # iteratively subdivides the input mesh using Loop subdivision to match the target resolution (recommended)
NEAT_3D_REGISTRATION = True  # forces dome peak registration to z-axis exactly (recommended)


############################################
# Script Functions / consts
############################################

# quick rotation matrices
ROT_X_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
ROT_Y_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
ROT_Z_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

# get connectivity matrix for the resampled mesh
def get_connectivity_matrix():
    nr_samples = RES_ANGLE
    if DYNAMIC_ANGLE_RES:
        nr_samples = 5

    faces = []
    p0 = np.zeros((nr_samples), dtype=np.int32)
    p1 = np.arange(1, nr_samples+1, 1, dtype=np.int32)
    p2 = np.roll(p1, -1)
    faces.append(np.stack((p0, p1, p2), axis=1))

    # connect dynamic rings
    current_idx = 1
    if DYNAMIC_ANGLE_RES:
        while nr_samples < RES_ANGLE:
            p0 = np.empty((nr_samples*2), dtype=np.int32)
            p0[0::2] = np.arange(current_idx, current_idx + nr_samples)
            p0[1::2] = np.arange(current_idx, current_idx + nr_samples)
            p0 = np.roll(p0, -1)

            p0_inner = np.arange(current_idx, current_idx + nr_samples, 1, dtype=np.int32)
            p2_inner = np.roll(p0_inner, -1)

            current_idx += nr_samples
            nr_samples *= 2

            p1 = np.arange(current_idx, current_idx + nr_samples, 1, dtype=np.int32)
            p2 = np.roll(p1, -1)

            p1_inner = np.arange(current_idx+1, current_idx + nr_samples, 2, dtype=np.int32)

            faces.append(np.stack((p0, p1, p2), axis=1))
            faces.append(np.stack((p0_inner, p1_inner, p2_inner), axis=1))

    # connect static rings
    already_drawn = int((len(faces)-1)/2)
    for i in range(current_idx, RES_ANGLE * (RES_GEOD - already_drawn - 1), RES_ANGLE):
        inner_ring = np.arange(i, i+RES_ANGLE)
        outer_ring = np.arange(i+RES_ANGLE, i+RES_ANGLE*2)

        p0 = np.empty((RES_ANGLE*2), dtype=np.int32)
        p0[0::2] = inner_ring
        p0[1::2] = inner_ring

        p1 = np.empty((RES_ANGLE*2), dtype=np.int32)
        p1[0::2] = outer_ring
        p1[1::2] = outer_ring
        p1 = np.roll(p1, -1)

        p2 = np.empty((RES_ANGLE*2), dtype=np.int32)
        p2[0::2] = outer_ring
        p2[1::2] = inner_ring
        p2 = np.roll(p2, -2)

        faces.append(np.stack((p0, p1, p2), axis=1))

    return np.vstack(faces)

FACES = get_connectivity_matrix()
N_FACES = FACES.shape[0]
FACES_FLAT = np.hstack((np.full((N_FACES, 1), 3), FACES)).flat

# save connectivity matrix / face list
if OUTPUT_FACES:
    np.savetxt(os.path.join(OUT_PATH, "faces_" + str(np.max(FACES_FLAT)+1) + ".txt"), FACES, fmt='%i')

def compute_normalized_mesh(file_nr, file_name):
    print('Processing file', file_nr, file_name)
    messages_stack = []
    try:
        #############################
        # loading, cleaning, subdividing mesh
        #############################
        mesh = pv.read(str(file_name)) # required for cleanup

        # Clean and compute consistent (!) normals with VTK
        mesh.clean(inplace=True, tolerance=1e-6)
        mesh.compute_normals(inplace=True, split_vertices=False)

        # Extract vertices, faces for igl
        AneuVertPos = np.array(mesh.points, dtype=np.float64, order='C')  # shape: (n, 3)
        faces = mesh.faces.reshape((-1, 4))[:, 1:4]  # PyVista stores faces as [n_pts, v0, v1, v2]
        AneuTriClean = np.array(faces, dtype=np.int32, order='C')

        # subdivide original mesh using Loop subdivision if mesh has insufficient (<1/3) points
        if LOOP_SUBDIV:
            target_vertices = RES_ANGLE * RES_GEOD
            source_vertices = AneuVertPos.shape[0]
            while target_vertices > source_vertices * 3:
                AneuVertPos, AneuTriClean = igl.loop(AneuVertPos, AneuTriClean, 1)
                source_vertices = AneuVertPos.shape[0]

        # check if the size of the mesh is reasonable (1-80mm diagnonal)
        bb_d = igl.bounding_box_diagonal(AneuVertPos)
        if not 1 < bb_d < 80:
            messages_stack.append(
                str(file_name) + "(file " + str(file_nr) + ") " +
                "WARNING: Bounding box diagonal has an unreasonable size of %.2fmm. Possible unit error?" %bb_d
            )

        #############################
        # geodesic parameterization / peak detection
        #############################
        bnd = igl.boundary_loop(AneuTriClean) # vertex source indices
        avg_edge_len = igl.avg_edge_length(AneuVertPos, AneuTriClean)
        t = avg_edge_len**2 * 10
        GeodesicDist = igl.heat_geodesic(AneuVertPos, AneuTriClean, t, bnd) # better at contours
        GeodesicDist_relaxed = igl.heat_geodesic(AneuVertPos, AneuTriClean, avg_edge_len**2 * 1000, bnd) # better everywhere else
        index_max_Geodesic = np.argmax(GeodesicDist_relaxed) # aneurysm peak vertex index

        # check fields
        unique_count = np.unique(GeodesicDist).size
        if unique_count < 100:
            messages_stack.append(
                str(file_name) + "(file " + str(file_nr) + ") " +
                "CRITICAL WARNING: Geodesic field resolve may have failed. Only %i unique distances in output." %unique_count
            )

        # compute peak geodesics
        if GEODESICS_METHOD != 0:
            GeodesicDist_peak = igl.heat_geodesic(AneuVertPos, AneuTriClean, t, np.array([index_max_Geodesic], dtype=np.int32))

        #############################
        # 3D registration
        #############################
        ostium_mean = np.mean(AneuVertPos[bnd,:], axis=0)
        ostium_translated = AneuVertPos[bnd,:] - ostium_mean
        cov = np.cov(ostium_translated, rowvar=False)
        evals, evecs = np.linalg.eigh(cov) # eigenvalues are sorted ascending
        rot_matrix = np.flip(evecs, axis=1) # -> flip x/z components to align ostium in x/y plane

        translated_pos = AneuVertPos - ostium_mean
        AneuVertPos = np.dot(translated_pos, rot_matrix)

        # turn around x if sac alignment is along -z
        if(np.max(AneuVertPos[:,2]) < np.abs(np.min(AneuVertPos[:,2]))):
            AneuVertPos = np.dot(AneuVertPos, ROT_X_180)
            rot_matrix = rot_matrix @ ROT_X_180

        # turn around z if sac alignment is along +y
        if np.mean(AneuVertPos[:,1]) > 0:
            AneuVertPos = np.dot(AneuVertPos, ROT_Z_180)
            rot_matrix = rot_matrix @ ROT_Z_180

        # Save translation and rotation for later use
        # np.savetxt(os.path.join(OUT_PATH, os.path.basename(file_name).replace('.obj', '_translation.txt')), -ostium_mean)
        # np.savetxt(os.path.join(OUT_PATH, os.path.basename(file_name).replace('.obj', '_rot_matrix.txt')), rot_matrix)

        # note: rotation along y is not considered because all degrees of freedom are now resolved
        # - the ostium is in x/y plane (least squares), longer side of saddle-shaped ostia are along x-axis
        # - the dome points towards +z
        # - orientation around z is such that the larger side of the sac is towards -y

        #############################
        # angle parameterization
        #############################
        # LSCM initalization
        b = np.zeros(2, dtype=np.int32)
        bnd_x = AneuVertPos[bnd][:,0]
        b[0] = bnd[np.argmin(bnd_x)]
        b[1] = bnd[np.argmax(bnd_x)]
        bc = AneuVertPos[b][:,:2].astype(np.float64, order='C') # remove z-component (project onto x/y plane)

        _, uv = igl.lscm(AneuVertPos.astype(np.float64, order='C'), AneuTriClean, b, bc)

        # translate aneurysm peak to (0)
        uv -= uv[index_max_Geodesic,:]

        # compute 2D angle of each vertex
        # since LSCM orientation is not consistent counter-clockwise orientation is enforced
        idx_min_y = np.argmin(uv[:,1])
        idx_max_y = np.argmax(uv[:,1])
        if AneuVertPos[idx_min_y,1] < AneuVertPos[idx_max_y,1]:
            Angle = np.arctan2(uv[:,1], uv[:,0]) + np.pi # [-pi, pi] -> [0, 2pi]
        else:
            Angle = -np.arctan2(uv[:,1], uv[:,0]) + np.pi # [pi, -pi] -> [2pi, 0]

        #############################
        # r parameterization
        #############################
        if GEODESICS_METHOD == 0:
            GD_norm = 1.0 - GeodesicDist / np.max(GeodesicDist) # for GD starting at ostium -> distorts peak geometry
        elif GEODESICS_METHOD == 1:
            GD_norm = GeodesicDist_peak / np.min(GeodesicDist_peak[bnd]) # for GD starting at peak -> removes geometry at ostium
        else:
            # evenly distributes geodesics between peak and ostium (best results)
            GD_norm_peak = GeodesicDist_peak / np.max(GeodesicDist_peak) # normalize GD starting at peak
            GD_norm_ostium = 1.0 - GeodesicDist / np.max(GeodesicDist) # normalize GD starting at ostium
            GD_norm_ostium_relaxed = 1.0 - GeodesicDist_relaxed / np.max(GeodesicDist_relaxed) # normalize relaxed GD starting at ostium

            GD_interp = GD_norm_ostium * GD_norm_ostium + (1-GD_norm_ostium) * GD_norm_ostium_relaxed
            GD_norm = GD_interp * GD_interp + (1-GD_interp) * GD_norm_peak

        # combine angle and geodesics (normalized)
        param_space_verts = np.stack((GD_norm * np.cos(Angle), GD_norm * np.sin(Angle), np.zeros(GD_norm.shape[0])), axis=1)

        #############################
        # create sample points
        #############################
        if DYNAMIC_ANGLE_RES:
            samples = []
            radii = np.linspace(0, 1, RES_GEOD + 1)
            nr_samples = 5
            for i in range(0, RES_GEOD):
                if nr_samples <= RES_ANGLE:
                    phi = np.linspace(0, 2*np.pi, nr_samples, endpoint=False)
                    cos_phi = np.cos(phi)
                    sin_phi = np.sin(phi)

                ring = np.zeros(shape=(nr_samples, 3), dtype=np.float64)
                ring[0:nr_samples,0] = radii[i+1] * cos_phi
                ring[0:nr_samples,1] = radii[i+1] * sin_phi
                samples.append(ring)

                if nr_samples < RES_ANGLE:
                    nr_samples *= 2

            samples = np.vstack(samples)

        else:
            phi = np.linspace(0, 2*np.pi, RES_ANGLE, endpoint=False)
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            samples = np.zeros(shape=(RES_ANGLE * RES_GEOD, 3), dtype=np.float64)
            radii = np.linspace(0, 1, RES_GEOD + 1)
            for i in range(0, RES_GEOD):
                start = i * RES_ANGLE
                stop = (i+1) * RES_ANGLE
                samples[start:stop,0] = radii[i+1] * cos_phi
                samples[start:stop,1] = radii[i+1] * sin_phi

        # add peak sample point
        samples = np.vstack(([[0.0, 0.0, 0.0]], samples))

        # find hit triangles
        _, l_indices, _ = igl.point_mesh_squared_distance(samples, param_space_verts, AneuTriClean)
        sample_tri_2D = param_space_verts[AneuTriClean[l_indices]]
        sample_tri_3D = AneuVertPos[AneuTriClean[l_indices]]

        #############################
        # map uniform coordinates
        #############################
        barycentric_coords = igl.barycentric_coordinates_tri(
            samples.astype(np.float64, order='C'),
            sample_tri_2D[:,0,:].astype(np.float64, order='C'),
            sample_tri_2D[:,1,:].astype(np.float64, order='C'),
            sample_tri_2D[:,2,:].astype(np.float64, order='C'))

        # map to 3D
        nr_triangles_hit = sample_tri_2D.shape[0]
        mapped_points = []
        for i in range(nr_triangles_hit):
            sample_3D = sample_tri_3D[i,0] * barycentric_coords[i,0] + \
                        sample_tri_3D[i,1] * barycentric_coords[i,1] + \
                        sample_tri_3D[i,2] * barycentric_coords[i,2]
            mapped_points.append(sample_3D)
        mapped_points = np.array(mapped_points)

        #############################
        # neat 3D registration (force peak onto z-axis)
        #############################
        if NEAT_3D_REGISTRATION:
            z_axis = np.copy(mapped_points[0]) # peak point
            x_axis = mapped_points[-RES_ANGLE] # ostium point on angle = 0
            y_axis = np.cross(x_axis, z_axis)
            x_axis = np.cross(y_axis, z_axis)

            x_axis /= np.linalg.norm(x_axis)
            y_axis /= np.linalg.norm(y_axis)
            z_axis /= np.linalg.norm(z_axis)

            R = np.row_stack((x_axis, y_axis, z_axis)) # transposed column matrix
            mapped_points = R.dot(mapped_points.T).T

        # test orientation
        ostium_first_half = mapped_points[int(-RES_ANGLE/2):]
        if np.sum(ostium_first_half[:,1]) < 0:
            messages_stack.append(
                str(file_name) + " (file " + str(file_nr) + "): " +
                "CRITICAL WARNING: Vertex ordering may be flipped / ostium orientation is wrong. <---------------------"
            )

        #############################
        # size normalization (if selected)
        #############################
        # normalize mesh size to unit sphere (do it here to avoid costly pdist on large meshes)
        if NORMALIZE_SIZE:
            mean = np.mean(mapped_points, axis=0)
            v_moved = mapped_points - mean # move to center
            max_rad = np.max(pdist(v_moved)) / 2 # largest pairwise vertex distance
            mapped_points = v_moved / max_rad # scale vertices
            mapped_points = mapped_points + (mean / max_rad)  # move back (relative distance)

    except Exception as error:
        messages_stack.append(
            "-------------------------\n" +
            str(file_name) + " (file ID " + str(file_nr) + ") " +
            "ERROR: Processing failed. " + str(error) +
            "\n-------------------------"
            )
        return messages_stack
    else:
        # write result then discard to free memory
        write_path = os.path.join(OUT_PATH, str(mapped_points.shape[0]) + "_vertices")
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        file_path = os.path.join(write_path, os.path.basename(file_name))
        igl.write_obj(file_path, mapped_points, FACES)
        print("Written:", file_path)
        return messages_stack

if __name__ == '__main__':
    #############################
    # Iterate directories to find .obj files
    #############################
    obj_files = []
    for path in IN_PATH_LIST:
        for file in os.listdir(path):
            if file.lower().endswith('.obj'):
                obj_files.append(os.path.join(path, file))
    print("Found %i .obj files." %len(obj_files))

    #############################
    # Compute normalized meshes
    #############################
    file_ids = range(len(obj_files))
    messages_list = []
    start_time = timer()

    # multiprocessing
    if USE_MULTIPROCESSING:
        print("Starting processing (multiprocessing)...")
        with Pool(8) as p:
            messages_list = p.starmap(compute_normalized_mesh, zip(file_ids, obj_files))
    
    # linear processing
    else:
        print("Starting processing (linear)...")
        for file_id in file_ids:
            messages_list.append(compute_normalized_mesh(file_id, obj_files[file_id]))

    # print aggregated warnings/errors after processing
    for messages in messages_list:
        for m in messages:
            print(m)
    
    total_time = timer() - start_time
    print("Processing done. Processed %i files. Total time: %.5fs" %(len(file_ids), total_time))
