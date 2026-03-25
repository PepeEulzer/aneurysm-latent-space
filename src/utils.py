import torch
import os

# low dependecy OBJ LOADER for torch
# Loads an .obj file and returns a tuple of vertices and faces as tensors

def yield_file(in_file):
    f = open(in_file)
    buf = f.read()
    f.close()
    for b in buf.split('\n'):
        if b.startswith('v '):
            yield ['v', [float(x) for x in b.split(" ")[1:]]]
        elif b.startswith('f '):
            triangles = b.split(' ')[1:]
            # -1 as .obj is base 1 but the Data class expects base 0 indices
            yield ['f', [int(t.split("/")[0]) - 1 for t in triangles]]
        else:
            yield ['', ""]

def read_obj(in_file):
    vertices = []
    faces = []

    for k, v in yield_file(in_file):
        if k == 'v':
            vertices.append(v)
        elif k == 'f':
            faces.append(v)

    if not len(faces) or not len(vertices):
        return None

    verts = torch.tensor(vertices, dtype=torch.float) # shape: n_verts, 3
    faces = torch.tensor(faces, dtype=torch.long).t().contiguous() # shape: 3, n_faces
    faces = torch.transpose(faces, 0, 1) # shape = n_faces, 3

    return verts, faces