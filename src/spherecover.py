import numpy as np
import numpy.linalg as LA
from scipy.constants import pi, golden
from scipy.spatial.transform import Rotation

def spherical_to_xyz(phi, theta, rad=1):
    
    x = rad * np.sin(phi) * np.cos(theta)
    y = rad * np.sin(phi) * np.sin(theta)
    z = rad * np.cos(phi)
    return x,y,z
    
def xyz_to_spherical(x, y, z):
    rad = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / rad)
    phi = np.atan2(y, x) + pi
    return rad, theta, phi

def gen_random(npts, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(seed=rng)

    phi = rng.uniform(0, np.pi, npts)      # polar angle
    theta = rng.uniform(0, 2*np.pi, npts)  # azimuthal angle

    return spherical_to_xyz(phi, theta)

def gen_anglegrid(npts):
    phi = np.linspace(0, 2*pi, npts)
    theta = np.linspace(0, pi, npts)

    phi_mesh, theta_mesh = np.meshgrid(phi, theta)
    phi_mesh = np.reshape(phi_mesh, [-1, ])
    theta_mesh = np.reshape(theta_mesh, [-1, ])
    return spherical_to_xyz(phi_mesh, theta_mesh)
    
def gen_vertices_dodeca():
    ones = np.array([
        [1, 1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, -1, -1],    
    ])
    v1 = np.array([
        [0, 1/golden, golden],
        [0, -1/golden, golden],
        [0, 1/golden, -golden],
        [0, -1/golden, -golden],
        ])
    v2 = np.array([
        [1/golden, golden, 0],
        [-1/golden, golden, 0],
        [1/golden, -golden, 0],
        [-1/golden, -golden, 0],
        ])
    v3 = np.array([
        [golden, 0, 1/golden],
        [-golden, 0, 1/golden],
        [golden, 0, -1/golden],
        [-golden, 0, -1/golden],
        ])
    xyz = np.vstack([ones, v1, v2, v3])
    
    return xyz
    
def gen_vertices_icosa():
    
    v1 = np.array([
        [0,  1,  golden],
        [0,  1, -golden],
        [0, -1,  golden],
        [0, -1, -golden],
    ])
    v2 = np.array([
        [ 1,  golden, 0],
        [ 1, -golden, 0],
        [-1,  golden, 0],
        [-1, -golden, 0],
    ])
    v3 = np.array([
        [ golden, 0,  1],
        [ golden, 0, -1],
        [-golden, 0,  1],
        [-golden, 0, -1],
    ])
    xyz = np.vstack([v1, v2, v3])
    # xyz = xyz / (2)
    
    return xyz

def align_to_z(xyz):
    zhat = np.array([0,0,1])
    rot, _ = Rotation.align_vectors(xyz[0], zhat)
    xyz = rot.as_matrix() @ xyz.T
    return xyz.T

def gen_regular_polygon_verticies(face_center, face_normal, n_vertex, r=1):
    theta = np.linspace(0, 2*pi, n_vertex+1)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(x)

    xyz = np.vstack([x,y,z])

    zhat = np.array([0,0,1])
    rot, _ = Rotation.align_vectors(face_normal, zhat)
    verts = rot.as_matrix() @ xyz + face_center.reshape([3,1])
    
    return verts.T

def subdivide_triangle(tri_xyz:np.ndarray):
    """Subdivide a triangle into 4 smaller triangles, by taking the midpoints of each side.

    Parameters
    ----------
    tri_xyz : np.ndarray
        The vertices of the triangle. Each row is a vertex with x,y,z. shape(3,3)

    Returns
    -------
    list[np.ndarray]
        The four triangles that result. List of NDArrays where each row is a vertex with x,y,z. shape(3,3)
    """
    inds1 = [0, 1, 2]
    inds2 = [1, 2, 0]
    new_xyz = (tri_xyz[inds1] + tri_xyz[inds2]) / 2
    # group pts into triangles
    t1 = np.vstack([tri_xyz[0], new_xyz[0], new_xyz[2]])
    t2 = new_xyz
    t3 = np.vstack([tri_xyz[1], new_xyz[0], new_xyz[1]])
    t4 = np.vstack([tri_xyz[2], new_xyz[1], new_xyz[2]])
    return [t1, t2, t3, t4]

def multi_subdivide_triangle(trixyz:np.ndarray, ndivs:int, output_all:bool=False):
    """Subdivide a triangle into smaller triangles by recursively halving the sides.

    Parameters
    ----------
    trixyz : np.ndarray
        The vertices of the triangle shape (3,3). Each row is a vertex with x,y,z.
    ndivs : int
        How many times to halve the sides and produce subdivisions.
    output_all : bool, optional
        Output all subdivisions or just the final round (smallest triangles), by default False => only smallest output.

    Returns
    -------
    list[np.ndarray]
        The list of triangles, each defined by a set of three vertices.
    """
    tris_keep = [trixyz]
    tris_walk = [trixyz]
    tot_tris = (4**(ndivs+1) - 1) / (4 - 1) # geometric sum
    while len(tris_keep) < tot_tris:
        tri_current = tris_walk.pop()
        tris = subdivide_triangle(tri_current)
        tris_keep += tris
        for t in tris:
            tris_walk.insert(0,t)
    if output_all:
        return tris_keep
    else:
        return tris_keep[-(4**ndivs):]

def get_faces_icosa():
    """Output the sets of vertices that define the faces of an icosahedron.
    Each face is a triangle and so has 3 vertices.
    The global orientation of the icosahedron is as defined by the `get_vertices_icosa()` function.

    Returns
    -------
    list[np.ndarray]
        List of NDArrays with three xyz points each. Each row is a vertex. shape = (3,3)
    """
    vertices = gen_vertices_icosa()
    vertmap = {i:v for i,v in enumerate(vertices)}
    faces = set()
    for i, pt in vertmap.items():
        dist = LA.norm(vertices - pt, axis=1)
        inds = np.argpartition(dist, 6)[1:6]
        inds = [int(ind) for ind in inds]
        mindist = np.inf
        for c, j in enumerate(inds):
            
            closest_k = -99

            for k in inds[c+1:]:
                dist_jk = LA.norm(vertmap[j] - vertmap[k])
                if dist_jk <= mindist:
                    mindist = dist_jk
            
            for k in inds[c+1:]:
                dist_jk = LA.norm(vertmap[j] - vertmap[k])
                if np.isclose(dist_jk, mindist):
                    closest_k = k
                    faces.add(tuple(sorted((i, j, closest_k))))
            
            
    
    face_verts = [np.vstack([vertices[i,:] for i in inds]) for inds in faces]
    
    return face_verts

def translate_subdiv_to_face(trisubs, tri_base, face_verts):
    face_center = face_verts.mean(axis=0) 
    rot2z, rotz = get_rotation_to_face(tri_base, face_verts)
    tris_arr = np.vstack(trisubs)
    
    tris_trans = rot2z.apply(rotz.apply(tris_arr, inverse=False), inverse=True)
    tris_trans += face_center.reshape([1,3])

    tris_trans = [tris_trans[3*i:3*(i+1)] for i in range(len(trisubs))]
    return tris_trans

def get_rotation_to_face(tri_base, face_verts):
    face_center = face_verts.mean(axis=0) 
    zhat = np.array([0,0,1])
    rot2z, _ = Rotation.align_vectors(face_center, zhat)
    face_verts_rot = rot2z.apply(face_verts)
    rotz, _ = Rotation.align_vectors(face_verts_rot, tri_base)
    return rot2z, rotz

def scale_to_sphere(tris, rad=1):
    pts = np.vstack(tris)
    pts = np.unique(pts, axis=0)
    pts_norm = LA.norm(pts, ord=2, axis=1, keepdims=True)
    pts_sphere = rad * pts / pts_norm
    
    return pts_sphere, pts

def gen_centers(polys:list[np.ndarray]):
    """Calculate the centroids of each polygon in the list.

    Parameters
    ----------
    polys : list[np.ndarray]
        Each element of the list is a NDArray that defines the polygon. shape = (n,3) with each row a vertex in x,y,z.

    Returns
    -------
    list[np.ndarray]
        The centroids in xyz coordinates.
    """
    cens = [p.mean(axis=0) for p in polys]
    return cens

def calc_closest_distribution(xyz, n_close=10):
    
    cov_met = [calc_closest_distances(pt, xyz, n_close=n_close) for pt in xyz]
    return np.array(cov_met)

def calc_closest_distances(pt, pts, n_close=10):
    dists = LA.norm(pts - pt, axis=1, ord=2)
    inds = np.argpartition(dists, n_close)[:n_close]
    dists_closest = dists[inds]
    return dists_closest.mean()

def plot_sphere(ax, center=(0,0,0), radius=1.0, color='cyan', alpha=0.3, resolution=30):
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2*np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)
    
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    
    ax.plot_surface(x, y, z, alpha=alpha, color=color)

__namespace__ = 'spherecover'
__author__ = 'Ivan Gadjev'
__year__ = '2026'
__status__ = 'development'
if __name__ == '__main__':
    print(f"This is the {__namespace__} namespace. Written by {__author__} in {__year__}.")