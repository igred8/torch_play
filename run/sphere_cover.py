import numpy as np
from scipy.constants import pi, golden
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

def angles_to_xys(phi, theta):
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x,y,z

def gen_random(npts, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(seed=rng)

    phi = rng.uniform(0, np.pi, npts)      # polar angle
    theta = rng.uniform(0, 2*np.pi, npts)  # azimuthal angle

    return angles_to_xys(phi, theta)

def gen_anglegrid(npts):
    phi = np.linspace(0, 2*pi, npts)
    theta = np.linspace(0, pi, npts)

    phi_mesh, theta_mesh = np.meshgrid(phi, theta)
    phi_mesh = np.reshape(phi_mesh, [-1, ])
    theta_mesh = np.reshape(theta_mesh, [-1, ])
    return angles_to_xys(phi_mesh, theta_mesh)
    
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
    # xyz = ones
    # for v in [v1, v2, v3]:
    #     v_ = v * ones
    #     xyz = np.concatenate([xyz, v_])
    # xyz = set([tuple(pt) for pt in xyz])
    # xyz = np.array([list(pt) for pt in xyz])
    # xyz = xyz / np.sqrt(2)
    xyz = align_to_z(xyz)
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
    
    # xyz = align_to_z(xyz)
    
    return xyz
    
def gen_icosa_subdiv(npts):

    face_centers = gen_vertices_dodeca()
    vertices = gen_vertices_icosa()

    polygons = []
    for cen in face_centers:
        dist = np.linalg.norm(vertices - cen, axis=1)
        inds = np.argpartition(dist, 3)[:3]
        polygons.append(inds)
    polygons = [vertices[inds] for inds in polygons]
    return polygons

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

def subdivide_triangle(tri_xyz):
    inds1 = [0, 1, 2]
    inds2 = [1, 2, 0]
    new_xyz = (tri_xyz[inds1] + tri_xyz[inds2]) / 2
    # group pts into triangles
    t1 = np.vstack([tri_xyz[0], new_xyz[0], new_xyz[2]])
    t2 = new_xyz
    t3 = np.vstack([tri_xyz[1], new_xyz[0], new_xyz[1]])
    t4 = np.vstack([tri_xyz[2], new_xyz[1], new_xyz[2]])
    return [t1, t2, t3, t4]

def multi_subdivide_triangle(trixyz, ndivs):
    tris_keep = [trixyz]
    tris_walk = [trixyz]
    tot_tris = (4**(ndivs+1) - 1) / (4 - 1) # geometric sum
    while len(tris_keep) < tot_tris:
        tri_current = tris_walk.pop()
        tris = subdivide_triangle(tri_current)
        tris_keep += tris
        for t in tris:
            tris_walk.insert(0,t)
    return tris_keep

if __name__ == '__main__':
    cmap = plt.colormaps['viridis']
    axkwargs = {
        'xlabel': 'x',
        'ylabel': 'y',
        # 'zlabel': 'z',
        'aspect': 'equal',
    }

    plot3D = False

    
    xyz = np.array([
        [1/2, -1/(2*np.sqrt(3)), 0],
        [-1/2, -1/(2*np.sqrt(3)), 0],
        [0, 1/np.sqrt(3), 0],
    ])
    tris = subdivide_triangle(xyz)
    ndivs = 4
    tris2 = multi_subdivide_triangle(xyz, ndivs=ndivs)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = fig.add_subplot(111, projection='3d')
    # for pts in [xyz]+tris:
    #     pts = np.vstack([pts,pts[0]])
    #     # ax.scatter(pts[:,0], pts[:,1], pts[:,2], depthshade=False)
    #     ax.plot(pts[:,0], pts[:,1],lw=1, marker='o', ms=4)
    i2 = 0
    for i in range(ndivs+1):
        i1 = i2 
        i2 = i1 + 4**(i)
        for j,pts in enumerate(tris2[i1:i2]):
            pts = np.vstack([pts,pts[0]])
            color_ = cmap(i/ndivs)
            ax.plot(pts[:,0], pts[:,1], color=color_, lw=(10-2.7*i), marker='o', ms=4)
    # for j,pts in enumerate(tris2[5:9]):
    #     pts = np.vstack([pts,pts[0]])
        
    #     ax.plot(pts[:,0], pts[:,1],  lw=1, marker='o', ms=4)

    ax.set(**axkwargs)

    if plot3D:
        polys = gen_icosa_subdiv(9)

        print(len(polys))

        # x,y,z = gen_dodeca()
        # vert_dodeca = np.vstack([x,y,z]).T
        # faces_icosa = [gen_regular_polygon_verticies(vec, vec, 3, r=1/np.sqrt(2)) for vec in vert_dodeca ]
        # print(np.linalg.norm(vert_dodeca - vert_dodeca[0], axis=1))
        
        # x,y,z = gen_icosa()
        # vert_icosa = np.vstack([x,y,z]).T
        # faces_dodeca = [gen_regular_polygon_verticies(vec, vec, 5, r=1/np.sqrt(2)) for vec in vert_icosa ]
        # print(np.linalg.norm(vert_icosa - vert_icosa[0], axis=1))
        

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.colormaps['viridis']
        for i,poly in enumerate(polys):
            collection = Poly3DCollection([poly], alpha=0.3, facecolor=cmap(i/19), edgecolor='black')
            ax.add_collection3d(collection)
        # collection = Poly3DCollection(faces_icosa, alpha=0.5, facecolor='tab:red', edgecolor='black')
        # ax.add_collection3d(collection)
        
        xyz_cen = gen_vertices_dodeca()
        # xyz_cen = gen_vertices_icosa()
        print(xyz_cen.shape)
        ii = np.arange(xyz_cen.shape[0])
        x = xyz_cen[:,0]
        y = xyz_cen[:,1]
        z = xyz_cen[:,2]
        # scatter = ax.scatter(x, y, z, c=ii, cmap='viridis', s=101)
        scatter = ax.scatter(x, y, z, c='black', s=101, marker='o', depthshade=False)

        # collection = Poly3DCollection(faces_dodeca, alpha=0.5, facecolor='tab:pink', edgecolor='black')
        # ax.add_collection3d(collection)
        # ii = np.arange(x.shape[0])
        # scatter = ax.scatter(x, y, z, c=ii, cmap='viridis', s=101)

        ax.set(**axkwargs)

        # x,y,z = gen_random(100, rng=4211)    
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=10)
        # plt.colorbar(scatter, label='Z (km)')

        # ax.set(**axkwargs)

        # x,y,z = gen_anglegrid(30)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
        # plt.colorbar(scatter, label='Z (km)')

        # ax.set(**axkwargs)
    
    
    plt.show(block=False)

    input("Press Enter to close plots...")
    plt.close('all')