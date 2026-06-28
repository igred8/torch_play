import numpy as np
import numpy.linalg as LA
from scipy.constants import pi, golden
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
    
def get_faces_icosa_wrong(npts):

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

def multi_subdivide_triangle(trixyz, ndivs, output_all=False):
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

def gen_centers(polys):
    cens = [p.mean(axis=0) for p in polys]
    return cens
        

def plot_sphere(ax, center=(0,0,0), radius=1.0, color='cyan', alpha=0.3, resolution=30):
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2*np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)
    
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    
    ax.plot_surface(x, y, z, alpha=alpha, color=color)

if __name__ == '__main__':
    cmap = plt.colormaps['viridis']
    axkwargs = {
        'xlabel': 'x',
        'ylabel': 'y',
        # 'zlabel': 'z',
        'aspect': 'equal',
    }

    plot3D = True

    
    
    xyz = 2*np.array([
        [1/2, -1/(2*np.sqrt(3)), 0],
        [-1/2, -1/(2*np.sqrt(3)), 0],
        [0, 1/np.sqrt(3), 0],
    ])
    tris = subdivide_triangle(xyz)
    ndivs = 1
    tris2 = multi_subdivide_triangle(xyz, ndivs=ndivs, output_all=False)
    tri_base = xyz
    polys = get_faces_icosa()
    polydivs = []
    for p in polys:
        polydivs += multi_subdivide_triangle(p, ndivs=ndivs, output_all=False)
    # tris_placed = []
    # for poly in polys:
    #     tris_ = translate_subdiv_to_face(tris2, tri_base, poly )
    #     tris_placed += tris_
    # print(len(tris_placed))

    pts_sphere, pts = scale_to_sphere(polydivs)
    print(pts_sphere.shape)

    centers = gen_centers(polydivs)
    pts_sphere_centers, pts_cen = scale_to_sphere(centers)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # ax = fig.add_subplot(111, projection='3d')
    # # for pts in [xyz]+tris:
    # #     pts = np.vstack([pts,pts[0]])
    # #     # ax.scatter(pts[:,0], pts[:,1], pts[:,2], depthshade=False)
    # #     ax.plot(pts[:,0], pts[:,1],lw=1, marker='o', ms=4)
    # i2 = 0
    # for i in range(ndivs+1):
    #     i1 = i2 
    #     i2 = i1 + 4**(i)
    #     for j,pts in enumerate(tris2[i1:i2]):
    #         pts = np.vstack([pts,pts[0]])
    #         color_ = cmap(i/ndivs)
    #         ax.plot(pts[:,0], pts[:,1], color=color_, lw=(10-2.7*i), marker='o', ms=4)
    
    # ax.set(**axkwargs)

    if plot3D:
        # polys = get_faces_icosa()

        print(len(polys))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.colormaps['viridis']
        for i,poly in enumerate(polys[:2]):
            # color_ = cmap(i/19)
            color_ = 'cyan'
            collection = Poly3DCollection([poly], alpha=1, facecolor=color_, edgecolor='black')
            ax.add_collection3d(collection)
        
        collection = Poly3DCollection(polydivs, alpha=1, facecolor='none', edgecolor='black')
        ax.add_collection3d(collection)
        # collection = Poly3DCollection(tris_placed, alpha=0.9, facecolor='none', edgecolor='black')
        # ax.add_collection3d(collection)
        pts *= 1.03
        pts_cen *= 1.1
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], color='black', s=12, depthshade=False)
        ax.scatter(pts_cen[:,0], pts_cen[:,1], pts_cen[:,2], color='red', s=12, depthshade=False)


        ax.set(**axkwargs)



        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts_sphere[:,0], pts_sphere[:,1], pts_sphere[:,2], color='black', s=12, depthshade=False)
        ax.scatter(pts_sphere_centers[:,0], pts_sphere_centers[:,1], pts_sphere_centers[:,2], color='red', s=12, depthshade=False)
        plot_sphere(ax, center=(0,0,0), radius=0.95, color='cyan', alpha=1)

        # xyz_cen = gen_vertices_dodeca()
        # # xyz_cen = gen_vertices_icosa()
        # print(xyz_cen.shape)
        # ii = np.arange(xyz_cen.shape[0])
        # x = xyz_cen[:,0]
        # y = xyz_cen[:,1]
        # z = xyz_cen[:,2]
        # # scatter = ax.scatter(x, y, z, c=ii, cmap='viridis', s=101)
        # scatter = ax.scatter(x, y, z, c='black', s=101, marker='o', depthshade=False)

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