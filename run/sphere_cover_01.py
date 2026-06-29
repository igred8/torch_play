import numpy as np
import numpy.linalg as LA
from scipy.constants import pi, golden
from scipy.spatial.transform import Rotation

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import sys
sys.path.append('./src')
sys.path.append('../src')
import spherecover as sc

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

    # define
    
    xyz = 2*np.array([
        [1/2, -1/(2*np.sqrt(3)), 0],
        [-1/2, -1/(2*np.sqrt(3)), 0],
        [0, 1/np.sqrt(3), 0],
    ])
    ndivs = 1
    out_all_triangles = False
    tris = sc.multi_subdivide_triangle(xyz, ndivs=ndivs, output_all=out_all_triangles)
    
    polys = sc.get_faces_icosa()
    polydivs = []
    for p in polys:
        polydivs += sc.multi_subdivide_triangle(p, ndivs=ndivs, output_all=out_all_triangles)

    pts_sphere, pts = sc.scale_to_sphere(polydivs)
    print(pts_sphere.shape)

    centers = sc.gen_centers(polydivs)
    pts_sphere_centers, pts_cen = sc.scale_to_sphere(centers)
    
    
    plot_flattri_flag = True
    plot_flattri_flag = False 
    if plot_flattri_flag:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        # for pts in [xyz]+tris:
        #     pts = np.vstack([pts,pts[0]])
        #     ax.plot(pts[:,0], pts[:,1],lw=1, marker='o', ms=4)
        i2 = 0
        for i in range(ndivs+1):
            i1 = i2 
            i2 = i1 + 4**(i)
            for j,pts in enumerate(tris[i1:i2]):
                pts = np.vstack([pts,pts[0]])
                color_ = cmap(i/ndivs)
                ax.plot(pts[:,0], pts[:,1], color=color_, lw=(10-2.7*i), marker='o', ms=4)
        
        ax.set(**axkwargs)


    print(len(polys))

    
    plot_polyhedron_flag = True
    # plot_polyhedron_flag = False 
    if plot_polyhedron_flag:
        fig_polyhedron = plt.figure()
        ax = fig_polyhedron.add_subplot(111, projection='3d')
        cmap = plt.colormaps['viridis']
        for i,poly in enumerate(polys[:1]):
            # color_ = cmap(i/19)
            color_ = 'cyan'
            collection = Poly3DCollection([poly], alpha=1, facecolor=color_, edgecolor='black')
            ax.add_collection3d(collection)
        
        collection = Poly3DCollection(polydivs[:4], alpha=1, facecolor='none', edgecolor='black')
        ax.add_collection3d(collection)
        # collection = Poly3DCollection(tris_placed, alpha=0.9, facecolor='none', edgecolor='black')
        # ax.add_collection3d(collection)
        pts *= 1.03
        pts_cen *= 1.1
        npts = 16
        x = pts[:npts,0]
        y = pts[:npts,1]
        z = pts[:npts,2]
        ax.scatter(x,y,z, color='black', s=12, depthshade=False)
        x = pts_cen[:npts,0]
        y = pts_cen[:npts,1]
        z = pts_cen[:npts,2]
        ax.scatter(x,y,z, color='red', s=12, depthshade=False)


        ax.set(**axkwargs)

    plot_sphere_flag = True
    plot_sphere_flag = False 
    if plot_sphere_flag:
        fig_sphere = plt.figure()
        ax = fig_sphere.add_subplot(111, projection='3d')
        ax.scatter(pts_sphere[:,0], pts_sphere[:,1], pts_sphere[:,2], color='black', s=12, depthshade=False)
        ax.scatter(pts_sphere_centers[:,0], pts_sphere_centers[:,1], pts_sphere_centers[:,2], color='red', s=12, depthshade=False)
        plot_sphere(ax, center=(0,0,0), radius=0.95, color='cyan', alpha=1)

        ax.set(**axkwargs)
    
    plt.show(block=False)

    input("Press Enter to close plots...")
    plt.close('all')