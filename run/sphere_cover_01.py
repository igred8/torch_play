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

def plot_sphere(ax, center=(0,0,0), radius=1.0, color='cyan', alpha=1, resolution=130):
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
    ndivs = 2
    out_all_triangles = False
    tris = sc.multi_subdivide_triangle(xyz, ndivs=ndivs, output_all=out_all_triangles)
    
    polys = sc.get_faces_icosa()
    polydivs = []
    for p in polys:
        polydivs += sc.multi_subdivide_triangle(p, ndivs=ndivs, output_all=out_all_triangles)

    pts_sphere, pts = sc.scale_to_sphere(polydivs)
    print(f'# of vertices = {pts.shape[0]}')
    # print(pts_sphere.shape)

    centers = sc.gen_centers(polydivs)
    pts_sphere_centers, pts_cen = sc.scale_to_sphere(centers)
    print(f'# of centers = {pts_cen.shape[0]}')
    
    xgold, ygold, zgold = sc.gen_golden_spiral(pts_cen.shape[0])
    pts_sphere_golden = np.vstack([xgold, ygold, zgold]).T

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


    print(f"# of base faces {len(polys)}")

    

    coverage_verts = sc.calc_closest_distribution(pts, n_close=5)
    coverage_cens = sc.calc_closest_distribution(pts_cen, n_close=5)
    coverage_gold = sc.calc_closest_distribution(pts_sphere_golden, n_close=5)

    nstart = 0
    nend = -1
    nstep = 1

    plot_polyhedron_flag = True
    plot_polyhedron_flag = False 
    if plot_polyhedron_flag:
        fig_polyhedron = plt.figure()
        ax = fig_polyhedron.add_subplot(111, projection='3d')
        cmap = plt.colormaps['viridis']
        for i,poly in enumerate(polys):
            # color_ = cmap(i/19)
            color_ = 'cyan'
            collection = Poly3DCollection([poly], alpha=1, facecolor=color_, edgecolor='black')
            ax.add_collection3d(collection)
        
        # collection = Poly3DCollection(polydivs, alpha=1, facecolor='none', edgecolor='black')
        # ax.add_collection3d(collection)

        # collection = Poly3DCollection(tris_placed, alpha=0.9, facecolor='none', edgecolor='black')
        # ax.add_collection3d(collection)
        pts *= 1.3
        pts_cen *= 1.21
        
        x = pts[nstart:nend:nstep,0]
        y = pts[nstart:nend:nstep,1]
        z = pts[nstart:nend:nstep,2]
        ax.scatter(x,y,z, color='tab:blue', s=12, depthshade=False)
        x = pts_cen[nstart:nend:nstep,0]
        y = pts_cen[nstart:nend:nstep,1]
        z = pts_cen[nstart:nend:nstep,2]
        ax.scatter(x,y,z, color='tab:green', s=12, depthshade=False)


        ax.set(**axkwargs)


    
    x = pts[nstart:nend:nstep, 0]
    y = pts[nstart:nend:nstep, 1]
    z = pts[nstart:nend:nstep, 2]
    r, th, phi = sc.xyz_to_spherical(x, y, z)
    x = pts_cen[nstart:nend:nstep, 0]
    y = pts_cen[nstart:nend:nstep, 1]
    z = pts_cen[nstart:nend:nstep, 2]
    rc, thc, phic = sc.xyz_to_spherical(x, y, z)
    rg, thg, phig = sc.xyz_to_spherical(xgold, ygold, zgold)

    plot_angles_flag = True
    # plot_angles_flag = False 
    if plot_angles_flag:
        fig_ang, axs = plt.subplots(3,1, figsize=[8,8])
        axs[0].scatter(phi, th, c='tab:blue', s=12)
        axs[1].scatter(phic, thc, c='tab:green', s=15, marker='x')
        axs[2].scatter(phig, thg, c='tab:orange', s=15, marker='d')

        axkwargs = {
            'xlim':[0, 2*pi],
            'ylim':[0, pi],
            'xlabel': 'phi',
            'ylabel': 'theta',
            # 'zlabel': 'z',
            'aspect': 'equal',
        }
        for ax in axs:
            ax.set(**axkwargs)


    plot_sphere_flag = True
    # plot_sphere_flag = False 
    if plot_sphere_flag:
        fig_sphere = plt.figure()
        ax = fig_sphere.add_subplot(111, projection='3d')
        ax.scatter(pts_sphere[:,0], pts_sphere[:,1], pts_sphere[:,2], color='tab:blue', s=12, depthshade=True)
        ax.scatter(pts_sphere_centers[:,0], pts_sphere_centers[:,1], pts_sphere_centers[:,2], color='tab:green', s=12, marker='x', depthshade=True)
        ax.scatter(pts_sphere_golden[:,0], pts_sphere_golden[:,1], pts_sphere_golden[:,2], color='tab:orange', s=12, marker='d', depthshade=True)
        plot_sphere(ax, center=(0,0,0), radius=0.95, color='grey', alpha=1)
        axkwargs = {
            
            'xlabel': 'x',
            'ylabel': 'y',
            'zlabel': 'z',
            'aspect': 'equal',
        }
        ax.set(**axkwargs)
    
    plot_covhist_flag = True
    # plot_covhist_flag = False 
    if plot_covhist_flag:
        fig, ax = plt.subplots(1,1)
        bins = np.arange(0,0.5,0.05)
        ax.hist(coverage_verts, bins=bins, color="tab:blue", alpha=0.7)
        ax.hist(coverage_cens, bins=bins, color="tab:green", alpha=0.7)
        ax.hist(coverage_gold, bins=bins, color="tab:orange", alpha=0.7)


    plt.show(block=False)

    input("Press Enter to close plots...")
    plt.close('all')