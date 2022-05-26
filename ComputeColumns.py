"""
Post-process effective column densities from GIZMO snapshot data, storing it in the field PartType0/SigmaEff in the snapshot.

Effective column density is defined as the column density that gives the spherically-averaged extinction, 

sigma_eff = \log( 1/(4\pi) \int d\Omega exp(-\Sigma(\Omega)) )

Columns are computed via raytracing along an arbitrary number of directions."""

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from sys import argv
import multiprocessing

ncores = 1
# UNCOMMENT THIS IF YOU WANT PARALLEL
# ncores = multiprocessing.cpu_count() 
nrays = 6 # raytrace 6 different directions


def RandomNormal(num=1):
    dir = np.random.normal(size=(num,3))
    dir /= np.sqrt(np.sum(dir**2,axis=1))[:,np.newaxis]
    return dir

def RandomOrthonormalBasis():
    n1, n2 = RandomNormal(2)
    n2 -= np.dot(n1,n2) * n1
    n2 /= np.sqrt(np.sum(n2**2))
    n3 = np.cross(n1, n2)
    return n1, n2, n3

def OrthonormalBasis(n1):
    n2 = RandomNormal()[0]
    n2 -= np.dot(n1,n2) * n1
    n2 /= np.sqrt(np.sum(n2**2))
    n3 = np.cross(n1, n2)
    return n1, n2, n3

def TransformToBasis(x, basis):
    return np.inner(x, basis)

def ComputeColumns(i, xb, sigmas, K, z, tree3d):
    ids = tree3d.query_ball_point(np.array([xb[i,0], xb[i,1], 0.]), K) # particles within distance K in the fake 3D space will be within their actual radius in the 2D space
    upper = z[ids] > z[i] # stuff in front
    lower = np.invert(upper) # stuff behind
    sigmas_i = sigmas[ids]
    return sigmas_i[upper].sum(), sigmas_i[lower].sum()

def ColumnsAlongAxis(m, x, rho, axis=np.array([0,0,1.])):
    """Computes column density of each particle integrated to infinity in both directions in an array of shape (N,2). 
    
    Does the column summation by searching for overlapping particles and summing their surface densities.
    CAN BE SLOW - BENCHMARK FOR YOUR PROBLEM, YMMV """
    vol = m/rho
    r_eff = (3*vol/(4*np.pi))**(1./3)
    #r_eff *= 32**(1./3)
    sigmas = m/(np.pi*r_eff**2)
    
    xb = TransformToBasis(x, OrthonormalBasis(axis))
    
    #x2d = xb[:,:2]
    z = xb[:,2]

    K = r_eff.max() * (1+1e-6) # largest possible particle radius
    w = np.sqrt(K**2 - r_eff**2) # this is the fake z coordinate in the 3D space we're embedding the tree in

    x3d = np.c_[xb[:,0],xb[:,1],w] # construct the fake 3D coordinates
    tree3d = cKDTree(x3d, leafsize=64) # construct the fake 3D tree

    columns = np.array([ComputeColumns(i,xb, sigmas, K, z, tree3d) for i in range(len(xb))])
    return columns

def ColumnsAlongAxis2(m, x, rho, axis=np.array([0,0,1.]), rmax=1e100):
    """Computes column density of each particle integrated to infinity in both directions in an array of shape (N,2)
    Does the column summation by searching for particles that the current particle overlaps and adding the current particle's sigma to those particles' columns.

    Empirically faster worst-case performance than ColumnsAlongAxis; YMMV.
    """
    vol = m/rho
    r_eff = (3*vol/(4*np.pi))**(1./3)

    r_eff = np.clip(r_eff, 0, rmax)
    sigmas = m/(np.pi*r_eff**2)
    
    xb = TransformToBasis(x, OrthonormalBasis(axis))
    
    z = xb[:,2]

    x2d = xb[:,:2]
    tree2d = cKDTree(xb[:,:2])
    columns = sigmas[:,np.newaxis]/2 * np.ones((len(xb),2))
    for i in range(len(xb)):
        ngb = np.array(tree2d.query_ball_point(x2d[i], r_eff[i]))
        ngb = ngb[ngb != i]
        zngb = xb[ngb,2]-xb[i,2]

        upper = zngb < 0 #z[i] > xngb[:,2]
        lower = np.invert(upper)
        columns[ngb[upper],0] += sigmas[i]
        columns[ngb[lower],1] += sigmas[i]
    return columns

for f in argv[1:]:
    print(f)
    F = h5py.File(f)

    x = np.array(F["PartType0"]["Coordinates"])
    m = np.array(F["PartType0"]["Masses"])
    rho = np.array(F["PartType0"]["Density"])

    axes = RandomNormal(nrays)
    if ncores > 1:
        columns = np.array([Parallel(n_jobs=ncores)(delayed(ColumnsAlongAxis2)(m, x, rho,axis=a) for a in axes)])[0]
    else:
        columns = np.array([ColumnsAlongAxis2(m, x, rho,axis=a) for a in axes])

    ext = np.exp(-columns.reshape(len(x),2*len(axes)))
    sigma_eff = -np.log(np.average(ext,axis=1))
    if "SigmaEff" in F["PartType0"].keys():
        F["PartType0/SigmaEff"][...] = sigma_eff
    else:
        F["PartType0"].create_dataset("SigmaEff", data=sigma_eff)
    F.close()
