import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import generic_filter
from scipy.spatial.distance import cdist
from scipy.optimize import minimize, least_squares, LinearConstraint
from sklearn.cluster import KMeans
import pandas as pd
from skimage import measure, color
from matplotlib import cm, patches, colors
import matplotlib
from src.segment import Segmenter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings


def sx_grad(p1, p2, q1x, q1y, q2x, q2y, rx, ry):
    """

    Computes part of the jacobian of the energy function (the partial derivative with respect to x) analytically
    to improve the speed and accuracy of initial optimisation.

    """
    t3 = p1-p2
    t5 = p1*q1x
    t6 = p2*q2x
    t7 = rx*t3
    t2 = -t5+t6+t7
    t9 = p1*q1y
    t10 = p2*q2y
    t11 = ry*t3
    t4 = -t9+t10+t11
    t8 = np.power(t2,2)
    t12 = np.power(t4,2)
    t13 = t8+t12
    t14 = np.divide(1,np.power(t13, 1.5))
    t15 = np.divide(1,np.sqrt(t13))
    t16 = q1x-rx
    t17 = q2x-rx

    Dx = np.array([p1*t15-p1*t8*t14,-p1*t2*t4*t14,
                   t15*t16-t2*t14*(t2*t16*2+t4*(q1y-ry)*2)*(0.5),
                   -p2*t15+p2*t8*t14,p2*t2*t4*t14,
                   -t15*t17+t2*t14*(t2*t17*2+t4*(q2y-ry)*2)*(0.5)]).T
    return Dx

def sy_grad(p1, p2, q1x, q1y, q2x, q2y, rx, ry):
    """

    Computes part of the jacobian of the energy function (the partial derivative with respect to y) analytically
    to improve the speed and accuracy of initial optimisation.

    """
    t3 = p1-p2
    t5 = p1*q1x
    t6 = p2*q2x
    t7 = rx*t3
    t2 = -t5+t6+t7
    t8 = p1*q1y
    t9 = p2*q2y
    t10 = ry*t3
    t4 = -t8+t9+t10
    t11 = np.power(t2,2)
    t12 = np.power(t4,2)
    t13 = t11+t12
    t14 = np.divide(1,np.power(t13,1.5))
    t15 = np.divide(1,np.sqrt(t13))
    t16 = q1y-ry
    t17 = q2y-ry
    Dy = np.array([-p1*t2*t4*t14,p1*t15-p1*t12*t14,
                   t15*t16-t4*t14*(t4*t16*2+t2*(q1x-rx)*2)*(0.5),
                   p2*t2*t4*t14,-p2*t15+p2*t12*t14,
                   -t15*t17+t4*t14*(t4*t17*2+t2*(q2x-rx)*2)*(0.5)]).T
    return Dy

def radius_grad_theta(p1,p2,q1x,q1y,q2x,q2y,t1,t2):
    """

    Computes the jacobian of the energy function (the partial derivative with respect to x) analytically to improve the
    speed and accuracy of theta optimisation.

    """
    t4 = p1-p2
    t5 = q1x-q2x
    t6 = q1y-q2y
    t7 = np.divide(1,np.power(t4,2))
    t8 = t1-t2
    t9 = t4*t8
    t10 = np.power(t5,2)
    t11 = np.power(t6,2)
    t12 = t10+t11
    t13 = t9-p1*p2*t12
    t14 = np.divide(1,np.sqrt(-t7*t13))
    t15 = np.divide(1,t4)
    Dr = np.array([t14*t15*(-0.5),t14*t15*(0.5)]).T
    return Dr

def rx_grad(p1, p2, q1x, q2x, rx):
    """

    Computes part of the jacobian of the nonlinear constraint function (the partial derivative with respect to x)
    analytically to improve the speed and accuracy of initial optimisation.

    """
    Dx = np.array([p1, np.zeros(p1.shape), q1x-rx, -p2, np.zeros(p1.shape), -q2x+rx]).T
    return Dx

def ry_grad(p1, p2, q1y, q2y, ry):
    """

    Computes part of the jacobian of the nonlinear constraint function (the partial derivative with respect to y)
    analytically to improve the speed and accuracy of initial optimisation.

    """
    Dy = np.array([np.zeros(p1.shape),p1,q1y-ry,np.zeros(p1.shape),-p2,-q2y+ry]).T
    return Dy

def rho_x_grad(p1,p2,q1x,q2x):
    """

    Computes part of the jacobian of the energy function analytically to improve the speed and accuracy of
    the main optimisation step.

    """
    t2 = p1-p2
    t3 = np.divide(1,t2)
    t4 = np.divide(1,np.power(t2,2))
    t5 = p1*q1x
    t6 = t5-p2*q2x
    z = np.zeros(p1.shape)
    dRhoX = np.array([p1*t3,z,z,q1x*t3-t4*t6,-p2*t3,z,z,-q2x*t3+t4*t6]).T
    return dRhoX

def rho_y_grad(p1,p2,q1y,q2y):
    """

    Computes part of the jacobian of the energy function analytically to improve the speed and accuracy of
    the main optimisation step.

    """
    t2 = p1-p2
    t3 = np.divide(1,t2)
    t4 = np.divide(1,np.power(t2,2))
    t5 = p1*q1y
    t6 = t5-p2*q2y
    z = np.zeros(p1.shape)
    dRhoY = np.array([z,p1*t3,z,q1y*t3-t4*t6,z,-p2*t3,z,-q2y*t3+t4*t6]).T
    return dRhoY

def radius_grad(p1,p2,q1x,q1y,q2x,q2y,t1,t2):
    """

    Computes part of the jacobian of the energy function analytically to improve the speed and accuracy of
    the main optimisation step.

    """
    t4 = p1-p2
    t5 = q1x-q2x
    t6 = q1y-q2y
    t7 = np.divide(1,np.power(t4,2))
    t8 = t1-t2
    t9 = t4*t8
    t10 = np.power(t5,2)
    t11 = np.power(t6,2)
    t12 = t10+t11
    t15 = p1*p2*t12
    t13 = t9-t15
    t16 = t7*t13
    t14 = np.divide(1,np.sqrt(-t16))
    t17 = q1x*2
    t18 = q2x*2
    t19 = t17-t18
    t20 = p1*p2*t7*t14*t19*(0.5)
    t21 = q1y*2
    t22 = q2y*2
    t23 = t21-t22
    t24 = p1*p2*t7*t14*t23*(0.5)
    t25 = np.divide(1,t4)
    t26 = np.divide(1,np.power(t4,3))
    t27 = t13*t26*2
    dR = np.array([t20,t24,t14*t25*(-0.5),
                   t14*(t27+t7*(-t1+t2+p2*t12))*(0.5),
                   -t20,-t24,t14*t25*(0.5),
                   t14*(t27-t7*(t1-t2+p1*t12))*(-0.5)]).T
    return dR

class VMSI():

    def __init__(self, vertices, cells, edges, width, height, verbose, optimiser='nlopt'):
        self.vertices = vertices
        self.cells = cells
        self.edges = edges
        self.width = width
        self.height = height
        self.verbose = verbose
        self.optimiser = optimiser

        # Mark fourfold vertices
        self.vertices['fourfold'] = [(np.shape(nverts)[0] != 3) for nverts in self.vertices['nverts']]

        # Initialize new columns
        self.edges['radius'] = np.zeros(len(self.edges))
        self.edges['rho'] = tuple((0,0) for _ in range(len(self.edges)))
        self.edges['fitenergy'] = np.zeros(len(self.edges))
        self.edges['tension'] = np.zeros(len(self.edges))
        self.cells['pressure'] = np.zeros(len(self.cells))
        self.cells['qx'] = np.zeros(len(self.cells))
        self.cells['qy'] = np.zeros(len(self.cells))
        self.cells['theta'] = np.zeros(len(self.cells))
        self.cells['stress'] = [np.array([0,0,0]) for _ in range(self.cells.shape[0])]

        # Initialize attributes
        self.dV = None
        self.dC = None
        self.involved_cells = None
        self.involved_vertices = None
        self.involved_edges = None
        self.bulk_cells = None
        self.bulk_vertices = None
        self.ext_cells = None
        self.ext_vertices = None
        self.cell_pairs = None
        self.edgearc_x = None
        self.edgearc_y = None
        self.avg_edge_length = None

        if self.optimiser == 'matlab':
            import matlab
            import matlab.engine
            import pathlib
            src_path = str(pathlib.Path(__file__).parent.resolve())

            # Initialise Matlab engine
            self.eng = matlab.engine.start_matlab()
            self.eng.cd(src_path)
        elif self.optimiser == 'nlopt':
            import nlopt


    def fit_circle(self):
        """

        Fit circle to each edge
        If edge is too flat, fit line instead

        """
        for i in range(len(self.edges)):
            r1 = np.array(self.vertices['coords'][self.edges['verts'][i][0]])
            r2 = np.array(self.vertices['coords'][self.edges['verts'][i][1]])

            edge_pixels = [np.unravel_index(pixel, (self.width, self.height)) for pixel in self.edges['pixels'][i]]

            nB = np.matmul(np.array([[0, 1], [-1, 0]]), r1 - r2)
            D = np.sqrt(np.sum(np.power(nB, 2)))
            nB = np.divide(nB, D)
            x0 = 0.5*(r1 + r2)

            delta = np.subtract(edge_pixels, x0)
            IP = (delta[:,0] * nB[0]) + (delta[:,1] * nB[1])
            L0 = D/2

            A = 2*np.sum(np.power(IP, 2))
            B = np.sum((np.sum(np.power(delta, 2), axis=1) - np.power(L0, 2)) * IP)
            y0 = np.divide(B, A)

            # Minimise the MSQ between edge pixels and fitted arc
            def energyfunc(x):
                return np.mean(np.power(np.sqrt(np.sum(np.power(delta-(x*nB), 2), axis=1)) - np.sqrt(np.power(x, 2) + np.power(L0, 2)), 2))

            if not np.isnan(y0):
                res = minimize(energyfunc, y0, tol=1e-8)
            else:
                res = minimize(energyfunc, 0, tol=1e-8)
            y = res.x
            E = res.fun

            linedistance = np.mean(np.power(IP, 2))
            # Store the radius, centre and fit energy of each fitted circular arc or straight line
            if (E < linedistance and len(edge_pixels) > 3):
                self.edges.at[i,'radius'] = np.sqrt(np.power(y, 2) + np.power(L0, 2))
                self.edges.at[i,'rho'] = x0 + (y * nB)
                self.edges.at[i,'fitenergy'] = E
            else:
                self.edges.at[i,'radius'] = np.Inf
                self.edges.at[i,'rho'] = np.array([np.Inf, np.Inf])
                self.edges.at[i,'fitenergy'] = linedistance
        return


    def remove_fourfold(self):
        """

        recursively removes fourfold (or greater) vertices by splitting vertex apart in direction of greatest variance

        """

        for v in range(len(self.vertices)):
            if (np.shape(self.vertices['nverts'][v])[0] > 3  and not (0 in self.vertices['ncells'][v])):
                while (np.shape(self.vertices['nverts'][v])[0] > 3):
                    num_v = len(self.vertices)
                    num_e = len(self.edges)

                    nverts = self.vertices['nverts'][v]
                    nedges = self.vertices['edges'][v]
                    ncells = self.vertices['ncells'][v]

                    R = np.array([self.vertices['coords'][vert] for vert in nverts])
                    rV = np.array(self.vertices['coords'][v])

                    R = R - np.mean(R, axis=0)
                    I = np.matmul(R.T,R)

                    W, V = np.linalg.eig(I)
                    direction = V[:,np.argmax(W)]

                    # create two new vertices, positive and negative
                    rV1 = rV + (direction/2)
                    rV2 = rV - (direction/2)

                    # set positive neighbour vertices to the 2 vertices closest to the direction of vertex movement
                    # all other neighbour vertices are negative
                    indices = np.argsort(np.dot(R, direction))[-2:]
                    pos_verts = np.zeros_like(nverts)
                    pos_verts[indices] = 1
                    neg_verts = 1 - pos_verts

                    # change vertex with current index to negative vertex
                    self.vertices.at[v,'coords'] = rV2.tolist()
                    self.vertices.at[v,'nverts'] = np.concatenate((nverts[neg_verts.astype('bool')], np.array([num_v])))
                    self.vertices.at[v,'fourfold'] = (np.shape(self.vertices['nverts'][v])[0] > 3)

                    neg_cells = ncells[[(sum(np.isin(nverts[neg_verts.astype('bool')], self.cells['nverts'][cell]))==2) for cell in ncells]]

                    # add positive vertex
                    self.vertices = self.vertices.append({'coords':[0,0],'ncells':np.array([]),'nverts':np.array([]),'edges':np.array([])}, ignore_index=True)
                    self.vertices.at[num_v,'coords'] = rV1.tolist()
                    self.vertices.at[num_v,'nverts'] = np.concatenate((nverts[pos_verts.astype('bool')], np.array([v])))
                    self.vertices.at[num_v,'fourfold'] = False

                    pos_cell = ncells[[(sum(np.isin(nverts[pos_verts.astype('bool')], self.cells['nverts'][cell]))==2) for cell in ncells]]

                    # update new positive vertex index for neighbour vertices
                    for vert in nverts[pos_verts.astype('bool')]:
                        self.vertices.at[vert, 'nverts'][self.vertices['nverts'][vert] == v] = num_v

                    joint_cells = ncells[np.invert(np.isin(ncells, np.array([pos_cell, neg_cells])))]

                    self.vertices.at[v,'ncells'] = np.concatenate((joint_cells, neg_cells))
                    self.vertices.at[num_v,'ncells'] = np.concatenate((joint_cells, pos_cell))

                    # update current edges
                    # this requires edges to be in the same order as vertices
                    neg_edges = nedges[neg_verts.astype('bool')]
                    pos_edges = nedges[pos_verts.astype('bool')]

                    self.edges.at[pos_edges[0], 'verts'][self.edges.at[pos_edges[0], 'verts'] == v] = num_v
                    self.edges.at[pos_edges[1], 'verts'][self.edges.at[pos_edges[1], 'verts'] == v] = num_v

                    # create new edge between new vertices
                    # edge is only one pixel long so no need to add pixels
                    self.edges = self.edges.append({'pixels':np.array([]),'verts':np.array([]),'cells':np.array([]),
                                                    'radius':np.array([]), 'rho':np.array([]), 'fitenergy':np.Inf, 'tension':float(0)},ignore_index=True)

                    self.edges.at[num_e,'verts'] = np.array([v, num_v])
                    self.edges.at[num_e,'cells'] = joint_cells
                    self.edges.at[num_e,'pixels'] = np.array([])
                    self.edges.at[num_e,'radius'] = np.Inf
                    self.edges.at[num_e,'rho'] = np.array([np.Inf, np.Inf])

                    # update edges of new vertices
                    self.vertices.at[v,'edges'] = np.concatenate((neg_edges, np.array([num_e])))
                    self.vertices.at[num_v,'edges'] = np.concatenate((pos_edges, np.array([num_e])))

                    # update cells

                    # update pos cells
                    for cell in pos_cell:
                        self.cells.at[cell,'nverts'][self.cells['nverts'][cell] == v] = num_v
                        self.cells.at[cell,'ncells'] = self.cells.at[cell, 'ncells'][np.isin(self.cells.at[cell,'ncells'], neg_cells, invert=True)]
                    # update neg cells
                    for cell in neg_cells:
                        self.cells.at[cell,'ncells'] = self.cells.at[cell, 'ncells'][np.isin(self.cells.at[cell,'ncells'], pos_cell, invert=True)]
                    # update joint cells
                    for cell in joint_cells:
                        self.cells.at[cell,'nverts'] = np.concatenate((self.cells.at[cell,'nverts'], np.array([num_v])))
                        self.cells.at[cell,'numv'] = self.cells.at[cell,'numv']+1
        return


    def make_convex(self):
        """

        remove concave vertices by moving vertex to ensure all angles < pi

        """

        # find boundary vertices
        boundary_cells = np.where(self.cells['holes'].to_numpy())[0]
        boundary_verts = np.unique(np.concatenate(self.cells.loc[boundary_cells,'nverts'].tolist()))

        # Iterate through all non-boundary verts
        for v in range(len(self.vertices)):
            if (v not in boundary_verts and len(self.vertices.at[v,'nverts']) == 3):

                rv = np.array(self.vertices['coords'][v])
                nverts = np.array(self.vertices['nverts'][v])

                n = np.array([self.vertices['coords'][nverts[0]],
                              self.vertices['coords'][nverts[1]],
                              self.vertices['coords'][nverts[2]]])

                n_centered = n - np.mean(n, axis=0)
                theta = np.mod(np.arctan2(n_centered[:,1], n_centered[:,0]), 2*np.pi)
                n = n[np.argsort(theta),:]

                r = n - rv
                r = np.divide(r.T,(np.linalg.norm(r, axis=1))).T

                z12 = np.cross(np.concatenate((r[0,:], np.array([0]))), np.concatenate((r[1,:], np.array([0]))))
                z23 = np.cross(np.concatenate((r[1,:], np.array([0]))), np.concatenate((r[2,:], np.array([0]))))
                z31 = np.cross(np.concatenate((r[2,:], np.array([0]))), np.concatenate((r[0,:], np.array([0]))))

                # Determine angle between all pairs of edges
                theta12 = np.mod(np.arctan2(z12[2], np.dot(r[0,:], r[1,:])), 2*np.pi)
                theta23 = np.mod(np.arctan2(z23[2], np.dot(r[1,:], r[2,:])), 2*np.pi)
                theta31 = np.mod(np.arctan2(z31[2], np.dot(r[2,:], r[0,:])), 2*np.pi)

                # If any theta>pi, move vertex until it's not
                # This accounts for measurement noise in junctions where one angle is close to pi
                if theta12 > np.pi:
                    deltaR = np.dot(n[0,:]-rv, np.matmul(np.array([[0,-1],[1,0]]), n[0,:]-n[1,:])) / np.dot(n[2,:]-rv, np.matmul(np.array([[0,-1],[1,0]]), n[0,:]-n[1,:]))
                    nrv = rv + 1.5*deltaR*(n[2,:]-rv)
                elif theta23 > np.pi:
                    deltaR = np.dot(n[1,:]-rv, np.matmul(np.array([[0,-1],[1,0]]), n[1,:]-n[2,:])) / np.dot(n[0,:]-rv, np.matmul(np.array([[0,-1],[1,0]]), n[1,:]-n[2,:]))
                    nrv = rv + 1.5*deltaR*(n[0,:]-rv)
                elif theta31 > np.pi:
                    deltaR = np.dot(n[2,:]-rv, np.matmul(np.array([[0,-1],[1,0]]), n[2,:]-n[0,:])) / np.dot(n[1,:]-rv, np.matmul(np.array([[0,-1],[1,0]]), n[2,:]-n[0,:]))
                    nrv = rv + 1.5*deltaR*(n[1,:]-rv)
                elif theta12 == np.pi:
                    nrv = rv + 0.5*r[2,:]
                elif theta23 == np.pi:
                    nrv = rv + 0.5*r[0,:]
                elif theta31 == np.pi:
                    nrv = rv + 0.5*r[1,:]
                else:
                    nrv = rv

                self.vertices.at[v,'coords'] = nrv.tolist()
        return


    def prepare_data(self):
        """

        Prepare data for tension inference:
        Remove fourfold vertices, fit circles, make vertices convex

        """

        # Fit circular arcs to each edge
        self.fit_circle()

        # Recursively remove n-fold vertices (for n>3) by splitting and moving resulting (n-1)-fold vertices apart
        self.remove_fourfold()

        # Inference cannot handle concave vertices (with one angle greater than pi) so remove these
        self.make_convex()

        return


    def classify_cells(self):
        """

        Determine which cells are involved in tension inference
        Initialize q as cell centroids

        """

        self.bulk_cells = np.array(range(0,len(self.cells)))

        boundary_cells = np.unique(np.concatenate([self.cells.at[0, 'ncells'], np.where(self.cells['holes'].to_numpy()[1:])[0]]))

        # Remove boundary cells and cells surrounded by boundary cells from bulk cells
#        all_threefold = np.array([not(any(self.vertices.loc[self.cells.at[cell, 'nverts'], 'fourfold'].to_numpy())) for cell in range(len(self.cells))])
#        self.bulk_cells = self.bulk_cells[all_threefold]
        self.bulk_cells = self.bulk_cells[np.isin(self.bulk_cells, boundary_cells, invert=True)]
        bad_cells = np.array([])
        for cell in self.bulk_cells:
            if np.sum(np.isin(self.cells.at[cell, 'ncells'], self.bulk_cells)) == 0:
                bad_cells = np.append(bad_cells, cell)
        self.bulk_cells = self.bulk_cells[np.isin(self.bulk_cells, bad_cells, invert=True)]

        # This excludes vertices surrounded by boundary cells; edges at these vertices are not constrained enough for accurate inference
        self.bulk_vertices = np.unique(np.concatenate([self.cells.at[cell, 'nverts'] for cell in self.bulk_cells]))

        self.involved_cells = np.unique(np.concatenate([self.vertices.at[vert, 'ncells'] for vert in self.bulk_vertices]))
        self.ext_cells = self.involved_cells[np.isin(self.involved_cells, self.bulk_cells, invert=True)]
        self.involved_cells = np.concatenate((self.bulk_cells, self.ext_cells))

        self.involved_vertices = np.unique(np.concatenate([self.vertices.at[vert, 'nverts'] for vert in self.bulk_vertices]))
        self.ext_vertices = self.involved_vertices[np.isin(self.involved_vertices, self.bulk_vertices, invert=True)]
        self.involved_vertices = np.concatenate((self.bulk_vertices, self.ext_vertices))

        x0 = np.vstack([np.stack(self.cells['centroids'][self.involved_cells]).T,
                        np.zeros(len(self.involved_cells))]).T

        return x0


    def build_diff_operators(self):
        """

        Compute difference operators to enable vectorized operations

        """
        # Build cell adjacency matrix
        adj_mat = np.zeros((len(self.involved_cells), len(self.involved_cells)))
        num_edges = 0
        edge_cells = self.edges.cells.to_list()

        for i in range(len(self.involved_cells)):
            cell = self.involved_cells[i]
            for ncell in self.cells.at[cell, 'ncells']:
                j = np.ravel(np.where(self.involved_cells == ncell))
                # Ensure that edge between neighbouring cells actually exists
                if j.size > 0 and adj_mat[i, j] == 0 and np.any(np.all(np.sort(np.array([cell, ncell])) == edge_cells, axis=1)):
                    adj_mat[i, j] = 1
                    adj_mat[j, i] = 1
                    num_edges += 1

        # Compute difference operators
        self.dC = np.zeros((num_edges, len(self.involved_cells)))
        self.dV = np.zeros((num_edges, len(self.involved_vertices)))
        self.cell_pairs = np.zeros((num_edges, 2), dtype=int)

        diff_index = 0
        for i in range(len(self.involved_cells)):
            ncells = np.ravel(np.where(adj_mat[i,:] == 1))
            ncells = ncells[ncells > i]

            for cell in ncells:
                self.dC[diff_index, i] = 1
                self.dC[diff_index, cell] = -1
                self.cell_pairs[diff_index] = np.array([i, cell])

                verts = np.intersect1d(self.cells['nverts'][self.involved_cells[i]], self.cells['nverts'][self.involved_cells[cell]])

                if (len(verts) == 2):
                    self.dV[diff_index, np.where(self.involved_vertices == verts[0])] = 1
                    self.dV[diff_index, np.where(self.involved_vertices == verts[1])] = -1

                diff_index += 1

        # Check for bad vertices and edges
        bad_verts = np.invert(np.sum(np.abs(self.dV), axis=0) == 0)
        self.dV = self.dV[:,bad_verts]
        self.involved_vertices = self.involved_vertices[bad_verts]

        bad_edges = np.invert(np.sum(np.abs(self.dV), axis=1) < 2)
        self.dV = self.dV[bad_edges,:]
        self.dC = self.dC[bad_edges,:]
        self.cell_pairs = self.cell_pairs[bad_edges,:]
        return


    def estimate_tau(self):
        """

        Estimate tension vector tau, which is tangent to the edge

        """

        self.build_diff_operators()

        self.involved_edges = -1 * np.ones(self.dC.shape[0], dtype=int)

        for i in range(len(self.edges)):
            edge_cells = self.edges.at[i, 'cells']
            idx = np.where((self.dC[:,np.where(edge_cells[0]==self.involved_cells)[0]] != 0) & (self.dC[:,np.where(edge_cells[1]==self.involved_cells)[0]] != 0))[0]
            self.involved_edges[idx] = i

        # initialise variables
        tau_1 = np.zeros((self.dV.shape[0], 2))
        tau_2 = np.zeros((self.dV.shape[0], 2))

        v_coords = np.concatenate([self.vertices['coords'][self.involved_vertices].tolist()])

        e_chord = np.matmul(self.dV, v_coords)

        e_cells = np.zeros((self.dV.shape[0], 2), dtype=int)
        r1 = np.zeros((self.dV.shape[0], 2))
        r2 = np.zeros((self.dV.shape[0], 2))

        for e in range(self.dV.shape[0]):

            e_verts = np.ravel([np.where(self.dV[e,:] == 1), np.where(self.dV[e,:] == -1)])
            e_cells[e,:] = np.ravel([np.where(self.dC[e,:] == 1), np.where(self.dC[e,:] == -1)])

            r1[e,:] = v_coords[e_verts[0],:]
            r2[e,:] = v_coords[e_verts[1],:]

            # If edge is curved, rotate t by pi/2 to get tau
            # t is a vector between the centre of curvature and vertex
            if self.involved_edges[e] >= 0 and self.edges.at[self.involved_edges[e], 'radius'] < np.inf:
                rho = self.edges.at[self.involved_edges[e], 'rho']

                t1 = np.divide(r1[e,:] - rho, np.linalg.norm(r1[e,:] - rho))
                t2 = np.divide(r2[e,:] - rho, np.linalg.norm(r2[e,:] - rho))

                if np.linalg.det(np.array((t1,t2))) > 0:
                    tau_1[e,:] = np.matmul(np.array([[0,-1],[1,0]]), t1)
                    tau_2[e,:] = np.matmul(np.array([[0,1],[-1,0]]), t2)
                else:
                    tau_1[e,:] = -np.matmul(np.array([[0,1],[-1,0]]), t1)
                    tau_2[e,:] = -np.matmul(np.array([[0,-1],[1,0]]), t2)
            # If edge is straight, simply take the edge vector to get tau
            else:
                tau_1[e,:] = -np.divide(e_chord[e,:], np.linalg.norm(e_chord[e,:]))
                tau_2[e,:] = np.divide(e_chord[e,:], np.linalg.norm(e_chord[e,:]))
        return e_cells, tau_1, tau_2, r1, r2


    def estimate_pressure(self, q, e_cells, tau_1, tau_2, r1, r2):
        """

        Initial estimate of pressure
        This provides the initial values for the initial minimisation step

        :param q: (numpy array) generating points q as defined under the VMSI formulation
        :param e_cells: (numpy array) indexes of cell pairs at each edge
        :param tau_1: (numpy array) unit tension vectors at vertex 1 for each edge
        :param tau_2: (numpy array) unit tension vectors at vertex 2 for each edge
        :param r1: (numpy array) co-ordinates of vertex 1 for each edge
        :param r2: (numpy array) co-ordinates of vertex 2 for each edge

        """

        L1 = np.zeros((e_cells.shape[0], q.shape[0]))
        L2 = np.zeros((e_cells.shape[0], q.shape[0]))

        for i in range(e_cells.shape[0]):
            L1[i, e_cells[i,0]] = np.dot(q[e_cells[i,0],:] - r1[i], tau_1[i])
            L1[i, e_cells[i,1]] = -np.dot(q[e_cells[i,1],:] - r1[i], tau_1[i])
            L2[i, e_cells[i,0]] = np.dot(q[e_cells[i,0],:] - r2[i], tau_2[i])
            L2[i, e_cells[i,1]] = -np.dot(q[e_cells[i,1],:] - r2[i], tau_2[i])

        scale = np.mean(np.linalg.norm(q, axis=1))
        b = np.zeros(2*L1.shape[0] + 1)
        b[-1] = scale

        # Initial estimates generated by minimising sum of
        # p_a((q_a-r_i) * tau_i) = p_b((q_b-r_i) * tau_i) for cells a and b at vertex i
        # Overall scale maintained such that mean pressure = 1

        L = np.vstack((L1, L2, np.array(np.divide(np.ones(q.shape[0]), q.shape[0]))))

        p = np.linalg.lstsq(L,b)[0]
        p = np.divide(p, np.mean(p))
        return p


    def generate_circular_arcs(self):
        """

        Construct circular arc for each edge and use these instead of raw segmented edges for minimization

        """

        # Each arc has a number of pixels equal to the average edge length
        self.avg_edge_length = int(np.median([self.edges.at[edge, 'pixels'].shape[0] for edge in self.involved_edges]))
        self.edgearc_x = np.zeros((len(self.involved_edges), self.avg_edge_length))
        self.edgearc_y = np.zeros((len(self.involved_edges), self.avg_edge_length))

        for i in range(len(self.involved_edges)):
            r = np.array([self.vertices.at[self.edges.at[self.involved_edges[i], 'verts'][0], 'coords'],
                          self.vertices.at[self.edges.at[self.involved_edges[i], 'verts'][1], 'coords']])
            # Construct circular arc for curved edges
            if self.edges.at[self.involved_edges[i], 'radius'] < np.inf:
                r_centered = np.subtract(r, self.edges.at[self.involved_edges[i], 'rho'])

                # This returns between [0, pi] so we should aways get a convex angle as expected
                theta = np.arccos(np.divide(np.dot(r_centered[0,:], r_centered[1,:]),
                                            np.multiply(np.linalg.norm(r_centered[0,:]), np.linalg.norm(r_centered[1,:]))))
                if np.linalg.det(r_centered) < 0:
                    r_centered = r_centered[[1,0],:]
                theta_range = np.linspace(0, theta, self.avg_edge_length)

                self.edgearc_x[i] = self.edges.at[self.involved_edges[i], 'rho'][0] + (r_centered[0,0]*np.cos(theta_range) - r_centered[0,1]*np.sin(theta_range))
                self.edgearc_y[i] = self.edges.at[self.involved_edges[i], 'rho'][1] + (r_centered[0,0]*np.sin(theta_range) + r_centered[0,1]*np.cos(theta_range))
            # Construct straight line for straight edges
            else :
                chord = r[1,:] - r[0,:]

                spacing = np.linspace(0, 1, self.avg_edge_length)

                self.edgearc_x[i] = r[0,0] + spacing*chord[0]
                self.edgearc_y[i] = r[0,1] + spacing*chord[1]
        return


    def estimate_theta(self, x):
        """
        Initialize theta, defined as p_a * z^2_a for each cell a

        :param x: (numpy array) with dimensions [num_cells x 3] containing q in x[0:2,:] and p in x[2,:]
        :return: (numpy array) with dimensions [num_cells x 1] containing initialized values of theta
        """
        q = x[:,0:2]
        p = x[:,2]

        r = np.zeros(len(self.involved_edges))
        r_flat = np.zeros(len(self.involved_edges))
        q_sq = np.sum(np.power(np.matmul(self.dC, q), 2), axis=1)

        rho = np.divide(np.matmul(self.dC, np.multiply(q.T,p).T).T, np.matmul(self.dC, p)).T

        for i in range(len(self.involved_edges)):
            edge = self.involved_edges[i]
            v1 = self.edges.at[edge, 'verts'][0]
            v2 = self.edges.at[edge, 'verts'][1]

            r1 = self.vertices['coords'][v1]
            r2 = self.vertices['coords'][v2]

            r[i] = np.mean(np.power(np.array([np.linalg.norm(r1 - rho[i]), np.linalg.norm(r2 - rho[i])]), 2))
            r_flat[i] = p[np.where(self.dC[i,:] == 1)] * p[np.where(self.dC[i,:] == -1)] * q_sq[i]

        dP = np.matmul(self.dC, p)
        r = np.multiply(r, np.power(dP, 2))

        A = np.multiply(self.dC.T, dP).T
        b = r_flat - r

        theta = np.linalg.lstsq(np.vstack((A, np.ones(A.shape[1]))),np.concatenate((b, np.array([0]))))[0]

        return theta


    def initial_minimization(self):
        """

        Initial minimisation step to ensure that vector t (between q_a and r_i for vertex i at cell a) is orthogonal
        to tension vector tau.
        This prevents the main minimisation step converging on the trivial solution q_a=q_b, p_a=p_b

        """
        # Initialize q
        x0 = self.classify_cells()

        # Initialize tau
        e_cells, tau_1, tau_2, r1, r2 = self.estimate_tau()

        # Initialize pressure
        x0[:,2] = self.estimate_pressure(x0[:,0:2], e_cells, tau_1, tau_2, r1, r2)

        q0 = x0[:,0:2]
        p0 = x0[:,2]

        b0 = np.matmul(self.dC, np.multiply(q0.T,p0).T)
        delta_p0 = np.matmul(self.dC, p0)

        # Get initial values for t_i and t_j
        t1_0 = b0 - (np.multiply(r1.T,delta_p0).T)
        t2_0 = b0 - (np.multiply(r2.T,delta_p0).T)

        if self.optimiser == 'nlopt':
            import nlopt
            scale = 0.5 * (np.mean(np.linalg.norm(t1_0, axis=1)) + np.mean(np.linalg.norm(t2_0, axis=1)))

            # Define energy function for initial minimisation
            def energy(x, grad=np.array([])):
                np.savetxt('./.init_opt.csv', x, delimiter=',',fmt='%f')
                x = x.reshape(x0.shape, order='F')
                q = x[:,0:2]
                p = x[:,2]

                b = np.matmul(self.dC, np.multiply(q.T,p).T)
                delta_p = np.matmul(self.dC, p)

                t1 = np.divide((b - (np.multiply(r1.T,delta_p).T)).T,np.linalg.norm(b - (np.multiply(r1.T,delta_p).T), axis=1)).T
                t2 = np.divide((b - (np.multiply(r2.T,delta_p).T)).T,np.linalg.norm(b - (np.multiply(r2.T,delta_p).T), axis=1)).T

                E = 0.5 * np.mean(np.power(np.sum(t1 * tau_1, axis=1), 2) +
                                  np.power(np.sum(t2 * tau_2, axis=1), 2))

                if grad.size > 0:
                    ip1 = np.sum(t1 * tau_1, axis=1)
                    ip2 = np.sum(t2 * tau_2, axis=1)

                    drX1 = sx_grad(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],0], q[self.cell_pairs[:,0],1], q[self.cell_pairs[:,1],0], q[self.cell_pairs[:,1],1], r1[:,0], r1[:,1])
                    drX2 = sx_grad(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],0], q[self.cell_pairs[:,0],1], q[self.cell_pairs[:,1],0], q[self.cell_pairs[:,1],1], r2[:,0], r2[:,1])
                    drY1 = sy_grad(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],0], q[self.cell_pairs[:,0],1], q[self.cell_pairs[:,1],0], q[self.cell_pairs[:,1],1], r1[:,0], r1[:,1])
                    drY2 = sy_grad(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],0], q[self.cell_pairs[:,0],1], q[self.cell_pairs[:,1],0], q[self.cell_pairs[:,1],1], r2[:,0], r2[:,1])

                    dE = np.multiply(ip1 * tau_1[:,0], drX1.T).T + np.multiply(ip1 * tau_1[:,1], drY1.T).T + \
                         np.multiply(ip2 * tau_2[:,0], drX2.T).T + np.multiply(ip2 * tau_2[:,1], drY2.T).T
                    dE = dE / self.dC.shape[0]

                    rows = np.concatenate([self.cell_pairs[:,0],self.cell_pairs[:,0]+self.dC.shape[1],self.cell_pairs[:,0]+2*self.dC.shape[1],
                                           self.cell_pairs[:,1],self.cell_pairs[:,1]+self.dC.shape[1],self.cell_pairs[:,1]+2*self.dC.shape[1]])

                    dE = np.bincount(rows, weights=np.ravel(dE,order='F'))
                    grad[:] = dE

                if self.verbose:
                    print(E)
                return E

            # Define nonlinear constraint for initial optimisation
            def nonlinear_con(x, grad=np.array([])):
                x = x.reshape(x0.shape, order='F')
                q = x[:,0:2]
                p = x[:,2]

                b = np.matmul(self.dC, np.multiply(q.T,p).T)
                delta_p = np.matmul(self.dC, p)

                l1 = np.linalg.norm(b - (np.multiply(r1.T,delta_p).T), axis=1)
                l2 = np.linalg.norm(b - (np.multiply(r2.T,delta_p).T), axis=1)

                E = 0.5*(np.mean(l1) + np.mean(l2)) - scale

                if grad.size > 0:
                    t1 = np.divide((b - (np.multiply(r1.T,delta_p).T)).T,np.linalg.norm(b - (np.multiply(r1.T,delta_p).T), axis=1)).T
                    t2 = np.divide((b - (np.multiply(r2.T,delta_p).T)).T,np.linalg.norm(b - (np.multiply(r2.T,delta_p).T), axis=1)).T

                    drX1 = rx_grad(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],0], q[self.cell_pairs[:,1],0], r1[:,0])
                    drX2 = rx_grad(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],0], q[self.cell_pairs[:,1],0], r2[:,0])
                    drY1 = ry_grad(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],1], q[self.cell_pairs[:,1],1], r1[:,1])
                    drY2 = ry_grad(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],1], q[self.cell_pairs[:,1],1], r2[:,1])


                    dE = 0.5 * (np.multiply(t1[:,0],drX1.T).T + np.multiply(t1[:,1],drY1.T).T) + \
                         0.5 * (np.multiply(t2[:,0],drX2.T).T + np.multiply(t2[:,1],drY2.T).T)

                    rows = np.concatenate([self.cell_pairs[:,0],self.cell_pairs[:,0]+self.dC.shape[1],self.cell_pairs[:,0]+2*self.dC.shape[1],
                                           self.cell_pairs[:,1],self.cell_pairs[:,1]+self.dC.shape[1],self.cell_pairs[:,1]+2*self.dC.shape[1]])

                    grad[:] = np.bincount(rows, weights=np.ravel(dE/self.dC.shape[0],order='F'))
                return E

            # Define linear constraint for initial optimisation
            def linear_con(x, grad=np.array([])):
                Aeq = np.concatenate((np.zeros(x0.shape[0]*2), np.ones(x0.shape[0])))
                E = np.dot(Aeq, x) - (np.mean(p0)*p0.shape[0])

                if grad.size > 0:
                    grad[:] = np.concatenate((np.zeros(2*x0.shape[0]), np.ones(x0.shape[0])))
                return float(E)

            # Configure optimiser
            local_opt = nlopt.opt(nlopt.LD_LBFGS, x0.size)
            init_opt = nlopt.opt(nlopt.AUGLAG, x0.size)
            init_opt.set_local_optimizer(local_opt)
            init_opt.set_min_objective(energy)
            lb = np.concatenate((-np.inf*np.ones(x0.shape[0]), -np.inf*np.ones(x0.shape[0]), 0.001*np.ones(x0.shape[0])))
            ub = np.concatenate((np.inf*np.ones(x0.shape[0]), np.inf*np.ones(x0.shape[0]), 2000*np.ones(x0.shape[0])))
            init_opt.set_lower_bounds(lb)
            init_opt.set_upper_bounds(ub)
            init_opt.add_inequality_constraint(nonlinear_con, 1e-6)
            init_opt.add_equality_constraint(linear_con, 1e-6)
            init_opt.set_maxeval(2000)

            # Optimisation
            # Nlopt can't handle initial values outside bounds so clip values before optimisation
            init_opt.optimize(np.clip(x0.ravel(order='F'), lb, ub))

            # For larger systems, the nlopt optimiser will not converge to the desired tolerance and does not return the results obtained at the final step
            # To get around this, retrieve results from log file
            x = np.genfromtxt('.init_opt.csv', delimiter=',')
            x = x.reshape(x0.shape, order='F')

            q = x[:,0:2]
            p = x[:,2]

            self.generate_circular_arcs()

            # Once q, p are optimised, perform initial optimisation for theta
            theta0 = self.estimate_theta(x)

            # Define energy function for theta optimisation
            def theta_energy(theta, grad=np.array([])):
                np.savetxt('./.theta_opt.csv', theta, delimiter=',',fmt='%f')

                dP = p[self.cell_pairs[:,0]] - p[self.cell_pairs[:,1]]
                dT = theta[self.cell_pairs[:,0]] - theta[self.cell_pairs[:,1]]
                dQ = q[self.cell_pairs[:,0],:] - q[self.cell_pairs[:,1],:]
                QL = np.sum(np.power(dQ, 2), axis=1)

                rho = np.divide(np.matmul(self.dC,np.multiply(p, q.T).T).T, dP).T
                r_sq = np.divide(((p[self.cell_pairs[:,0]] * p[self.cell_pairs[:,1]] * QL) - (dP * dT)),np.power(dP, 2))
                ind_z = r_sq<0
                r_sq[r_sq<0] = 0

                r = np.sqrt(r_sq)

                delta_x = np.subtract(rho[:,0],self.edgearc_x.T).T
                delta_y = np.subtract(rho[:,1],self.edgearc_y.T).T

                dMag = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))

                E = 0.5 * np.mean(np.sum(np.power(np.subtract(dMag.T, r).T, 2), axis=1))

                if grad.size > 0:
                    d = np.subtract(dMag.T, r).T

                    avg_d = np.sum(d, axis=1)
                    dR = radius_grad_theta(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],0], q[self.cell_pairs[:,0],1], q[self.cell_pairs[:,1],0], q[self.cell_pairs[:,1],1], theta[self.cell_pairs[:,0]], theta[self.cell_pairs[:,1]])
                    dR[ind_z,:] = 0

                    dE = np.divide(-np.multiply(avg_d, dR.T).T,self.dC.shape[0])
                    rows = np.concatenate([self.cell_pairs[:,0], self.cell_pairs[:,1]])

                    grad[:] = np.bincount(rows, weights=np.ravel(dE,order='F'))
                if self.verbose:
                    print(E)
                return float(E)

            def theta_eqlincon(theta, grad=np.array([])):
                Aeq = np.ones((1, len(self.involved_cells)))
                E = np.dot(Aeq, theta)

                if grad.size > 0:
                    grad[:] = np.ones(theta0.shape[0])
                return float(E)

            # Define nonlinear constraint for theta optimisation
            def theta_neqlincon(result, theta, grad=np.array([])):
                dP = p[self.cell_pairs[:,0]] - p[self.cell_pairs[:,1]]
                dQ = q[self.cell_pairs[:,0],:] - q[self.cell_pairs[:,1],:]
                QL = np.sum(np.power(dQ, 2), axis=1)

                A = np.multiply(dP, self.dC.T).T
                b = p[self.cell_pairs[:,0]] * p[self.cell_pairs[:,1]] * QL

                E = np.dot(A, theta) - b
                result[:] = E
                if grad.size > 0:
                    grad[:] = np.multiply(self.dC.T, dP).T
                return

            if (theta_energy(np.zeros_like(theta0)) < theta_energy(theta0)):
                theta0 = np.zeros_like(theta0)

            # Configure optimiser
            theta_local_opt = nlopt.opt(nlopt.LD_LBFGS, theta0.size)
            theta_opt = nlopt.opt(nlopt.AUGLAG, theta0.size)
            theta_opt.set_local_optimizer(theta_local_opt)
            theta_opt.set_ftol_abs(1e-5)
            theta_opt.set_min_objective(theta_energy)
            theta_opt.add_inequality_mconstraint(theta_neqlincon, 1e-5*np.ones(self.dC.shape[0]))
            # Having trouble with NLopt generic failures so disable equality constraint for now
            # This shouldn't matter since it's only setting the scale which we change during tiling anyway
    #        theta_opt.add_equality_constraint(theta_eqlincon, 1e-5)
            theta_opt.set_maxeval(2000)

            # Optimise
            theta_opt.optimize(theta0)

            theta = np.genfromtxt('.theta_opt.csv', delimiter=',')
            theta = np.array(theta)

        elif self.optimiser == 'matlab':
            # Equivalent optimisation steps for matlab optimiser instead
            import matlab
            p_mat = matlab.double(p0.tolist())
            q_mat = matlab.double((q0).tolist())
            d0_mat = matlab.double(self.dC.tolist())
            bCells_mat = matlab.double((self.cell_pairs+1).tolist())
            r1_mat = matlab.double((r1).tolist())
            r2_mat = matlab.double((r2).tolist())
            t1_mat = matlab.double(tau_1.tolist())
            t2_mat = matlab.double(tau_2.tolist())

            x = self.eng.initial_optimization(p_mat, q_mat, d0_mat, bCells_mat, r1_mat, r2_mat, t1_mat, t2_mat)
            x = np.array(x)

            q = x[:,0:2]
            p = x[:,2]

            self.generate_circular_arcs()
            theta0 = self.estimate_theta(x)

            p_mat = matlab.double(p.tolist())
            q_mat = matlab.double((q).tolist())
            theta0_mat = matlab.double(theta0.tolist())
            rBX_mat = matlab.double((self.edgearc_x).tolist())
            rBY_mat = matlab.double((self.edgearc_y).tolist())

            theta = self.eng.theta_optimization(theta0_mat, q_mat, p_mat, d0_mat, bCells_mat, rBX_mat, rBY_mat)
            theta = np.array(theta)
        return q, p, theta


    def fit(self):

        """

        Perform the minimization of equation 5 with respect to the variables (q, z, p)

        """
        self.prepare_data()

        if self.verbose:
            print("Initial minimization")
        # Perform initial minimization for p, q and theta
        q0, p0, theta0 = self.initial_minimization()
        X0 = np.vstack([q0.T, theta0.squeeze(), p0.squeeze()]).T

        q0 = X0[:,0:2]
        theta0 = X0[:,2]
        p0 = X0[:,3]

        if self.verbose:
            print("Main minimization")

        if self.optimiser == 'nlopt':
            import nlopt
            def objective(X, grad=np.array([])):
                np.savetxt('./.main_opt.csv', X, delimiter=',',fmt='%f')
                X = X.reshape(X0.shape, order='F')

                q = X[:,0:2]
                p = X[:,3]
                theta = X[:,2]

                dP = p[self.cell_pairs[:,0]] - p[self.cell_pairs[:,1]]
                dT = theta[self.cell_pairs[:,0]] - theta[self.cell_pairs[:,1]]
                dQ = q[self.cell_pairs[:,0],:] - q[self.cell_pairs[:,1],:]
                QL = np.sum(np.power(dQ, 2), axis=1)

                rho = np.divide(np.matmul(self.dC,np.multiply(p, q.T).T).T, dP).T
                r_sq = np.divide(((p[self.cell_pairs[:,0]] * p[self.cell_pairs[:,1]] * QL) - (dP * dT)),np.power(dP, 2))
                ind_z = r_sq<=0
                r_sq[ind_z] = 0

                r = np.sqrt(r_sq)

                delta_x = np.subtract(rho[:,0],self.edgearc_x.T).T
                delta_y = np.subtract(rho[:,1],self.edgearc_y.T).T

                dMag = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))

                E = 0.5 * np.mean(np.sum(np.power(np.subtract(dMag.T, r).T, 2), axis=1))

                if grad.size>0:

                    d = np.subtract(dMag.T, r).T

                    dRhoX = rho_x_grad(p[self.cell_pairs[:,0]],p[self.cell_pairs[:,1]],q[self.cell_pairs[:,0],0],q[self.cell_pairs[:,1],0])
                    dRhoY = rho_y_grad(p[self.cell_pairs[:,0]],p[self.cell_pairs[:,1]],q[self.cell_pairs[:,0],1],q[self.cell_pairs[:,1],1])
                    dR = radius_grad(p[self.cell_pairs[:,0]],p[self.cell_pairs[:,1]],q[self.cell_pairs[:,0],0],q[self.cell_pairs[:,0],1],q[self.cell_pairs[:,1],0],q[self.cell_pairs[:,1],1],theta[self.cell_pairs[:,0]],theta[self.cell_pairs[:,1]])

                    dNormX = np.sum(np.multiply(delta_x,np.divide(d, dMag)),axis=1)
                    dNormY = np.sum(np.multiply(delta_y,np.divide(d, dMag)),axis=1)

                    avg_d = np.sum(d, axis=1)
                    dR[ind_z] = 0

                    dE = np.divide(np.multiply(dNormX, dRhoX.T).T+np.multiply(dNormY, dRhoY.T).T-np.multiply(avg_d, dR.T).T,self.dC.shape[0])
                    rows = np.concatenate([self.cell_pairs[:,0],self.cell_pairs[:,0]+self.dC.shape[1],self.cell_pairs[:,0]+2*self.dC.shape[1],self.cell_pairs[:,0]+3*self.dC.shape[1],
                                           self.cell_pairs[:,1],self.cell_pairs[:,1]+self.dC.shape[1],self.cell_pairs[:,1]+2*self.dC.shape[1],self.cell_pairs[:,1]+3*self.dC.shape[1]])
                    dE = np.bincount(rows, weights=np.ravel(dE,order='F'))
                    grad[:] = dE.ravel()

                if self.verbose:
                    print(E)
                return E

            def nonlinear_con(result, X, grad=np.array([])):
                X = X.reshape(X0.shape, order='F')

                q = X[:,0:2]
                p = X[:,3]
                theta = X[:,2]

                result[:] = (np.matmul(self.dC,p) * np.matmul(self.dC, theta)) - (p[self.cell_pairs[:,0]] * p[self.cell_pairs[:,1]] * np.sum(np.power(np.matmul(self.dC, q), 2), axis=1))

                if grad.size>0:
                    X = X.reshape(X0.shape, order='F')

                    q = X[:,0:2]
                    p = X[:,3]
                    theta = X[:,2]

                    # Calculate jacobian of nonlinear constraints
                    dP = np.matmul(self.dC,p)
                    dT = np.matmul(self.dC, theta)
                    dQ = np.matmul(self.dC, q)
                    QL = np.sum(np.power(dQ, 2), axis=1)

                    gX = np.multiply(self.dC.T, -2*p[self.cell_pairs[:,0]]*p[self.cell_pairs[:,1]]*dQ[:,0]).T
                    gY = np.multiply(self.dC.T, -2*p[self.cell_pairs[:,0]]*p[self.cell_pairs[:,1]]*dQ[:,1]).T
                    gTh = np.multiply(self.dC.T, dP).T
                    gP = np.multiply(self.dC.T, dT).T - np.divide(np.multiply(QL*p[self.cell_pairs[:,0]]*p[self.cell_pairs[:,1]],np.abs(self.dC).T).T,p)
                    grad[:] = np.hstack([gX,gY,gTh,gP])
                return

            def linear_con(result, X, grad=np.array([])):
                Aeq = np.vstack((np.concatenate((np.zeros(3*theta0.shape[0]), np.ones(theta0.shape[0])/theta0.shape[0])),
                                 np.concatenate((np.zeros(2*theta0.shape[0]), np.ones(theta0.shape[0])/theta0.shape[0], np.zeros(theta0.shape[0])))))
                E = np.dot(Aeq, X) - np.array([np.mean(X0[:,3]), np.mean(X0[:,2])])
                if grad.size > 0:
                    grad[:] = np.vstack((np.concatenate((np.zeros(3*self.dC.shape[1]), np.ones(self.dC.shape[1])/self.dC.shape[1])),
                                         np.concatenate((np.zeros(2*self.dC.shape[1]), np.ones(self.dC.shape[1])/self.dC.shape[1], np.zeros(self.dC.shape[1])))))
                result[:] = E
                return

            local_opt = nlopt.opt(nlopt.LD_LBFGS, X0.size)

            main_opt = nlopt.opt(nlopt.AUGLAG, X0.size)
            main_opt.set_local_optimizer(local_opt)
            main_opt.set_min_objective(objective)
            lb = np.concatenate((-np.inf*np.ones(3*theta0.shape[0]), 0.001*np.ones(theta0.shape[0])))
            ub = np.concatenate((np.inf*np.ones(3*theta0.shape[0]), 1000*np.ones(theta0.shape[0])))
            main_opt.set_lower_bounds(lb)
            main_opt.set_upper_bounds(ub)
            main_opt.add_inequality_mconstraint(nonlinear_con, 1e-6*np.ones(self.dC.shape[0]))
            main_opt.add_equality_mconstraint(linear_con, 1e-6*np.ones(2))
            main_opt.set_maxeval(2000)

            main_opt.optimize(np.clip(X0.ravel(order='F'),lb,ub))

            X = np.genfromtxt('.main_opt.csv', delimiter=',')
            X = X.reshape(X0.shape, order='F')
        elif self.optimiser == 'matlab':
            import matlab

            X0_mat = matlab.double(X0.tolist())
            bCells_mat = matlab.double((self.cell_pairs+1).tolist())
            d0_mat = matlab.double(self.dC.tolist())
            rBX_mat = matlab.double((self.edgearc_x).tolist())
            rBY_mat = matlab.double((self.edgearc_y).tolist())

            res = self.eng.main_minimization(bCells_mat, d0_mat, rBX_mat, rBY_mat, X0_mat, float(self.avg_edge_length))

            X = np.array(res)
            X = X.reshape(X0.shape, order='F')

            self.eng.quit()

        q = X[:,0:2]
        p = X[:,3]
        theta = X[:,2]

        # compute the tensions
        T = self.get_tensions(q, p, theta)
        self.upload_mechanics(p, T, q, theta)
        return


    def get_tensions(self, q, p, theta):

        """

        applies the Young-Laplace law to obtain the tensions at every edge

        """
        T = np.matmul(self.dC, q)
        T = np.sum(np.power(T, 2), axis=1)
        T = T * np.abs(np.array([p[alpha] * p[beta] for (alpha,beta) in self.cell_pairs]))
        T = T - np.multiply(np.matmul(self.dC, p), np.matmul(self.dC, theta))
        T = np.sqrt(T)
        return T

    def upload_mechanics(self, p, T, q, theta):
        """

        Match tensions and pressures to edges and cells

        """

        self.cells.pressure[np.sort(self.involved_cells)] = p[np.argsort(self.involved_cells)]
        self.cells.qx[np.sort(self.involved_cells)] = q[np.argsort(self.involved_cells),0]
        self.cells.qy[np.sort(self.involved_cells)] = q[np.argsort(self.involved_cells),1]
        self.cells.theta[np.sort(self.involved_cells)] = theta[np.argsort(self.involved_cells),]

        for i in range(len(self.edges)):
            edge_cells = self.edges.at[i, 'cells']

            if all(np.isin(edge_cells, self.involved_cells)) and np.isin(i, self.involved_edges):
                cell_ind1 = np.where(self.involved_cells == edge_cells[0])[0]
                cell_ind2 = np.where(self.involved_cells == edge_cells[1])[0]

                edge_ind = np.where(np.squeeze((self.dC[:,cell_ind1] != 0) & (self.dC[:,cell_ind2] != 0), axis=1))[0]
                self.edges.at[i, 'tension'] = T[edge_ind]
        return

    def return_tensions(self):
        return self.edges.tension.values

    def return_pressures(self):
        return self.cells.pressure.values[1:]

    def compute_stresstensor(self):
        """

        Compute stress tensor for each cell from tensions and pressures.
        This uses a small-angle approximation assuming edge curvatures are small.

        """

        p = np.array([self.cells.at[cell, 'pressure'] for cell in self.involved_cells])
        T = np.zeros(self.cell_pairs.shape[0])
        edge_verts = np.array(self.edges.verts.to_list())

        i1 = -1*np.ones_like(T)
        for e in range(len(T)):
            verts = self.involved_vertices[self.dV[e,:] != 0]

            if len(verts) == 2:
                ind = np.where(np.all(verts == np.sort(edge_verts, axis=1), axis=1))[0]
            else:
                ind = np.array([])

            if ind.size>0 and self.edges.at[int(ind), 'tension'].size>0:
                T[e] = self.edges.at[int(ind), 'tension']
                i1[e] = int(ind)
            else:
                T[e] = 1

        rv = np.array([self.vertices.at[vertex, 'coords'] for vertex in self.involved_vertices])


        rb = np.matmul(self.dV, rv)
        D = np.sqrt(np.sum(np.power(rb, 2), 1))
        D[D==0] = 1
        rb = np.divide(rb.T, D).T
        Rot = np.array([[0,-1],[1,0]])
        nb = np.matmul(rb, Rot.T)
        dP = np.matmul(self.dC, p)

        sigmaB = np.zeros([rb.shape[0], 3])
        sigmaB[:,0] = rb[:,0] * T * rb[:,0]
        sigmaB[:,1] = rb[:,0] * T * rb[:,1]
        sigmaB[:,2] = rb[:,1] * T * rb[:,1]

        sigmaP = np.zeros([rb.shape[0], 3])
        sigmaP[:,0] = nb[:,0] * dP * D * nb[:,0]
        sigmaP[:,1] = nb[:,0] * dP * D * nb[:,1]
        sigmaP[:,2] = nb[:,1] * dP * D * nb[:,1]

        sigma = np.matmul(np.abs(self.dC.T), sigmaB) + 0.5 * np.matmul(self.dC.T, sigmaP)

        A = np.array(self.cells.area.to_list())[self.involved_cells]
        sigma = np.divide(sigma.T, A)
        for c in range(len(self.involved_cells)):
            self.cells.at[self.involved_cells[c], 'stress'] = sigma[:,c]

        return


    def plot(self, options='', mask=np.array([]), line_thickness=5, size=10, file=None):
        """

        Plots results of stress inference.

        :param mask: (numpy array) image on which to overlay plotted objects. If plotting pressure, must be labelled, segmented image.
        :param options: (list) list of options for plotting. Available options are: 'stress', 'pressure', 'tension', 'CAP'.
        :param line_thickness: (int) thickness of lines used for tension and CAP plotting. Default: 5.
        :param size: (int) text size for legends. Default: 10.
        :param file: (str) filename to save plot to. If none provided, outputs plot to console.
        """
        options = [option.lower() for option in options]

        # Pressure first as requires remapping cell area colours
        if mask.size > 0:
            if np.isin('pressure', options):
                img = np.zeros_like(mask).astype(float)
                colourmap = cm.get_cmap('plasma')
                centroids = np.array(self.cells.loc[self.involved_cells, 'centroids'].tolist())
                pressures = self.cells.pressure.to_numpy()[self.involved_cells]
                props = measure.regionprops(mask)
                img_centroids = np.array([np.flip(regionprops.centroid) for regionprops in measure.regionprops(mask)])

                maxP = np.percentile(pressures, 90)
                indices = np.argmin(cdist(centroids, img_centroids), axis=1)
                involvedcells_labels = np.zeros(len(self.involved_cells))
                for i in range(len(self.involved_cells)):
                    img_index = indices[i]
                    img_label = props[img_index].label
                    involvedcells_labels[i] = img_label
                    p_norm = np.divide(pressures[i], maxP)
                    if p_norm > 1:
                        p_norm = 1
                    img[mask==img_label] = p_norm
                img = colourmap(img)
                img[~np.isin(mask, involvedcells_labels),:] = (1,1,1,1)
            else:
                colourmap = cm.get_cmap('Set3')
                colours = np.array([colourmap(np.mod(i,12)) for i in range(len(np.unique(mask)))])
                img = color.label2rgb(mask, mask, colors=colours, alpha=0.7, bg_label=0, bg_color=(0, 0, 0))
        else:
            if np.isin('pressure', options):
                return("Error: 'pressure' plotting specified without providing labelled, segmented image.")
            mask = np.ones((self.height, self.width))
            # Need to map zeros to black otherwise background will be purple
            img = color.label2rgb(mask, mask, colors=[(1,1,1)], alpha=1)

        # Set up figure
        fig, ax = plt.subplots(1,1,figsize=np.divide(mask.shape,72), dpi=72)
        ax.imshow(img)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        divider = make_axes_locatable(ax)

        # Now we add colourbar for pressure if it was specified
        if np.isin('pressure', options):
            cax1 = divider.append_axes("right", size="5%", pad=0.5)
            p_cb = plt.colorbar(mappable=cm.ScalarMappable(norm=colors.Normalize(np.min(pressures), maxP), cmap=colourmap), cax=cax1)
            p_cb.set_label('Pressure (a.u.)', size=size)
            p_cb.ax.tick_params(labelsize=size)

        if all(np.isin(options, ['stress', 'pressure', 'tension', 'cap'])):
            for option in options:
                if option == 'stress':
                    stress = np.array([self.cells.at[cell, 'area'] * np.array([[self.cells.at[cell, 'stress'][0], self.cells.at[cell, 'stress'][1]],[self.cells.at[cell, 'stress'][1], self.cells.at[cell, 'stress'][2]]]) for cell in range(len(self.cells))])
                    eigvals, eigvects = np.linalg.eig(stress)
                    scalefct = np.sqrt(np.median(np.multiply(eigvals[:,0], eigvals[:,1])))

                    for i in range(len(self.involved_cells)):
                        cell = self.involved_cells[i]
                        if np.max(stress[i] > 0):
                            centroid = self.cells.at[cell, 'centroids']
                            eigval = eigvals[i,:]
                            eigvect = eigvects[i,:,:]
                            idx = np.flip(np.argsort(eigval))
                            eigval = eigval[idx]
                            eigvect = eigvect[:,idx]

                            # scale eigenvalues
                            eigval = np.divide(eigval, scalefct)
                            eigval[eigval>3] = 3
                            eigval[eigval<0] = 0
                            eigval = eigval * 0.4 * np.mean(np.sqrt(np.divide(np.array(self.cells.area.to_list())[1:], np.pi)))

                            # calculate angle of rotation
                            theta = np.arctan2(eigvect[1,0], eigvect[0,0])
                            if theta < 0:
                                theta = theta + 2*np.pi
                            theta = np.degrees(theta)
                            stress_ellipse = patches.Ellipse(centroid, eigval[0], eigval[1], theta, fill=False, color='red', lw=3)
                            ax.add_patch(stress_ellipse)
                elif option == 'tension':
                    maxT = np.percentile(self.return_tensions(), 95)
                    minT = np.percentile(self.return_tensions(), 1)
                    colourmap = cm.get_cmap('hot')

                    for e in range(len(self.edges)):
                        if self.edges.at[e, 'tension'] > 0:
                            radius = self.edges.at[e, 'radius']
                            tension_norm = np.divide(self.edges.at[e, 'tension'], maxT-minT)
                            if tension_norm > 1:
                                tension_norm = 1
                            colour = colourmap(tension_norm)

                            if np.isinf(radius):

                                v = np.array([self.vertices.at[self.edges.at[e, 'verts'][0], 'coords'], self.vertices.at[self.edges.at[e, 'verts'][1], 'coords']])
                                points = np.array([np.linspace(v[0,0], v[1,0], len(self.edges.at[e, 'pixels'])),
                                                   np.linspace(v[0,1], v[1,1], len(self.edges.at[e, 'pixels']))]).T
                                ax.plot(points[:,0], points[:,1], lw=line_thickness, color=colour)
                            else:
                                v = np.array([self.vertices.at[self.edges.at[e, 'verts'][0], 'coords'], self.vertices.at[self.edges.at[e, 'verts'][1], 'coords']])
                                rho = self.edges.at[e, 'rho']

                                theta = np.arctan2(v[:,1]-rho[1], v[:,0]-rho[0])
                                theta[theta<0] = theta[theta<0] + 2*np.pi
                                theta = np.sort(theta)

                                if theta[1] - theta[0] > np.pi:
                                    theta[1] = theta[1] - 2*np.pi

                                theta_range = np.linspace(theta[0], theta[1], len(self.edges.at[e, 'pixels']))
                                points = np.array([rho[0] + radius*np.cos(theta_range),
                                                   rho[1] + radius*np.sin(theta_range)]).T
                                ax.plot(points[:,0], points[:,1], lw=line_thickness, color=colour)
                    cax2 = divider.append_axes("right", size="5%", pad=0.5)
                    t_cb = plt.colorbar(mappable=cm.ScalarMappable(norm=colors.Normalize(minT, maxT), cmap=colourmap), cax=cax2)
                    t_cb.set_label('Tension (a.u.)', size=size)
                    t_cb.ax.tick_params(labelsize=size)
                elif option == 'cap':
                    for e in range(len(self.edges)):
                        radius = self.edges.at[e, 'radius']

                        if np.isinf(radius):

                            v = np.array([self.vertices.at[self.edges.at[e, 'verts'][0], 'coords'], self.vertices.at[self.edges.at[e, 'verts'][1], 'coords']])
                            points = np.array([np.linspace(v[0,0], v[1,0], len(self.edges.at[e, 'pixels'])),
                                               np.linspace(v[0,1], v[1,1], len(self.edges.at[e, 'pixels']))]).T
                            ax.plot(points[:,0], points[:,1], lw=line_thickness, color='b')
                        else:
                            v = np.array([self.vertices.at[self.edges.at[e, 'verts'][0], 'coords'], self.vertices.at[self.edges.at[e, 'verts'][1], 'coords']])
                            rho = self.edges.at[e, 'rho']

                            theta = np.arctan2(v[:,1]-rho[1], v[:,0]-rho[0])
                            theta[theta<0] = theta[theta<0] + 2*np.pi
                            theta = np.sort(theta)

                            if theta[1] - theta[0] > np.pi:
                                theta[1] = theta[1] - 2*np.pi

                            theta_range = np.linspace(theta[0], theta[1], len(self.edges.at[e, 'pixels']))
                            points = np.array([rho[0] + radius*np.cos(theta_range),
                                               rho[1] + radius*np.sin(theta_range)]).T
                            ax.plot(points[:,0], points[:,1], lw=line_thickness, color='b')
        if file is not None:
            plt.savefig(filename)
        else:
            plt.show()
        return

    def smooth_stress(self, smoothsize=10):
        """

        Compute smoothed stress tensor and stores it in the stress_smooth field
        This reduces variability in stress between individual cells and makes it easier to spot tissue-wide trends

        :param smoothsize: (int) Degree of smoothing; controls width of gaussian kernel over cell neighbours
        """
        self.cells['stress_smooth'] = [np.array([0,0,0]) for _ in range(self.cells.shape[0])]

        rc = np.zeros((len(self.involved_cells), 2))
        stress = np.zeros((len(self.involved_cells), 3))
        for i in range(len(self.involved_cells)):
            cell = self.involved_cells[i]
            rc[i,:] = self.cells.at[cell, 'centroids']
            stress[i,:] = self.cells.at[cell, 'stress']

        # scaling factor based on mean cell area
        A = np.array(self.cells.area.to_list()[1:])
        r0 = 0.5 * np.mean(np.sqrt(np.divide(A, np.pi)))

        dist = cdist(rc, rc)
        smk = np.exp(-np.power(dist, 2)/(2*np.power(smoothsize*r0,2)))
        smk = np.divide(smk, np.sum(smk, axis=1))

        stress[:,0] = smk @ stress[:,0]
        stress[:,1] = smk @ stress[:,1]
        stress[:,2] = smk @ stress[:,2]

        for i in range(len(self.involved_cells)):
            cell = self.involved_cells[i]
            self.cells.at[cell, 'stress_smooth'] = stress[i,:]
        return

    def output_results(self, metrics=['centroids','pressure','stress','inertia','perimeter','feret_d','moments_hu','bbox', 'area'], neighbours=False):
        """
        Outputs force inference/morphometric quantities and cell adjacency matrix as Pandas Dataframes.

        :param metrics: (list) list of metrics to output. Default: all metrics.
        :param neighbours: (bool) whether to output adjacency matrix. Nonzero values in matrix represent junction tension. Default: False.
        """

        # Assign actual id's to each cell instead of just using labels
        cell_ids = {label:id for label, id in zip(self.involved_cells, ['cell_' + str(number+1) for number in range(len(self.involved_cells))])}

        if metrics is not None:
            out_df = self.cells.loc[self.involved_cells, metrics]
            if 'centroids' in out_df.columns:
                out_df['centroid_x'] = np.array(out_df['centroids'].tolist())[:,0]
                out_df['centroid_y'] = np.array(out_df['centroids'].tolist())[:,1]
                out_df.drop('centroids', axis=1, inplace=True)
            if 'stress' in out_df.columns:
                # We want each value to make sense on its own
                # Therefore, compute eigenvalues, orientation and anisotropy of stress tensor
                stress = np.array([np.array([[out_df.at[cell, 'stress'][0], out_df.at[cell, 'stress'][1]],[out_df.at[cell, 'stress'][1], out_df.at[cell, 'stress'][2]]]) for cell in out_df.index])
                eigvals, eigvects = np.linalg.eig(stress)
                eigvals = np.abs(eigvals)
                idx = np.flip(np.argsort(eigvals, axis=1), axis=1)
                eigvals = eigvals[np.arange(eigvals.shape[0])[:,None], idx]
                eigvects = eigvects[np.arange(eigvects.shape[0])[:,None], idx]
                out_df['stresstensor_eigval1'] = eigvals[:,0]
                out_df['stresstensor_eigval2'] = eigvals[:,1]
                # define orientation as absolute angle between largest eigenvector and x-axis
                out_df['stresstensor_orientation'] = np.arctan2(np.abs(eigvects[:,0,:])[:,1], np.abs(eigvects[:,0,:])[:,0])
                # define anisotropy using definition from Hartkamp et al., J. Chem. Phys. 2012
                out_df['stresstensor_anisotropy'] = np.divide(eigvals[:,0] - eigvals[:,1], (eigvals[:,0] + eigvals[:,1]))
                out_df.drop('stress', axis=1, inplace=True)
            if 'inertia' in out_df.columns:
                # We want each value to make sense on its own
                # Therefore, compute eigenvalues, orientation and anisotropy of inertia tensor
                inertia = np.array([np.array([[out_df.at[cell, 'inertia'][0], out_df.at[cell, 'inertia'][1]],[out_df.at[cell, 'inertia'][1], out_df.at[cell, 'inertia'][2]]]) for cell in out_df.index])
                eigvals, eigvects = np.linalg.eig(inertia)
                eigvals = np.abs(eigvals)
                idx = np.flip(np.argsort(eigvals, axis=1), axis=1)
                eigvals = eigvals[np.arange(eigvals.shape[0])[:,None], idx]
                eigvects = eigvects[np.arange(eigvects.shape[0])[:,None], idx]
                out_df['inertiatensor_eigval1'] = eigvals[:,0]
                out_df['inertiatensor_eigval2'] = eigvals[:,1]
                # define orientation as absolute angle between largest eigenvector and x-axis
                out_df['inertiatensor_orientation'] = np.arctan2(np.abs(eigvects[:,0,:])[:,1], np.abs(eigvects[:,0,:])[:,0])
                out_df['inertiatensor_anisotropy'] = np.divide(eigvals[:,0] - eigvals[:,1], (eigvals[:,0] + eigvals[:,1]))
                out_df.drop('inertia', axis=1, inplace=True)
            if 'moments_hu' in out_df.columns:
                # Hu moments 1 and 3 (2 is omitted as it is equivalent to inertia tensor anisotropy)
                out_df['moments_hu_1'] = np.array(out_df['moments_hu'].tolist())[:,0]
                out_df['moments_hu_3'] = np.array(out_df['moments_hu'].tolist())[:,2]
                out_df.drop('moments_hu', axis=1, inplace=True)
            if 'bbox' in out_df.columns:
                out_df['bbox_x'] = np.array(out_df['bbox'].tolist())[:,0]
                out_df['bbox_y'] = np.array(out_df['bbox'].tolist())[:,1]
                out_df.drop('bbox', axis=1, inplace=True)
            out_df.rename(index=cell_ids, inplace=True)

        if neighbours:
            # Generate adjacency matrix.
            # Magnitude of non-zero values represents the tension (in arbitrary units) along the junction.
            adj_mat = pd.DataFrame(np.zeros([len(self.involved_cells), len(self.involved_cells)]), index=self.involved_cells, columns=self.involved_cells)

            for cell in self.involved_cells:
                adj_cells = self.cells.at[cell, 'ncells']
                for adj_cell in adj_cells[np.isin(adj_cells, self.involved_cells)]:
                    adj_mat.loc[cell, adj_cell] = self.edges[self.edges.cells.apply(tuple) == tuple(np.sort([cell, adj_cell]))]['tension'].values
            adj_mat.rename(index=cell_ids, columns=cell_ids, inplace=True)

        if neighbours and metrics is not None:
            return out_df, adj_mat
        elif metrics is None:
            return adj_mat
        elif not neighbours:
            return out_df
        else:
            return False

def run_VMSI(img, is_labelled=False, holes_mask=None, tile=False, cells_per_tile=150, verbose=False, overlap=0.3, optimiser='nlopt'):
    """
    Main function to run stress inference in a single step.

    :param verbose: (bool) whether to provide detailed output. Default:False.
    :param img: (numpy array) segmented image.
    :param is_labelled: (bool) whether all cells in image are labelled. Default: False.
    :param holes_mask: (numpy array) binary image containing interior holes in segmented image. Default: None.
    :param tile: (bool) whether to break image into tiles for faster inference. Not recommended for images with tissue-scale anisotropy. Default: False.
    :param overlap: (float) fraction of overlap between tiles. Default: 0.3.
    :param optimiser: (str) which optimiser to use. Currently available options are 'nlopt' (default) , 'matlab'.

    :return: VMSI object containing the inferred tensions, pressures and stress, as well as cell morphology metrics.
    """

    warnings.filterwarnings('ignore')

    # If cells are not labelled, label them
    if not is_labelled:
        img = measure.label(img)
    # Test whether there are enough cells to tile image
    if not len(np.unique(img))-1 > 2*cells_per_tile and tile:
        print('Not enough cells for tiling; proceeding with a single tile.')
        tile = False
    if tile:

        # Generating tiling
        tiles, holes_masks, offset, adj_tiles = create_image_tiles(img, holes_mask, cells_per_tile=cells_per_tile, overlap=overlap)
        models = []

        pairwise_tensions = []
        pairwise_pressures = []

        # Iterate through each tile and do stress inference
        for i in range(len(tiles)):
            tile = tiles[i]
            tile_holes_mask = holes_masks[i]
            # process segmented image for input into VMSI
            seg = Segmenter(masks=tile, labelled=is_labelled)
            VMSI_obj, labelled_mask = seg.process_segmented_image(holes_mask=tile_holes_mask)
            # create the model
            model = VMSI(vertices=VMSI_obj.V_df, cells=VMSI_obj.C_df, edges=VMSI_obj.E_df, height=tile.shape[0], width=tile.shape[1], verbose=verbose, optimiser=optimiser)
            # fit the model parameters
            model.fit()
            models.append(model)
        # For each pair of adjacent tiles, determine cell overlap and record overlapping tensions and pressures
        for i in range(len(adj_tiles)):
            pair = adj_tiles[i]
            model_1 = models[pair[0]]
            model_2 = models[pair[1]]

            # Match pressures
            model1_cells = np.array(model_1.cells.centroids.tolist())[1:] + offset[pair[0]]
            model2_cells = np.array(model_2.cells.centroids.tolist())[1:] + offset[pair[1]]

            cell_pwdist = cdist(model1_cells, model2_cells)
            # We expect matching cells to have assigned centroids that are <=2px apart in the two tiles
            # This should be the same regardless of image size
            matching_cells = np.where(cell_pwdist<=2)
            pressures = np.array([model_1.return_pressures()[matching_cells[0]], model_2.return_pressures()[matching_cells[1]]])
            pairwise_pressures.append(pressures[:,np.all(pressures>0, axis=0)])

            # Match edges
            model1_edge_r1 = np.array(model_1.vertices.coords[np.array(model_1.edges.verts.tolist())[:,0]].tolist()) + offset[pair[0]]
            model1_edge_r2 = np.array(model_1.vertices.coords[np.array(model_1.edges.verts.tolist())[:,1]].tolist()) + offset[pair[0]]
            model2_edge_r1 = np.array(model_2.vertices.coords[np.array(model_2.edges.verts.tolist())[:,0]].tolist()) + offset[pair[1]]
            model2_edge_r2 = np.array(model_2.vertices.coords[np.array(model_2.edges.verts.tolist())[:,1]].tolist()) + offset[pair[1]]

            r1_pwdist = cdist(model1_edge_r1, model2_edge_r1)
            r2_pwdist = cdist(model1_edge_r2, model2_edge_r2)

            matching_edges = np.where(np.logical_and(r1_pwdist <= 2, r2_pwdist <= 2))
            tensions = np.array([model_1.return_tensions()[matching_edges[0]], model_2.return_tensions()[matching_edges[1]]])
            pairwise_tensions.append(tensions[:,np.all(tensions>0, axis=0)])

        # Scale tensions and pressures globally
        def p_energy(x):
            E = np.sum([np.sum(np.power(pairwise_pressures[i][0,:] + x[adj_tiles[i][0]] - pairwise_pressures[i][1,:] - x[adj_tiles[i][1]], 2)) for i in range(len(adj_tiles))])
            return E
        constr = LinearConstraint(np.ones(len(tiles))/len(tiles), 0, 0)
        res = minimize(p_energy, np.zeros(len(tiles)), tol=1e-8, constraints=[constr])
        p_scale = res.x

        def t_energy(x):
            E = np.sum([np.sum(np.power(pairwise_tensions[i][0,:]*x[adj_tiles[i][0]] - pairwise_tensions[i][1,:]*x[adj_tiles[i][1]], 2)) for i in range(len(adj_tiles))])
            return E
        constr = LinearConstraint(np.ones(len(tiles))/len(tiles), 1, 1)
        res = minimize(t_energy, np.ones(len(tiles)), tol=1e-8, constraints=[constr])
        t_scale = res.x

        model = merge_models(models, p_scale, t_scale, offset, img, verbose=verbose, holes_mask=holes_mask)
    else:
        # process segmented image for input into VMSI
        seg = Segmenter(masks=img, labelled=is_labelled)
        VMSI_obj, labelled_mask = seg.process_segmented_image(holes_mask=holes_mask)
        # create the model
        model = VMSI(vertices=VMSI_obj.V_df, cells=VMSI_obj.C_df, edges=VMSI_obj.E_df, height=img.shape[0], width=img.shape[1], verbose=verbose, optimiser=optimiser)
        # fit the model parameters
        model.fit()
        # compute stress tensor
        model.compute_stresstensor()
    return model

def create_image_tiles(img, holes_mask, cells_per_tile=150, overlap=0.3):
    """

    Split image into overlapping tiles, each of which contain approximately the same number of cells.

    :param img: (numpy array) labelled, segmented image to be split into tiles.
    :param holes_mask: (numpy array) binary image containing interior holes in segmented image.
    :param cells_per_tile: (int) number of cells per tile. Default: 150.
    :param overlap: (float) fraction of overlap between each adjacent tiles. Default: 0.3.

    :return tiles: (list) list of tiles of original image, where each tile is a numpy array.
    :return holes_masks: (list) list of tiles of holes masks, where each tile is a numpy array.
    :return offset: (list) offset between each tile's top left corner relative to the untiled image.
    :return adj_tiles: (list) pairs of adjacent tiles with sufficient overlap to allow scaling after inference.
    """
    cell_properties = pd.DataFrame(measure.regionprops_table(img, properties=('label','bbox','centroid')))
    ncells = cell_properties.shape[0]

    # Get tile centres using K-means
    ntiles = int(2*np.round(ncells/(2*cells_per_tile)))
    cell_centroids = cell_properties[['centroid-0', 'centroid-1']].values
    kmeans = KMeans(n_clusters=ntiles, random_state=0)
    kmeans_res = kmeans.fit_predict(cell_centroids)

    # Compute bounding box for each tile
    tiles = []
    holes_masks = []
    offset = []
    tiles_bbox = np.zeros([ntiles, 4], dtype=int)
    for i in range(ntiles):
        tile_index = np.unique(kmeans_res)[i]
        cell_bbox = cell_properties[['bbox-0','bbox-1','bbox-2','bbox-3']].iloc[np.where(kmeans_res == tile_index)[0]].values
        tile_bbox = np.array([np.min(cell_bbox[:,0]), np.min(cell_bbox[:,1]), np.max(cell_bbox[:,2]), np.max(cell_bbox[:,3])])
        # Expand bounding box by overlap percentage (in each direction)
        tile_bbox += np.round((overlap/2)*np.array([-(tile_bbox[3]-tile_bbox[1]), -(tile_bbox[2]-tile_bbox[0]), (tile_bbox[3]-tile_bbox[1]), (tile_bbox[2]-tile_bbox[0])])).astype(int)
        tile_bbox = np.clip(tile_bbox, [0, 0, 0, 0], [img.shape[0], img.shape[1], img.shape[0], img.shape[1]])
        tiles_bbox[i,:] = tile_bbox

        tiles.append(img[tile_bbox[0]:tile_bbox[2], tile_bbox[1]:tile_bbox[3]])
        if holes_mask is not None:
            holes_masks.append(holes_mask[tile_bbox[0]:tile_bbox[2], tile_bbox[1]:tile_bbox[3]])
        else:
            holes_masks.append(None)
        offset.append(np.array([tile_bbox[1], tile_bbox[0]]))

    # Determine adjacent tiles using overlap; threshold by half overlap fraction multiplied by smallest tile area
    adj_tiles = []
    for tile_1 in range(ntiles):
        for tile_2 in range(tile_1+1, ntiles):
            bbox_1 = tiles_bbox[tile_1,:]
            bbox_2 = tiles_bbox[tile_2,:]
            if not (bbox_1[2]<bbox_2[0] or bbox_1[0]>bbox_2[2] or bbox_1[1]>bbox_2[3] or bbox_1[3]<bbox_2[1]):
                # If there is overlap, determine overlap area
                x_left = np.maximum(bbox_1[0], bbox_2[0])
                y_top = np.maximum(bbox_1[1], bbox_2[1])
                x_right = np.minimum(bbox_1[2], bbox_2[2])
                y_bottom = np.minimum(bbox_1[3], bbox_2[3])

                # The intersection of two axis-aligned bounding boxes is always an axis-aligned bounding box
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                # Only use overlap that contains actual cells, not holes
                intersection_area -= np.sum(holes_mask[x_left:x_right, y_top:y_bottom])
                area_1 = (bbox_1[2] - bbox_1[0]) * (bbox_1[3] - bbox_1[1])
                area_2 = (bbox_2[2] - bbox_2[0]) * (bbox_2[3] - bbox_2[1])

                if intersection_area > (overlap/3)*np.minimum(area_1, area_2):
                    adj_tiles.append([tile_1, tile_2])


    # Output tiling for debug
    fig, ax = plt.subplots(1,1,figsize=np.divide(img.shape,72), dpi=72)
    ax.imshow(img)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for tile_index in range(tiles_bbox.shape[0]):
        bbox = tiles_bbox[tile_index,:]
        plt.hlines(bbox[0], bbox[1], bbox[3], linewidth=10, colors='black')
        plt.hlines(bbox[2], bbox[1], bbox[3], linewidth=10, colors='black')
        plt.vlines(bbox[1], bbox[0], bbox[2], linewidth=10, colors='black')
        plt.vlines(bbox[3], bbox[0], bbox[2], linewidth=10, colors='black')
    plt.tight_layout()
    plt.show()
    return tiles, holes_masks, offset, adj_tiles


def merge_models(models, p_scale, t_scale, offset, img, verbose, holes_mask=None):
    """

    Merges VMSI objects created for each tile into a single object based on scaling factors calculated for tensions and pressures

    :param models: (list) list of VMSI objects for each tile
    :param p_scale: (numpy array) additive scaling factors for cell pressures for each tile.
    :param t_scale: (numpy array) multiplicative scaling factors for junction tensions for each tile.
    :param offset: (list) offset between each tile's top left corner relative to the untiled image.
    :param img: (numpy array) labelled image.
    :param verbose: (bool) output verbosity.
    :param holes_mask: (numpy array) binary image containing interior holes in segmented image.

    :return: single VMSI object containing inferred mechanics for the entire image.
    """
    # process segmented image for input into VMSI
    seg = Segmenter(masks=img, labelled=True)
    VMSI_obj, labelled_mask = seg.process_segmented_image(holes_mask=holes_mask)
    # create the model
    merged_model = VMSI(vertices=VMSI_obj.V_df, cells=VMSI_obj.C_df, edges=VMSI_obj.E_df, height=img.shape[0], width=img.shape[1], verbose=verbose)
    merged_model.prepare_data()

    # counters for number of inferred values for each edge and cell
    ncell = np.zeros(len(merged_model.cells))
    nedge = np.zeros(len(merged_model.edges))
    for i in range(len(models)):
        model = models[i]
        model.cells.pressure = model.cells.pressure.apply(lambda x: x + p_scale[i])
        model.edges.tension = model.edges.tension.apply(lambda x: x * t_scale[i])
        model.compute_stresstensor()

        # Match cells and edges for each model to the merged model
        def match_index(coords_1, coords_2):
            pdist = cdist(coords_1, coords_2)
            indices = np.argmin(pdist, axis=1)
            return indices

        cell_indices = match_index(np.array(model.cells.loc[model.involved_cells].centroids.tolist()) + offset[i], np.array(merged_model.cells.centroids.tolist()))
        merged_model.involved_cells = (np.unique(np.concatenate((merged_model.involved_cells, cell_indices)))) if merged_model.involved_cells is not None else (cell_indices)
        merged_model.cells.loc[cell_indices, ['pressure', 'stress']] = merged_model.cells.loc[cell_indices, ['pressure', 'stress']].values + model.cells.loc[model.involved_cells, ['pressure', 'stress']].values
        ncell[cell_indices] += 1

        for edge in model.involved_edges:
            index = int(np.where(np.all(np.sort(cell_indices[np.ravel(np.argwhere(np.isin(model.involved_cells, model.edges.at[edge, 'cells'])))]) == np.sort(np.array(merged_model.edges.cells.tolist())), axis=1))[0])
            merged_model.edges.at[index, 'tension'] = merged_model.edges.at[index, 'tension'] + model.edges.at[edge, 'tension']
            nedge[index] += 1

    # Take mean over edges and cells with multiple inferred values
    merged_model.cells.loc[ncell>0, 'pressure'] = np.divide(merged_model.cells.loc[ncell>0, 'pressure'].values,ncell[ncell>0])
    merged_model.cells.loc[ncell>0, 'stress'] = np.divide(merged_model.cells.loc[ncell>0, 'stress'].values,ncell[ncell>0])
    merged_model.edges.loc[nedge>0, 'tension'] = np.divide(merged_model.edges.loc[nedge>0, 'tension'].values,nedge[nedge>0])

    return merged_model