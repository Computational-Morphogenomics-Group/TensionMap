import cv2
import glob
import cyipopt
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from cellpose import models
from cellpose import utils
from cellpose import plot
from scipy.ndimage import generic_filter
import itertools
from scipy.optimize import minimize, leastsq
from scipy.sparse import coo_matrix
import pandas as pd
import matlab.engine
import matlab
from matplotlib import cm

class VMSI():

    def __init__(self, vertices, cells, edges, width=500, height=500):
        self.vertices = vertices
        self.cells = cells
        self.edges = edges
        self.width = width
        self.height = height

        # Mark fourfold vertices
        self.vertices['fourfold'] = [(np.shape(nverts)[0] > 3) for nverts in self.vertices['nverts']]

        # Initialize new columns
        self.edges['radius'] = np.zeros(len(self.edges))
        self.edges['rho'] = tuple((0,0) for _ in range(len(self.edges)))
        self.edges['fitenergy'] = np.zeros(len(self.edges))
        self.edges['tension'] = np.zeros(len(self.edges))
        self.cells['pressure'] = np.zeros(len(self.cells))
        self.cells['qx'] = np.zeros(len(self.cells))
        self.cells['qy'] = np.zeros(len(self.cells))
        self.cells['theta'] = np.zeros(len(self.cells))

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

        self.eng = matlab.engine.start_matlab()
        self.eng.cd('./src')


    def fit_circle(self):
        """

        fit circle to each edge
        if edge is too flat, fit line instead

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

            def energyfunc(x):
                return np.mean(np.power(np.sqrt(np.sum(np.power(delta-(x*nB), 2), axis=1)) - np.sqrt(np.power(x, 2) + np.power(L0, 2)), 2))

            if not np.isnan(y0):
                res = scipy.optimize.minimize(energyfunc, y0, tol=1e-8)
            else:
                res = scipy.optimize.minimize(energyfunc, 0, tol=1e-8)
            y = res.x
            E = res.fun

            linedistance = np.mean(np.power(IP, 2))
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

        recursively removes fourfold (or greater) vertices by moving vertex apart in direction of greatest variance

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
                    self.vertices.at[num_v,'fourfold'] = 0

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
                                                    'radius':np.array([]), 'rho':np.array([])},ignore_index=True)

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
        boundary_verts = self.cells.at[0,'nverts']

        # iterate through all non-boundary verts
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

                theta12 = np.mod(np.arctan2(z12[2], np.dot(r[0,:], r[1,:])), 2*np.pi)
                theta23 = np.mod(np.arctan2(z23[2], np.dot(r[1,:], r[2,:])), 2*np.pi)
                theta31 = np.mod(np.arctan2(z31[2], np.dot(r[2,:], r[0,:])), 2*np.pi)

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

        prepare data for tension inference
        remove fourfold vertices, fit circular arcs to edges, make vertices convex

        """

        # Fit circular arcs to each edge
        self.fit_circle()

        # Recursively remove fourfold vertices by moving them apart
        self.remove_fourfold()

        # Inference cannot handle concave vertices (with one angle greater than pi) so remove these
        self.make_convex()
        return


    def classify_cells(self):
        """

        determine which cells are involved in tension inference
        initialize q as cell centroids

        """

        # This is enough for now, but may need to update to deal with holes
        self.bulk_cells = np.array(range(1,len(self.cells)))

        boundary_cells = np.unique(self.cells.at[0, 'ncells'])

        # Remove boundary cells and cells surrounded by boundary cells from bulk cells
        self.bulk_cells = self.bulk_cells[np.isin(self.bulk_cells, boundary_cells, invert=True)]
        bad_cells = np.array([])
        for cell in self.bulk_cells:
            if np.sum(np.isin(self.cells.at[cell, 'ncells'], self.bulk_cells)) == 0:
                bad_cells = np.append(bad_cells, cell)
        self.bulk_cells = self.bulk_cells[np.isin(self.bulk_cells, bad_cells, invert=True)]

        # This excludes vertices surrounded by boundary cells; is this a requirement for improved inference?
#        self.bulk_vertices = np.unique(np.concatenate([self.cells.at[cell, 'nverts'] for cell in self.bulk_cells]))
        # Try another way instead - bulk cells are all verta that are not part of external cell
        self.bulk_vertices = np.arange(0,len(self.vertices))
        self.bulk_vertices = self.bulk_vertices[np.isin(np.arange(0,len(self.vertices)), self.cells.at[0, 'nverts'], invert=True)]

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

        compute difference operators to enable vectorized operations

        """
        # Build cell adjacency matrix
        adj_mat = np.zeros((len(self.involved_cells), len(self.involved_cells)))
        num_edges = 0

        for i in range(len(self.involved_cells)):
            cell = self.involved_cells[i]
            for ncell in self.cells.at[cell, 'ncells']:
                j = np.ravel(np.where(self.involved_cells == ncell))
                if j.size > 0 and adj_mat[i, j] == 0:
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

        estimate tension vector tau from vector t

        """

        self.build_diff_operators()

        self.involved_edges = -1 * np.ones(self.dC.shape[0], dtype=int)
        for i in range(self.dC.shape[0]):
            edge_cells = self.involved_cells[np.ravel(np.where(self.dC[i,] != 0))]
            self.involved_edges[i] = np.ravel(np.where(np.sum([np.sort(self.edges.at[edge, 'cells']) == np.sort(edge_cells) for edge in range(len(self.edges))], axis=1) == 2))

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
            else:
                tau_1[e,:] = -np.divide(e_chord[e,:], np.linalg.norm(e_chord[e,:]))
                tau_2[e,:] = np.divide(e_chord[e,:], np.linalg.norm(e_chord[e,:]))
        return e_cells, tau_1, tau_2, r1, r2


    def estimate_pressure(self, q, e_cells, tau_1, tau_2, r1, r2):
        """

        initial estimate of pressure

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

        L = np.vstack((L1, L2, np.array(np.divide(np.ones(q.shape[0]), q.shape[0]))))

        p = np.linalg.lstsq(L,b)[0]
        p = np.divide(p, np.mean(p))
        return p


    def generate_circular_arcs(self):
        """

        construct circular arc for each edge and use these instead of raw segmented edges for minimization

        """

        self.avg_edge_length = int(np.median([self.edges.at[edge, 'pixels'].shape[0] for edge in self.involved_edges]))
        self.edgearc_x = np.zeros((len(self.involved_edges), self.avg_edge_length))
        self.edgearc_y = np.zeros((len(self.involved_edges), self.avg_edge_length))

        for i in range(len(self.involved_edges)):
            r = np.array([self.vertices.at[self.edges.at[self.involved_edges[i], 'verts'][0], 'coords'],
                          self.vertices.at[self.edges.at[self.involved_edges[i], 'verts'][1], 'coords']])
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
            else :
                chord = r[1,:] - r[0,:]

                spacing = np.linspace(0, 1, self.avg_edge_length)

                self.edgearc_x[i] = r[0,0] + spacing*chord[0]
                self.edgearc_y[i] = r[0,1] + spacing*chord[1]
        return


    def estimate_theta(self, x):
        """

        Initialize theta, defined as p_a * z^2_a for each cell a

        :param x: A [num_cells x 3] numpy array containing q in x[0:2,:] and p in x[2,:]
        :return: A [num_cells x 1] numpy array containing initialized values of theta
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

            r[i] = np.power(np.mean(np.array([np.linalg.norm(r1 - rho[i]), np.linalg.norm(r2 - rho[i])])), 2)
            r_flat[i] = p[np.where(self.dC[i,:] == 1)] * p[np.where(self.dC[i,:] == -1)] * q_sq[i]

        dP = np.matmul(self.dC, p)
        r = np.multiply(r, np.power(dP, 2))

        A = np.multiply(self.dC.T, dP).T
        b = r_flat - r

        theta = np.linalg.lstsq(np.vstack((A, np.ones(A.shape[1]))),np.concatenate((b, np.array([0]))))[0]

        return theta


    def initial_minimization(self):
        """

        Find initial values for p, q and theta

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

        print("Initial minimization")
        # Perform initial minimization for p, q and theta
        q0, p0, theta0 = self.initial_minimization()
        X0 = np.vstack([q0.T, theta0.T, p0]).T

        X0_mat = matlab.double(X0.tolist())
        bCells_mat = matlab.double((self.cell_pairs+1).tolist())
        d0_mat = matlab.double(self.dC.tolist())
        rBX_mat = matlab.double((self.edgearc_x).tolist())
        rBY_mat = matlab.double((self.edgearc_y).tolist())

        res = self.eng.main_minimization(bCells_mat, d0_mat, rBX_mat, rBY_mat, X0_mat, float(self.avg_edge_length))

        X = np.array(res)
        X = X.reshape(X0.shape, order='F')

        q = X[:,0:2]
        p = X[:,3]
        theta = X[:,2]

        self.eng.quit()

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

        self.cells.pressure[1:] = p[np.argsort(self.involved_cells)]
        self.cells.qx[1:] = q[np.argsort(self.involved_cells),0]
        self.cells.qy[1:] = q[np.argsort(self.involved_cells),1]
        self.cells.theta[1:] = theta[np.argsort(self.involved_cells),]

        for i in range(len(self.edges)):
            edge_cells = self.edges.at[i, 'cells']

            if all(np.isin(edge_cells, self.involved_cells)):
                cell_ind1 = np.where(self.involved_cells == edge_cells[0])[0]
                cell_ind2 = np.where(self.involved_cells == edge_cells[1])[0]

                edge_ind = np.where(np.squeeze((self.dC[:,cell_ind1] != 0) & (self.dC[:,cell_ind2] != 0), axis=1))[0]
                self.edges.at[i, 'tension'] = T[edge_ind]
        return

    def return_tensions(self):
        return self.edges.tension.to_numpy()

    def return_pressures(self):
        return self.cells.pressure.to_numpy()[1:]

    def draw_tension(self, mask):
        img = mask

        fig, ax = plt.subplots(1,1,figsize=np.divide(mask.shape,72), dpi=72)
        ax.imshow(img)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        maxT = np.max(self.return_tensions())
        minT = np.min(self.return_tensions())
        colourmap = cm.get_cmap('inferno')

        for e in range(len(self.edges)):
            radius = self.edges.at[e, 'radius']
            tension_norm = np.divide(self.edges.at[e, 'tension'], maxT-minT)
            colour = colourmap(tension_norm)

            if np.isinf(radius):

                v = np.array([self.vertices.at[self.edges.at[e, 'verts'][0], 'coords'], self.vertices.at[self.edges.at[e, 'verts'][1], 'coords']])
                points = np.array([np.linspace(v[0,0], v[1,0], len(self.edges.at[e, 'pixels'])),
                                   np.linspace(v[0,1], v[1,1], len(self.edges.at[e, 'pixels']))]).T
                ax.plot(points[:,0], points[:,1], lw=3, color=colour)
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
                ax.plot(points[:,0], points[:,1], lw=3, color=colour)

        plt.show()
        return