import matplotlib.pyplot as plt
import numpy as np
from cellpose import models, utils, plot
from scipy.ndimage import generic_filter
from scipy.spatial.distance import cdist
from scipy.optimize import minimize, least_squares, LinearConstraint
import pandas as pd
from skimage import measure, color
from matplotlib import cm, patches, colors
import matplotlib
from src.segment import Segmenter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matlab
import matlab.engine

class VMSI():

    def __init__(self, vertices, cells, edges, width, height, verbose):
        self.vertices = vertices
        self.cells = cells
        self.edges = edges
        self.width = width
        self.height = height
        self.verbose = verbose

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

        # Initialise Matlab engine
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
                res = minimize(energyfunc, y0, tol=1e-8)
            else:
                res = minimize(energyfunc, 0, tol=1e-8)
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
        remove fourfold vertices, fit circles, make vertices convex

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
        self.bulk_cells = np.array(range(0,len(self.cells)))

        boundary_cells = np.unique(np.concatenate([self.cells.at[0, 'ncells'], np.where(self.cells['holes'].to_numpy()[1:])[0]]))

        # Remove boundary cells and cells surrounded by boundary cells from bulk cells
        all_threefold = np.array([not(any(self.vertices.loc[self.cells.at[cell, 'nverts'], 'fourfold'].to_numpy())) for cell in range(len(self.cells))])
        self.bulk_cells = self.bulk_cells[all_threefold]
        self.bulk_cells = self.bulk_cells[np.isin(self.bulk_cells, boundary_cells, invert=True)]

        bad_cells = np.array([])
        for cell in self.bulk_cells:
            if np.sum(np.isin(self.cells.at[cell, 'ncells'], self.bulk_cells)) == 0:
                bad_cells = np.append(bad_cells, cell)
        self.bulk_cells = self.bulk_cells[np.isin(self.bulk_cells, bad_cells, invert=True)]

        # This excludes vertices surrounded by boundary cells; is this a requirement for improved inference?
        self.bulk_vertices = np.unique(np.concatenate([self.cells.at[cell, 'nverts'] for cell in self.bulk_cells]))
        # Try another way instead - bulk cells are all verts that are not part of external cell
        #        self.bulk_vertices = np.arange(0,len(self.vertices))
        #        self.bulk_vertices = self.bulk_vertices[np.isin(np.arange(0,len(self.vertices)), self.cells.at[0, 'nverts'], invert=True)]

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
        edge_cells = self.edges.cells.to_list()

        for i in range(len(self.involved_cells)):
            cell = self.involved_cells[i]
            for ncell in self.cells.at[cell, 'ncells']:
                j = np.ravel(np.where(self.involved_cells == ncell))
                # Ensure that edge between neighbouring cells actually exists
                if j.size > 0 and adj_mat[i, j] == 0 and any(np.all(np.sort(np.array([cell, ncell])) == edge_cells, axis=1)):
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
        return


    def estimate_tau(self):
        """

        estimate tension vector tau from vector t

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

        if self.verbose:
            print("Initial minimization")
        # Perform initial minimization for p, q and theta
        q0, p0, theta0 = self.initial_minimization()
        X0 = np.vstack([q0.T, theta0.T, p0]).T

        X0_mat = matlab.double(X0.tolist())
        bCells_mat = matlab.double((self.cell_pairs+1).tolist())
        d0_mat = matlab.double(self.dC.tolist())
        rBX_mat = matlab.double((self.edgearc_x).tolist())
        rBY_mat = matlab.double((self.edgearc_y).tolist())

        if self.verbose:
            print("Main minimization")
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

        self.cells.pressure[np.sort(self.involved_cells)] = p[np.argsort(self.involved_cells)]
        self.cells.qx[np.sort(self.involved_cells)] = q[np.argsort(self.involved_cells),0]
        self.cells.qy[np.sort(self.involved_cells)] = q[np.argsort(self.involved_cells),1]
        self.cells.theta[np.sort(self.involved_cells)] = theta[np.argsort(self.involved_cells),]

        for i in range(len(self.edges)):
            edge_cells = self.edges.at[i, 'cells']

            if all(np.isin(edge_cells, self.involved_cells)):
                cell_ind1 = np.where(self.involved_cells == edge_cells[0])[0]
                cell_ind2 = np.where(self.involved_cells == edge_cells[1])[0]

                edge_ind = np.where(np.squeeze((self.dC[:,cell_ind1] != 0) & (self.dC[:,cell_ind2] != 0), axis=1))[0]
                self.edges.at[i, 'tension'] = T[edge_ind]
        return

    def return_tensions(self):
        return self.edges.tension.values

    def return_pressures(self):
        return self.cells.pressure.values[1:]

    def compute_stresstensor(self, mode=0):
        """

        Compute stress tensor for each cell from tensions and pressures

        :param mode: 0 or 1, specifies how calculation is performed
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

        # Implement mode 0 first since that looks more sensible
        if mode == 1:
            sigma = np.zeros([len(self.bulk_cells),3])
            for c in self.bulk_cells:
                c_verts = self.cells.at[c, 'nverts']
                for v in c_verts:
                    r0 = rv[self.involved_vertices==v,:]
        elif mode == 0:
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

            A = np.array(self.cells.area.to_list())[np.sort(self.involved_cells)]
            sigma = np.divide(sigma.T, A)
            for c in range(len(self.involved_cells)):
                self.cells.at[self.involved_cells[c], 'stress'] = sigma[:,c]
        return


    def plot(self, options, mask=np.array([])):
        """
        :type options: list
        :param mask: image on which to overlay plotted objects. If plotting pressure, must be labelled, segmented image.
        :param options: list of options for plotting. Available options are: 'stress', 'pressure', 'tension', 'CAP'

        """
        options = [option.lower() for option in options]
        if mask.size > 0:
            if np.isin('pressure', options):
                colourmap = cm.get_cmap('viridis')
                centroids = np.array(self.cells.centroids.tolist())[1:]
                pressures = self.cells.pressure.to_numpy()
                img_centroids = np.array([np.flip(regionprops.centroid) for regionprops in measure.regionprops(mask)])
                colours = [(0,0,0) for _ in range(img_centroids.shape[0])]

                maxP = np.max(pressures)

                for i in range(centroids.shape[0]):
                    img_index = int((np.where(centroids[i,0] == img_centroids[:,0]) and np.where(centroids[i,1] == img_centroids[:,1]))[0])
                    colours[img_index] = colourmap(np.divide(pressures[i], maxP))
                img = color.label2rgb(mask, mask, colors=colours, alpha=1)
            else:
                colourmap = cm.get_cmap('Set3')
                colours = np.array([colourmap(np.mod(i,12)) for i in range(len(np.unique(mask)))])
                colours = np.vstack(([0,0,0,0], colours))
                img = color.label2rgb(mask, mask, colors=colours, alpha=0.7)
        else:
            if np.isin('pressure', options):
                return("Error: 'pressure' plotting specified without providing labelled, segmented image.")
            mask = np.zeros((self.height, self.width))
            img = color.label2rgb(mask, mask, colors=[(0,0,0)], alpha=1)
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
            cax1 = divider.append_axes("right", size="5%", pad=1)
            p_cb = plt.colorbar(mappable=cm.ScalarMappable(norm=colors.Normalize(np.min(pressures), np.max(pressures)), cmap=colourmap), cax=cax1)
            p_cb.set_label('Pressure')

        if all(np.isin(options, ['stress', 'pressure', 'tension', 'cap'])):
            for option in options:
                if option == 'stress':
                    stress = np.array([self.cells.at[cell, 'area'] * np.array([[self.cells.at[cell, 'stress'][0], self.cells.at[cell, 'stress'][1]],[self.cells.at[cell, 'stress'][1], self.cells.at[cell, 'stress'][2]]]) for cell in range(len(self.cells))])
                    eigvals, eigvects = np.linalg.eig(stress)
                    scalefct = np.sqrt(np.median(np.multiply(eigvals[:,0], eigvals[:,1])))

                    for i in range(len(self.cells)):
                        cell = i
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
                    maxT = np.max(self.return_tensions())
                    minT = np.min(self.return_tensions())
                    colourmap = cm.get_cmap('inferno')

                    for e in range(len(self.edges)):
                        if self.edges.at[e, 'tension'] > 0:
                            radius = self.edges.at[e, 'radius']
                            tension_norm = np.divide(self.edges.at[e, 'tension'], maxT-minT)
                            colour = colourmap(tension_norm)

                            if np.isinf(radius):

                                v = np.array([self.vertices.at[self.edges.at[e, 'verts'][0], 'coords'], self.vertices.at[self.edges.at[e, 'verts'][1], 'coords']])
                                points = np.array([np.linspace(v[0,0], v[1,0], len(self.edges.at[e, 'pixels'])),
                                                   np.linspace(v[0,1], v[1,1], len(self.edges.at[e, 'pixels']))]).T
                                ax.plot(points[:,0], points[:,1], lw=5, color=colour)
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
                                ax.plot(points[:,0], points[:,1], lw=5, color=colour)
                    cax2 = divider.append_axes("right", size="5%", pad=1)
                    t_cb = plt.colorbar(mappable=cm.ScalarMappable(norm=colors.Normalize(minT, maxT), cmap=colourmap), cax=cax2)
                    t_cb.set_label('Tension')
                elif option == 'cap':
                    for e in range(len(self.edges)):
                        radius = self.edges.at[e, 'radius']

                        if np.isinf(radius):

                            v = np.array([self.vertices.at[self.edges.at[e, 'verts'][0], 'coords'], self.vertices.at[self.edges.at[e, 'verts'][1], 'coords']])
                            points = np.array([np.linspace(v[0,0], v[1,0], len(self.edges.at[e, 'pixels'])),
                                               np.linspace(v[0,1], v[1,1], len(self.edges.at[e, 'pixels']))]).T
                            ax.plot(points[:,0], points[:,1], lw=5, color='b')
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
                            ax.plot(points[:,0], points[:,1], lw=5, color='b')
        else:
            return("Invalid plotting option. Only valid options are 'stress', 'pressure', 'tension', 'CAP'.")
        plt.show()
        return

    def output_results(self, filename, metrics=None):
        """
        Outputs force inference and morphometric quantities as csv.
        @:param metrics: list of metrics to output.
        """
        if metrics is None:
            metrics = ['centroids','pressure','stress','eccentricity','inertia','perimeter','axis_major','axis_minor','feret_d','equiv_diam_area','moments_hu','bbox','orientation']
        out_df = self.cells.loc[self.involved_cells, metrics]
        out_df.to_csv(filename, sep=',',index=False)
        return

def run_VMSI(img, is_labelled=False, tile=True, verbose=False, overlap=0.3):
    """
    Main function to run stress inference.
    :param verbose: (bool) whether to provide detailed output.
    :param img: (numpy array) segmented image.
    :param is_labelled: (bool) whether all cells in image are labelled.
    :param tile: (bool) whether to break image into tiles for faster inference. By default, this is True.
    :return: VMSI object containing the inferred network.
    """

    # TODO: process img to create single numpy array with integer values (i.e. deal with RGB masks)
    if tile:
        if not is_labelled:
            img = measure.label(img)
        tiles, offset, adj_tiles = create_image_tiles(img, overlap=overlap)
        models = []

        pairwise_tensions = []
        pairwise_pressures = []
        for tile in tiles:
            # process segmented image for input into VMSI
            seg = Segmenter(masks=tile, labelled=is_labelled)
            VMSI_obj, labelled_mask = seg.process_segmented_image()
            # create the model
            model = VMSI(vertices=VMSI_obj.V_df, cells=VMSI_obj.C_df, edges=VMSI_obj.E_df, height=tile.shape[0], width=tile.shape[1], verbose=verbose)
            # fit the model parameters
            model.fit()
            models.append(model)
        for i in range(len(adj_tiles)):
            pair = adj_tiles[i]
            model_1 = models[pair[0]]
            model_2 = models[pair[1]]

            # Match pressures
            model1_cells = np.array(model_1.cells.centroids.tolist())[1:] + offset[pair[0]]
            model2_cells = np.array(model_2.cells.centroids.tolist())[1:] + offset[pair[1]]

            cell_pwdist = cdist(model1_cells, model2_cells)
            # We expect matching cells to have assigned centroids that are <=10px apart in the two tiles
            # This may need to be changed depending on image size
            matching_cells = np.where(cell_pwdist<=10)
            pressures = np.array([model_1.return_pressures()[matching_cells[0]], model_2.return_pressures()[matching_cells[1]]])
            pairwise_pressures.append(pressures[:,np.all(pressures>0, axis=0)])

            # Match edges
            model1_edge_r1 = np.array(model_1.vertices.coords[np.array(model_1.edges.verts.tolist())[:,0]].tolist()) + offset[pair[0]]
            model1_edge_r2 = np.array(model_1.vertices.coords[np.array(model_1.edges.verts.tolist())[:,1]].tolist()) + offset[pair[0]]
            model2_edge_r1 = np.array(model_2.vertices.coords[np.array(model_2.edges.verts.tolist())[:,0]].tolist()) + offset[pair[1]]
            model2_edge_r2 = np.array(model_2.vertices.coords[np.array(model_2.edges.verts.tolist())[:,1]].tolist()) + offset[pair[1]]

            r1_pwdist = cdist(model1_edge_r1, model2_edge_r1)
            r2_pwdist = cdist(model1_edge_r2, model2_edge_r2)

            matching_edges = np.where(np.logical_and(r1_pwdist <= 10, r2_pwdist <= 10))
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

        model = merge_models(models, p_scale, t_scale, offset, img, verbose=verbose)
    else:
        # process segmented image for input into VMSI
        seg = Segmenter(masks=img, labelled=is_labelled)
        VMSI_obj, labelled_mask = seg.process_segmented_image()
        # create the model
        model = VMSI(vertices=VMSI_obj.V_df, cells=VMSI_obj.C_df, edges=VMSI_obj.E_df, height=img.shape[0], width=img.shape[1], verbose=verbose)
        # fit the model parameters
        model.fit()
        # compute stress tensor
        model.compute_stresstensor()
    return model

def create_image_tiles(img, cells_per_tile=100, overlap=0.3):
    ncells = len(measure.regionprops(img))

    if (ncells/cells_per_tile <= 2):
        return 'Error: image is too small for tiling using the specified number of cells per tile'
    else:
        ntiles = 2*np.round(ncells/(2*cells_per_tile))
        if ntiles == 2:
            if img.shape[0] > img.shape[1]:
                xsplits = 1
                ysplits = 2
            else:
                xsplits = 2
                ysplits = 1
        else:
            xsplits = int(np.round(np.sqrt(ntiles*img.shape[1]/img.shape[0])))
            ysplits = int(np.round(xsplits * img.shape[0] / img.shape[1]))

        if xsplits*ysplits != ntiles:
            return(f'Error! xsplits={xsplits} and ysplits={ysplits} do not make up {ntiles} tiles')

        xstep = int(np.round(img.shape[1]/xsplits))
        ystep = int(np.round(img.shape[0]/ysplits))
        xoverlap = int(np.round(overlap * xstep))
        yoverlap = int(np.round(overlap * ystep))
        offset = []
        tiles = []
        adj_tiles = [[x*ysplits+y, (x+1)*ysplits+y] for x in range(xsplits-1) for y in range(ysplits)] + \
                    [[x*ysplits+y, x*ysplits+y+1] for x in range(xsplits) for y in range(ysplits-1)]

        for i in range(xsplits):
            for j in range(ysplits):
                start_x = i*xstep
                start_y = j*ystep
                end_x = (i+1)*xstep
                end_y = (j+1)*ystep

                if i != 0:
                    start_x = start_x - xoverlap
                if i != xsplits-1:
                    end_x = end_x + xoverlap
                if j != 0:
                    start_y = start_y - yoverlap
                if j != ysplits-1:
                    end_y = end_y + yoverlap

                tiles.append(img[start_y:end_y,start_x:end_x])
                offset.append(np.array([start_x, start_y]))
    return tiles, offset, adj_tiles

def merge_models(models, p_scale, t_scale, offset, img, verbose):
    # process segmented image for input into VMSI
    seg = Segmenter(masks=img, labelled=True)
    VMSI_obj, labelled_mask = seg.process_segmented_image()
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