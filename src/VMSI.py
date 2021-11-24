import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from cellpose import models
from cellpose import utils
from cellpose import plot
from scipy.ndimage import generic_filter
import itertools
from tqdm import tqdm
from scipy.optimize import minimize, leastsq
import pandas as pd


class VMSI():
    
    def __init__(self, vertices, cells, edges, width=500, height=500):
        self.vertices = vertices
        self.cells = cells
        self.edges = edges
        self.width = width
        self.height = height

        # Mark fourfold vertices
        self.vertices['fourfold'] = [(np.shape(nverts)[0] != 3) for nverts in self.vertices['nverts']]


    def fit_circle(self):
        """

        fit circle to each edge
        if edge is too flat, fit line instead

        """
        for i in range(len(self.edges)):
            r1 = np.array(self.vertices.loc[self.edges.loc[i]['verts'][0]-1].coords)
            r2 = np.array(self.vertices.loc[self.edges.loc[i]['verts'][1]-1].coords)

            edge_pixels = [np.unravel_index(pixel, (self.width, self.height)) for pixel in self.edges.loc[i]['pixels']]

            nB = np.matmul(np.array([[0, 1], [-1, 0]]), r1 - r2)
            D = np.sqrt(np.sum(np.power(nB, 2)))
            nB = np.divide(nB, D)
            x0 = 0.5*(r1 + r2)

            delta = edge_pixels - x0
            IP = (delta[:,0] * nB[0]) + (delta[:,1] * nB[1])
            L0 = D/2

            A = 2*np.sum(np.power(IP, 2))
            B = np.sum((np.sum(np.power(delta, 2), axis=1) - np.power(L0, 2)) * IP)
            y0 = np.divide(B, A)

            def energyfunc(x):
                return np.mean(np.power(np.sqrt(np.sum(np.power(delta-(x*nB), 2), axis=1)) - np.sqrt(np.power(x, 2) + np.power(L0, 2)), 2))

            if not np.isnan(y0):
                res = scipy.optimize.minimize(energyfunc, y0)
            else:
                res = scipy.optimize.minimize(energyfunc, 0)
            y = res.x
            E = res.fun

            linedistance = np.mean(np.power(IP, 2))
            if (E < linedistance & len(edge_pixels) > 3):
                self.edges.loc[i]['radius'] = np.sqrt(np.power(y, 2) + np.power(L0, 2))
                self.edges.loc[i]['rho'] = x0 + (y * nB)
                self.edges.loc[i]['fitenergy'] = E
            else:
                self.edges.loc[i]['radius'] = np.Inf
                self.edges.loc[i]['rho'] = np.array([np.Inf, np.Inf])
                self.edges.loc[i]['fitenergy'] = linedistance

        return

    def remove_fourfold(self):
        """

        recursively removes fourfold (or greater) vertices by moving vertex apart in direction of greatest variance

        """

        for v in range(len(self.vertices)):
            if (np.shape(self.vertices['nverts'][v])[0] > 3  and not (1 in self.vertices['ncells'][v])):
                while (np.shape(self.vertices['nverts'][v])[0] > 3):
                    num_v = len(self.vertices)
                    num_e = len(self.edges)

                    nverts = self.vertices['nverts'][v]
                    nedges = self.vertices['edges'][v]
                    ncells = self.vertices['ncells'][v]

                    R = self.vertices['coords'][nverts]
                    rV = self.vertices['coords'][v]

                    R = R - np.mean(R, axis=1)
                    I = R * R.T

                    W, V = np.linalg.eig(I)
                    direction = V[:,np.argmax(W)]

                    rV1 = rV + (direction/2)
                    rV2 = rV - (direction/2)

                    # set positive neighbour vertices to the 2 vertices closest to the direction of vertex movement
                    # all other neighbour vertices remain with negative vertex
                    indices = np.argsort(np.dot(R, direction))[-2:]
                    pos_verts = np.zeros_like(indices)
                    pos_verts[indices] = 1
                    neg_verts = 1 - pos_verts

                    # change vertex with current index to negative vertex
                    self.vertices['coords'][v] = rV2
                    self.vertices['nverts'][v] = np.array([nverts[neg_verts.astype('bool')], num_v+1])
                    self.vertices['fourfold'][v] = (np.shape(self.vertices['nverts'][v])[0] > 3)

                    neg_cells = ncells[[(np.isin(self.vertices['nverts'][v], self.cells['nverts'][cell]) == 2) for cell in ncells]]

                    # add positive vertex
                    self.vertices['coords'][num_v+1] = rV1
                    self.vertices['nverts'][num_v+1] = np.array([nverts[pos_verts.astype('bool')], v])
                    self.vertices['fourfold'][num_v+1] = 0

                    pos_cell = ncells[[(np.isin(self.vertices['nverts'][num_v+1], self.cells['nverts'][cell]) == 2) for cell in ncells]]

                    # update new positive vertex index for neighbour vertices
                    for vert in nverts[pos_verts.astype('bool')]:
                        self.vertices['nverts'][vert][self.vertices['nverts'][vert] == v] = num_v+1

                    joint_cells = ncells[not (np.isin(ncells, np.array([pos_cell, neg_cells])))]

                    self.vertices['ncells'][v] = np.array([joint_cells, neg_cells])
                    self.vertices['ncells'][num_v+1] = np.array([joint_cells, pos_cell])

                    # update current edges
                    neg_edges = nedges[neg_verts.astype('bool')]
                    pos_edges = nedges[pos_verts.astype('bool')]

                    self.edges['verts'][pos_edges[0]] = num_v+1
                    self.edges['verts'][pos_edges[1]] = num_v+1

                    # create new edge between new vertices
                    # edge is only one pixel long so no need to add pixels
                    self.edges['verts'][num_e+1] = np.array([v, num_v+1])
                    self.edges['cells'][num_e+1] = joint_cells
                    self.edges['pixels'][num_e+1] = np.array([])
                    self.edges['radius'][num_e+1] = np.Inf
                    self.edges['rho'][num_e+1] = np.array([np.Inf, np.Inf])

                    # update edges of new vertices
                    self.vertices['edges'][v] = np.array([neg_edges, num_e+1])
                    self.vertices['edges'][num_v+1] = np.array([pos_edges, num_e+1])

                    # update cells

                    # update pos cells
                    for cell in pos_cell:
                        self.cells['nverts'][cell][self.cells['nverts'][cell] == v] = num_v + 1
                        self.cells['ncells'][cell] = self.cells['ncells'][cell][neg_cells not in self.cells['ncells'][cell]]
                    # update neg cells
                    for cell in neg_cells:
                        self.cells['ncells'][cell] = self.cells['ncells'][cell][pos_cell not in self.cells['ncells'][cell]]
                    # update joint cells
                    for cell in joint_cells:
                        self.cells['nverts'][cell] = np.array([self.cells['nverts'][cell], num_v+1])
                        self.cells['nverts'][cell] = self.cells['nverts'][cell]+1
        return

    def transform(self, q, z, p):
        """
        
        transform from points to CAP tiling
        via Equations 6 and 7
        
        """
        
        center = {alpha : {beta: None for beta in self.cells} for alpha in self.cells}
        radius = {alpha : {beta: None for beta in self.cells} for alpha in self.cells}
        
        for (alpha, beta) in self.cell_pairs:
            center[alpha][beta] = (p[beta-1]*q[beta-1] - p[alpha-1]*q[alpha-1]) / (p[beta-1] - p[alpha-1])
            radius[alpha][beta] = np.sqrt(((p[alpha-1]*p[beta-1]) * (np.linalg.norm(q[alpha-1] - q[beta-1])**2))/(p[alpha-1] - p[beta-1])**2 \
                                          - (p[alpha-1] * (z[alpha-1]**2) - p[beta-1] * (z[beta-1]**2))/(p[alpha-1] - p[beta-1]))
        return center, radius

    def prepare_data(self):
        '''

        prepare data for tension inference
        remove fourfold vertices, fit circles, identify border cells

        '''

        # Fit circular arcs to each edge
        self.fit_circle()

        # Recursively remove fourfold vertices by moving them apart
        self.remove_fourfold()
        return
        
    
    def energy(self, theta):
        """
        
        theta is a flattened num_cells x 4 dimensional vector
        
        """
                
        # retrieve the 'encoded' information
        q, z, p = self.extract_values(theta)
        
        # transform
        center, radius = self.transform(q, z, p) 
        
        # compute the loss
        energy = 0
        for (alpha, beta) in self.cell_pairs:
            edge = self.edges[alpha][beta]
            for pixel in edge:
                energy += (np.linalg.norm(pixel - center[alpha][beta]) - radius[alpha][beta])**2
        
        return energy/(2*self.num_edges)
        
    def is_square(self, i):
        x = i // 2
        visited = set([x])
        while x * x != i:
            x = (x + (i // x)) // 2
            if x in visited:
                return False
            visited.add(x)
        return True
        
    def extract_values(self, theta):
        N = len(theta)//4
        x = theta[:N]
        y = theta[N:2*N]
        z = theta[2*N:3*N]
        p = theta[3*N:4*N]
        q = np.array([x,y]).T
        return q, z, p
    
    def initialize_points(self):
        x = [] ; y = [] ; z = [] ; p = []
        
        for alpha in self.cells:
            center = self.barycenters[alpha]
    
            x.append(center[0]) ; y.append(center[1]) ; z.append(0)
            p.append(np.random.uniform(0.001, 0.005))
                
        return np.array(x + y + z + p)
    
    def get_vertices(self, alpha, beta):
        """
        
        return the two vertices between any two cells
        
        """
        
        edge = self.edges[alpha][beta]
        
        # find the edge endpoints
        edge_mean = np.mean(edge, axis=0)
        start_ind = np.argmax(np.array([np.linalg.norm(x - edge_mean) for x in edge]))
        end_ind = np.argmax(np.array([np.linalg.norm(x - edge[start_ind]) for x in edge]))
        start = np.array(edge[start_ind]) ; end = np.array(edge[end_ind])
                    
        return [start, end]
    
    def get_tangents(self, alpha, beta):
        """
        finds the normalized tangents at the two vertices in the edge between alpha and beta
        """
        v1, v2 = self.vertices[alpha][beta]
        
        # find closest points in the edge to each of the vertices and find line that goes through them
        edge = [e for e in self.edges[alpha][beta] if list(e) not in [list(v1), list(v2)]]

        v1_b = np.mean(np.array([edge[i] for i in np.array([np.linalg.norm(x - v1) for x in edge]).argsort()[:5]]), axis=0)
        v2_b = np.mean(np.array([edge[i] for i in np.array([np.linalg.norm(x - v2) for x in edge]).argsort()[:5]]), axis=0)

        # compute tangents
        t1 = (v1_b - v1) / np.linalg.norm(v1_b - v1)
        t2 = (v2_b - v2) / np.linalg.norm(v2_b - v2)
        
        return [t1, t2]
        
    
    def initialize(self):
        """
        
        Minimization to determine initial (p, q) -- we want the vector pointing from the center each of the 
        vertices to be orthogonal to the tangents. Minimization of C1 with some constraints to avoid trivial 
        solutions
        
        """
        
        # define the shorthand t_i where i in {1, 2} denotes the vertex in question between cells alpha and beta. It is 
        # the vector from the center of cell alpha to vertex i in between alpha and beta
        t = lambda alpha, beta, q, p, i : (p[alpha-1] - p[beta-1])*self.vertices[alpha][beta][i] - (p[alpha-1]*q[alpha-1] - p[beta-1]*q[beta-1])
        
        
        def extract(theta):
            # get p and q from theta used in the following minimization
            N = len(theta)//3
            x = theta[:N] ; y = theta[N:2*N] ; q = np.array([x,y]).T
            p = theta[2*N:3*N]
            return q, p

        theta0_with_z = list(self.initialize_points()) ; N = len(theta0_with_z)//4
        theta0 = np.array(theta0_with_z[:2*N] + theta0_with_z[3*N:4*N])  # only keep q and p
        
        bounds = []
        N = len(theta0)//3
        for i in range(len(theta0)):
            if i <= N: # x
                bounds.append((max(0, theta0[i]-10), min(theta0[i]+10, self.width)))
            elif i <= 2*N: # y 
                bounds.append((max(0, theta0[i]-10), min(theta0[i]+10, self.height)))
            else: # p
                bounds.append((0.001, 0.005))

        # minimize to find the optimal q and p
        print("Finding the initialization of q and p")
        q, p = extract(theta0)
        optimal = minimize(E_initial, theta0, options={'disp':True}, bounds=bounds).x
        print(E_initial(optimal))

        q, p =  extract(optimal)
        x = q.T[0] ; y = q.T[1]
        
        # find z. For NOW, set to be 0 -- later, we will implement the actual equation
        z = [0 for _ in range(len(p))]

        return np.array(list(x) + list(y) + list(z) + list(p))
        
    def load_constraints(self, theta0):
        """
        come up with the constraints and bounds
        """
        constraints = []
        bounds = []
            
        N = len(theta0)//4
        
        for i in range(len(theta0)):
            if i <= N: # x
                bounds.append((max(0, theta0[i]-10), min(theta0[i]+10, self.width)))
            elif i <= 2*N: # y 
                bounds.append((max(0, theta0[i]-10), min(theta0[i]+10, self.height)))
            elif i <= 3*N: # z
                bounds.append((0, 0.5))
            else: # p
                bounds.append((0.001, 0.005))
                
        # non-imaginary constraint, i.e. force things inside of the root in Equation 7 to not be negative
        def imaginary_constraint(alpha, beta):
            def constraint(theta):
                N = len(theta)//4
                x = theta[:N]
                y = theta[N:2*N]
                z = theta[2*N:3*N]
                p = theta[3*N:4*N]
                q = np.array([x, y]).T

                return ((p[alpha-1]*p[beta-1]) * np.linalg.norm(q[alpha-1] - q[beta-1])**2)/(p[alpha-1] - p[beta-1])**2 \
                        - (p[alpha-1] * (z[alpha-1]**2) - p[beta-1] * (z[beta-1]**2))/(p[alpha-1] - p[beta-1])
            
            return constraint
        
        for (alpha, beta) in self.cell_pairs:
            constraints.append({'type': 'ineq', 'fun': imaginary_constraint(alpha, beta)})
        
        return bounds, constraints
    
    def fit(self):
        
        """
        
        Perform the minimization of equation 5 with respect to the variables (q, z, p)
        
        """
        
        # initialize the vector
        theta0 = self.initialize()
        #theta0 = self.initialize_points()
                
        # get the bounds and constraints
        bounds, constraints = self.load_constraints(theta0)
        
        print("Main minimization")
        # minimize
        print(theta0)
        optimal = minimize(self.energy, theta0, options={'disp':True}, bounds=bounds, constraints=constraints).x
        
        # extract the values from the optimal vector
        q, z, p = self.extract_values(optimal)
        
        # compute the tensions
        self.get_tensions(q, z, p)
        
        return q, z, p
    
    def get_tensions(self, q, z, p):
        
        """
        
        applies the Young-Laplace law to obtain the tensions at every edge
        
        """
        
        center, radius = self.transform(q, z, p) 
        
        for (alpha, beta) in self.cell_pairs:
            if alpha not in self.edge_cells or beta not in self.edge_cells: 
                self.tension[alpha][beta] = np.abs((p[alpha-1] - p[beta-1]) * radius[alpha][beta])
                self.tension[beta][alpha] = self.tension[alpha][beta]
        
    def get_normalized_tensions(self):
        """
        
        normalize the tensions between 0 and 1. Used for plotting in the CAP tiling
        
        """
        tensions_normalized = {alpha: {beta: None for beta in self.cells} for alpha in self.cells}
        min_T = np.inf ; max_T = -np.inf
        
        for (alpha, beta) in self.cell_pairs:
            if alpha not in self.edge_cells or beta not in self.edge_cells:
                if self.tension[alpha][beta] > max_T: max_T = self.tension[alpha][beta]
                elif self.tension[alpha][beta] < min_T: min_T = self.tension[alpha][beta]
        for (alpha, beta) in self.cell_pairs:
            if alpha not in self.edge_cells or beta not in self.edge_cells:
                tensions_normalized[alpha][beta] = (self.tension[alpha][beta] - min_T) / (max_T - min_T)
                tensions_normalized[beta][alpha] = -1 * tensions_normalized[alpha][beta]
        return tensions_normalized
            
    def CAP(self, img, q, z, p, linewidth=2, endpoint_size=5):
        
        """
        
        Takes the generating points determined by the minimization, 
        finds the corresponding (center, radius) pairs for each edge, 
        and plots the resuling cirles between the first and last elements in each edge
        
        """
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        center, radius = self.transform(q, z, p) 
        
        # normalize the tensions (to see the differences when plotting the colors)
        tensions_normalized = self.get_normalized_tensions()

        for (alpha, beta) in self.cell_pairs: 
            if alpha not in self.edge_cells or beta not in self.edge_cells:
                # get the tension in the edge
                T = tensions_normalized[alpha][beta]

                # get the corresponding CAP center and radius
                rho, R = center[alpha][beta], radius[alpha][beta]

                # define the set of points in the circle using polar coordinates
                circle = [rho + R*np.array([np.cos(theta), np.sin(theta)]) for theta in np.linspace(0, 2*np.pi, num=70000)]

                # find the edge endpoints
                start, end = self.vertices[alpha][beta]

                # find the points in the set that are closest to the edge extremities
                cap_s = np.argmin(np.linalg.norm(circle - start, axis=1))
                cap_e = np.argmin(np.linalg.norm(circle - end, axis=1))
                arcs = [circle[cap_e:] + circle[:cap_s+1], circle[cap_s:cap_e+1], circle[cap_s:] + circle[:cap_e+1], circle[cap_e:cap_s+1]]
                lengths = [len(arc) if len(arc) != 0 else np.inf for arc in arcs]
                arc = arcs[np.argmin(np.array(lengths))]

                try:
                    # plot the continuous path between those two points
                    for point in arc:
                        # color will be determined by the tension
                        img = cv2.circle(img, (int(point[1]), int(point[0])), linewidth, (1, (1-T), (1-T)), -1)

                    # plot the arc endpoints
                    img = cv2.circle(img, (int(arc[0][1]), int(arc[0][0])), endpoint_size, (0, 0, 0), -1)
                    img = cv2.circle(img, (int(arc[-1][1]), int(arc[-1][0])), endpoint_size, (0, 0, 0), -1)
                except:
                    print(T)

                # plot the generating points
                #img = cv2.circle(img, (int(q[alpha-1][1]), int(q[alpha-1][0])), 3, (1, 1, 1), -1)
                #img = cv2.circle(img, (int(q[beta-1][1]), int(q[beta-1][0])), 3, (1, 1, 1), -1)
        
        return img
        
def get_actual(model, seg, dtr, generating_points):
    actual_model = VMSI(cell_pairs = seg.pairs(), edges = seg.edges(), num_cells = len(seg.cells[0]), 
             cells = seg.cells[0], edge_cells = seg.get_edge_cells(), barycenters = seg.barycenters[0], height=256, width=256)
    q, z, p = actual_model.extract_values(model.initialize_points())
    actual_model.get_tensions(q, z, p)
    
    # get the actual q, z, and p by finding the closest points that were used to generate the image
    q_actual = []
    z_actual = []
    p_actual = []

    generating_q = np.array([point.q[::-1] for point in generating_points.points])
    generating_z = np.array([point.z for point in generating_points.points])
    generating_p = np.array([point.p for point in generating_points.points])

    for i in range(len(q)):
        closest_index = np.argmin([np.linalg.norm(x - q[i]) for x in generating_q])
        q_actual.append(generating_q[closest_index])
        z_actual.append(generating_z[closest_index])
        p_actual.append(generating_p[closest_index])

    q_actual = np.array(q_actual)
    z_actual = np.array(z_actual)
    p_actual = np.array(p_actual)
    
    img = actual_model.CAP(dtr.transform.copy(), q_actual, z_actual, p_actual)
   
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.show()
    
    return actual_model
    
def evaluate(model, seg, dtr, generating_points):
    actual_model = get_actual(model, seg, dtr, generating_points)
    
    predicted = model.tension
    actual = actual_model.tension

    x_points = []
    y_points = []
    for (alpha, beta) in model.cell_pairs:
        if alpha not in model.edge_cells or beta not in model.edge_cells:
            try:
                pred = abs(predicted[alpha][beta])
                actual = abs(actual[alpha][beta])
                
                x_points.append(pred)
                y_points.append(actual)
            except:
                pass

    print(scipy.stats.pearsonr(x_points, y_points))
    plt.figure(figsize=(5,5))
    plt.scatter(x_points, y_points)
    return x_points, y_points
