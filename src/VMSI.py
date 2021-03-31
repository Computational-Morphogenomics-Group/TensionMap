import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from cellpose import models
from cellpose import utils
from cellpose import plot
from scipy.ndimage import generic_filter
import itertools
from tqdm import tqdm
from scipy.optimize import minimize, leastsq

class VMSI():
    
    def __init__(self, cell_pairs, edges, num_cells, cells, barrycenters, edge_cells, width=500, height=500):
        self.cell_pairs = cell_pairs
        self.edges = edges
        self.num_edges = len(edges)
        self.num_cells = num_cells
        self.width = width
        self.height = height
        self.barrycenters = barrycenters
        self.cells = cells
        self.tension = {alpha: {beta: None for beta in self.cells} for alpha in self.cells}
        self.edge_cells = edge_cells
        
        # remove edges if they don't have at least 3 points
        for (alpha, beta) in self.cell_pairs:
            if len(self.edges[alpha][beta]) < 3: 
                self.cell_pairs.remove((alpha, beta))
                self.edges[alpha][beta] = None
        
        # init vertices
        self.vertices = {alpha: {beta: None for beta in self.cells} for alpha in self.cells}
        for (alpha, beta) in self.cell_pairs:
            self.vertices[alpha][beta] = self.get_vertices(alpha, beta)
        
        # init tangents at vertices
        self.tangents = {alpha: {beta: None for beta in self.cells} for alpha in self.cells}
        for (alpha, beta) in self.cell_pairs:
            self.tangents[alpha][beta] = self.get_tangents(alpha, beta)
    
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
            center = self.barrycenters[alpha]
    
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
        
        
        # add constraints
        def ratio_constraint(theta):
            """
            The solution is constrained so that the ratio of the average
            magnitude of t to the average pressure differential
            equals the averaged measured radius of curvature
            in the image. 
            
            The average measured radius of curvature is found via least squares
            """
            
            # find the ratio
            q, p = extract(theta)
            avg_t_norm = 0
            avg_p_differential = 0
            for (alpha, beta) in self.cell_pairs:
                avg_p_differential += np.abs(p[alpha-1] - p[beta-1])
                for i in range(2):
                    avg_t_norm += np.linalg.norm(t(alpha, beta, q, p, i))
            
            ratio = avg_t_norm / 2*avg_p_differential
            # R_avg is precomputed outside of this function 
            return ratio - R_avg
        
        # Try to make the tangent and the ti orthogonal by minimizing this function
        def E_initial(theta):            
            q, p = extract(theta)
            E = 0
            
            E1 = abs(ratio_constraint(theta))
            E += 1000000*E1
            
            if len([True for i in range(len(p)) if p[i] < 0]):
                return np.inf
            
            for (alpha, beta) in self.cell_pairs:
                # for both vertices at the extremities
                for i in range(2):
                    E += (t(alpha, beta, q, p, i) @ self.tangents[alpha][beta][i])**2
            return E

        def fit_circle(edge):
            """
            fitting the circle involves finding a point st the distance from that point to every
            point in the egde is the same. Error to minimize is thus the variance of the distances from the center
            """ 
            def error(x_c, y_c):
                center = np.array([x_c, y_c])
                return np.std([np.linalg.norm(x - center) for x in edge])**2
            
            # optimize
            center0 = np.mean(edge, axis=0)
            x_c0, y_c0 = center0[0], center0[1] 
            center = leastsq(error, x_c0, y_c0)[0]
            R = np.mean([np.linalg.norm(x - center) for x in edge])
            return R
        
        R_avg = np.mean([fit_circle(self.edges[alpha][beta]) for (alpha, beta) in self.cell_pairs])
        
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
            
    def CAP(self, img, q, z, p):
        
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
                        img = cv2.circle(img, (int(point[1]), int(point[0])), 2, (1, (1-T), (1-T)), -1)

                    # plot the arc endpoints
                    img = cv2.circle(img, (int(arc[0][1]), int(arc[0][0])), 5, (0, 0, 0), -1)
                    img = cv2.circle(img, (int(arc[-1][1]), int(arc[-1][0])), 5, (0, 0, 0), -1)
                except:
                    print(T)

                # plot the generating points
                #img = cv2.circle(img, (int(q[alpha-1][1]), int(q[alpha-1][0])), 3, (1, 1, 1), -1)
                #img = cv2.circle(img, (int(q[beta-1][1]), int(q[beta-1][0])), 3, (1, 1, 1), -1)
        
        return img
        
def get_actual(seg, dtr):
    actual_model = VMSI(cell_pairs = seg.pairs(), edges = seg.edges(), num_cells = len(seg.cells[0]), 
             cells = seg.cells[0], edge_cells = seg.get_edge_cells(), barrycenters = seg.barrycenters[0], height=256, width=256)
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
    
def evaluate(model, seg=seg, dtr=dtr):
    actual_model = get_actual(seg, dtr)
    
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