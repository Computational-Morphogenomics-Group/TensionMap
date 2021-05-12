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

class Point:
    def __init__(self, x, y, z, p):
        self.x = x
        self.y = y
        self.z = z
        
        self.q = np.array([x,y])
        self.p = p
    
    def distance(self, r):
        return np.sqrt(self.p *(np.linalg.norm(r - self.q, axis=-1)**2 + np.linalg.norm(self.z)**2))
    
    def numpy(self):
        return np.array([self.x, self.y, self.z])
    
class Points:
    def __init__(self, num_points = 121, max_x = 1000, max_y = 800, max_z = 0, min_p = 0.001, max_p = 0.005):
        self.num_points = num_points
        self.N = np.sqrt(num_points).astype(int)
        
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z
        self.min_p = min_p
        self.max_p = max_p
        
        self.grid = self.initialize_grid()
        self.points = self.get_random_points(num_points)
    
    def initialize_grid(self):
        grid = []
        for i in range(self.max_x):
            for j in range(self.max_y):
                grid.append([i,j])
        return np.array(grid)

    def sort(self, CMs):
        """
        sort the points according to the norm of the center of their spanned cells.
        Uses the precomputed distance transform
        IMPORTANT: the center != generating point location
        """
        norms = np.array([np.linalg.norm(x) for x in CMs])

        sorted_norm_args = norms.argsort()
        old_points = self.points.copy()
        for j in range(len(sorted_norm_args)):
            self.points[j] = old_points[sorted_norm_args[j]] 
        
    def get_random_points(self, num_points):
        x_points = np.linspace(20, self.max_x-20, self.N).astype(int)
        y_points = np.linspace(20, self.max_y-20, self.N).astype(int)
                
        points = []
        for i in range(self.N):
            for j in range(self.N):
                x = int(x_points[i] + np.random.normal(0, 10))
                y = int(y_points[j] + np.random.normal(0, 10))
                z = int(np.random.uniform(0, self.max_z))
                p = np.random.uniform(self.min_p, self.max_p)
                
                x = np.clip(x, 0, self.max_x-1)
                y = np.clip(y, 0, self.max_y-1)
                
                points.append(Point(x,y,z,p))
        return points
    
    def image(self, show=False):
        img = np.ones((self.max_y, self.max_x)).astype('float32')
        for point in self.points:
            img = cv2.circle(img, (point.x, point.y), self.N, (0, 0, 0), -1)
        if show:
            plt.figure(figsize=(15, 10))
            plt.imshow(img, cmap='gray')
            plt.show()
        return img
    
    def numpy(self):
        img = np.zeros((self.max_x, self.max_y, self.max_z))
        for point in self.points:
            img[point.x, point.y, point.z] = 1
        return img
        
class DistanceTransform():
    def __init__(self, points):
        self.grid = points.grid
        self.points = points.points
        self.max_x = points.max_x
        self.max_y = points.max_y
        
        self.distances = [] # self.distances[i] contains the distance matrix for points[i] on the grid
        
        for i, point in enumerate(self.points):
            self.distances.append(point.distance(self.grid))
        
        self.distances = np.array(self.distances)
        self.transform = None
        self.CMs = np.array([[0, 0] for _ in range(len(self.points))])
        self.points_in_cell = [0 for _ in range(len(self.points))]
        
        self.pressure_mask = None
        
    def normalize(self, x):
        max_ = np.max(x)
        min_ = np.min(x)
        return (x - min_)/(max_ - min_)
    
    def compute_transform(self):
        """
        return image with the intensity of r = (x,y) corresponding to min_{i} p_i d_i^2(r) where i corresponds 
        to the index of the point in points. Also computes the center of mass for each cell. 
        """
        
        transform = np.zeros((self.max_x, self.max_y)).astype('float32')
        pressures = np.zeros((self.max_x, self.max_y)).astype('float32')
        
        for i, r in enumerate(self.grid):
            closest_cell = np.argmin(self.distances[:, i])
            transform[r[0], r[1]] = self.distances[:, i][closest_cell]
            
            pressures[r[0], r[1]] = self.points[closest_cell].p
            
            # update center of mass of corresponding cell
            self.CMs[closest_cell] += r
            self.points_in_cell[closest_cell] += 1
        
        self.CMs = [self.CMs[i]/self.points_in_cell[i] for i in range(len(self.points))]
        self.transform = self.normalize(transform).T
        self.pressure_mask = self.normalize(pressures).T
    
    def visualize_transform(self):
        res_rgb = cv2.cvtColor(self.transform, cv2.COLOR_GRAY2RGB)
        for point in self.points:
            res_rgb = cv2.circle(res_rgb, (point.x, point.y), 3, (0, 0, 1), -1)

        plt.figure(figsize=(15, 10))
        plt.imshow(res_rgb)
        plt.show()
