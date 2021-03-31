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

class Segmenter:
    def __init__(self):
        self.model = models.Cellpose(model_type='cyto')
        self.images = []
        self.N = None
        self.masks = []
        self.outlines = []
        self.vertices = []
        self.cells = []
        self.borders = []
        self.adjacent_cells = []
        self.barrycenters = []
    
    def closest_nonzero(self, img, pt):
        """
        returns the closest non-zero value to some point in the image. 
        Does so by iteratively constructing a spiral around the point.
        """
        steps_before_rotating = 0
        sign = -1
        current_point = pt

        while True:
            for j in range(2):
                for k in range(steps_before_rotating):
                    if j == 0:
                        # step in the y direction
                        current_point = current_point + sign * np.array([0, 1])
                    elif j == 1:
                        # step in the x direction
                        current_point = current_point + sign * np.array([1, 0])
                    try:
                        if img[current_point[0], current_point[1]] != 0 and current_point[0] < img.shape[0] and current_point[1] < img.shape[1] and current_point[0] >= 0 and current_point[1] >= 0:
                            return img[current_point[0], current_point[1]]
                    except:
                        # case where the spiral goes out of bounds (when zeroed points are near the edges of the image)
                        pass

            sign *= -1
            steps_before_rotating += 1
    
    
    def finetune_masks(self):
        """
        fine tune the masks by filling in points where they are zero
        """
        
        def swap_cell_ids(img, i, j):
            """
            auxiliary function -- takes two cell ids in the mask and swaps them in the mask
            """
            print("swapping", i, "and", j)
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    if img[x][y] == i: 
                        img[x][y] = j
                    elif img[x][y] == j:
                        img[x][y] = i
            return img
        
        for i in range(self.N):            
            # find where the mask image is zero
            zeros = np.array(np.where(self.masks[i] == 0)).T
            
            new_img = self.masks[i].copy()
            for point in zeros:
                new_img[point[0], point[1]] = self.closest_nonzero(self.masks[i], point)
            
            # fix the mask ordering --> order according to the distance of the cells' barrycenters from the origin
            self.masks[i] = new_img.copy() 
                        
            cells = np.unique(new_img)
            center_norms = []
            for cell in cells:
                center = np.mean(np.array(np.where(new_img == cell)).T, axis=0)
                norm = np.linalg.norm(center)
                center_norms.append(norm)
 
            sorted_norm_args = np.array(center_norms).argsort()
            for j in range(len(sorted_norm_args)):
                self.masks[i][np.where(new_img == sorted_norm_args[j] + 1)] = j + 1              
    
    def compute_barrycenters(self, i=0):
        # initialize
        if len(self.barrycenters) == 0:
            self.barrycenters = [None for i in range(self.N)]
        if self.barrycenters[i] == None:
            self.barrycenters[i] = {alpha:None for alpha in self.cells[i]}
            
        # find the barrycenter for each cell
        for alpha in self.cells[i]:
            self.barrycenters[i][alpha] = np.mean(np.array(np.where(self.masks[i] == alpha)).T, axis=0)
    
    def compute_vertices(self):
        """
        find the vertices according to the masks (i.e. intersections of three colors). The self.vertices[i]
        contains the indices of the vertices in the mask corresponding to image i. 
        In addition, it computes the outline
        """
        
        # define 3x3 kernel that finds the number of different colors in the neighborhood of a point
        kernel = lambda neighborhood : len(set(neighborhood))
        
        for i in range(self.N):
            # convolve that kernel with the masks
            res = generic_filter(self.masks[i], kernel, (3, 3))
            
            self.outlines.append(res)

            # vertices are the the points where the value is 3
            indices = np.array(np.where(res >= 3)).T
            self.vertices.append(indices)
    
    def find_cells(self):
        for i in range(self.N):
            # colors in the mask are unique --> each corresponds to a cell
            self.cells.append(np.unique(self.masks[i]))
      
    def get_border(self, alpha, beta, view_border = False, i = 0):
        """
        return the points in the border between cells alpha and beta
        """
        border = self.masks[i].copy()
                
        # zero out everything except the two cells
        border[np.where((border != alpha) & (border != beta))] = 0
   
        # to find the border, count the values in each point's neighborhood (removing 1 if the value is 0)
        kernel = lambda x : len(set(x.flatten())) - list(x.flatten()).count(0)
        
        # only do the convolution over the area taken up by the two masks for performance reasons
        # so we create some bounding box over which we'll convolve
        non_zero = np.transpose(np.where(border != 0))
        max_x = np.max(non_zero.T[0]) ; max_y = np.max(non_zero.T[1])
        min_x = np.min(non_zero.T[0]) ; min_y = np.min(non_zero.T[1])
        border = border[min_x:max_x, min_y:max_y]
        
        border = generic_filter(border, kernel, (3, 3))
                
        # identify elements in the border (i.e. those that are 2)
        border_points = np.transpose(np.where(border == 2))
        for k in range(len(border_points)):
            border_points[k][0] += min_x
            border_points[k][1] += min_y
        
        # if not initialized, initialize the borders to be None
        if len(self.borders) == 0:
            for j in range(self.N):
                self.borders.append({c1 : {c2 : None for c2 in self.cells[j]} for c1 in self.cells[j]})
        
        if len(border_points) != 0:
            self.borders[i][alpha][beta] = border_points
            self.borders[i][beta][alpha] = border_points
    
        if view_border:
            img = self.masks[i].copy()
            for point in self.borders[i][alpha][beta]:
                img = cv2.circle(img, (point[1], point[0]), 1, (255, 255, 255), -1)

            plt.figure(figsize=(15, 10))
            plt.imshow(img)
            plt.show()

    
    def edge(self, alpha, beta, i=0):
        """
        get the stored edge between cells alpha and beta in an image
        """
        try:
            return self.borders[i][alpha][beta]
        except:
            print("edges does not exist")
    
    def edges(self, i=0):
        """
        get all of the edges for a given image
        """
        return self.borders[i]
    
    def pairs(self, i=0):
        """
        returns the stored adjacent cells for the given image
        """
        pairs_l = []
        for (a,b) in self.adjacent_cells[i]:
            if self.borders[i][a][b] is None:
                print("empty border for", a,b)
            if self.borders[i][a][b] is not None:
                pairs_l.append((a,b))
        return pairs_l
    
    def get_edge_cells(self, i=0):
        """
        returns the cells on the edge of the image
        """
        cells = set()
        mask = self.masks[i]
        for i in range(mask.shape[0]):
            cells.add(mask[i][0])
            cells.add(mask[i][mask.shape[1]-1])
        for j in range(mask.shape[1]):
            cells.add(mask[0][j])
            cells.add(mask[mask.shape[0]-1][j])
        return cells
    
    def neighbors(self, v, i=0):
        """
        
        returns the mask values in the neighborhood of vertex v, where v is an index
        
        """
        neighs = {(v[0], v[1] + 1), (v[0] + 1, v[1] + 1), (v[0] + 1, v[1]), (v[0] + 1, v[1] - 1),
                (v[0], v[1] - 1), (v[0] - 1, v[1] - 1), (v[0] - 1, v[1]), (v[0] - 1, v[1] + 1)}

        if v[0] == 0:
            neighs.discard((v[0] - 1, v[1]))
            neighs.discard((v[0] - 1, v[1] - 1))
            neighs.discard((v[0] - 1, v[1] + 1))

        if v[1] == 0:
            neighs.discard((v[0], v[1] - 1))
            neighs.discard((v[0] - 1, v[1] - 1))
            neighs.discard((v[0] + 1, v[1] - 1))

        if v[0] == self.masks[i].shape[0]-1:
            neighs.discard((v[0] + 1, v[1]))
            neighs.discard((v[0] + 1, v[1] - 1))
            neighs.discard((v[0] + 1, v[1] + 1))

        if v[1] == self.masks[i].shape[1]-1:
            neighs.discard((v[0], v[1] + 1))
            neighs.discard((v[0] - 1, v[1] + 1))
            neighs.discard((v[0] + 1, v[1] + 1))

        return [self.masks[i][x] for x in neighs]
    
    def get_adjacent_cells(self, i=0):
        if len(self.adjacent_cells) == 0:
            self.adjacent_cells = [[] for _ in range(self.N)]
        
        # get all pairs of adjacent colors
        all_pairs = []
        for v in self.vertices[i]:
            neighboring_colors = self.neighbors(v)
            # get unique colors in the neighborhood
            unique = set(neighboring_colors)
            # pairs of colors in the neighborhood of v
            pairs = list(itertools.combinations(unique, 2))
            all_pairs += pairs

        for p in set(all_pairs):
            if (p[0], p[1]) not in self.adjacent_cells[i] and (p[1], p[0]) not in self.adjacent_cells[i]:
                self.adjacent_cells[i].append((p[0], p[1]))
    
    def find_edges(self, i=0):
        self.get_adjacent_cells()
        for (alpha, beta) in tqdm(self.adjacent_cells[i]):
            self.get_border(alpha, beta, i=i) 
            
    def segment(self, images, diameter=None):
        """
        
        Main function -- segments the image into cells and identifies the edges 
        
        """
        
        if type(images) != list: images = [images]
        self.images = images
        self.N = len(images)
        
        print("Evaluating the neural network")
        masks, flows, styles, diams = self.model.eval(images, diameter=diameter, flow_threshold=None, channels=[0,0])
    
        # original masks
        self.masks = masks
                
        print("Fixing the masks")
        # finetune the mask
        self.finetune_masks()
                
        print("Computing the vertices")
        # compute the vertices and outline
        self.compute_vertices()
        
        print("Identifying the cells")
        # compute the cells 
        self.find_cells()
        
        print("Finding the borders between cells")
        # find all edges between cells
        self.find_edges()
        
        print("Finding the cell barycenters")
        # find alll barycenters
        self.compute_barrycenters()
    
    def visualize(self, name='outlines', specific_cell = None, show_vertices = True, i = 0, overlay=False, return_img=False):
        """
        visualize the masks on the ith image
        """
        
        if name == 'masks':
            segmented = self.masks[i].copy()
        else:
            segmented = self.outlines[i].copy()
        
        if specific_cell != None:
            segmented[segmented != specific_cell] = 0
        
        if overlay:
            image = cv2.cvtColor(self.images[i], cv2.COLOR_GRAY2RGB)
            #img = plot.mask_overlay(image, segmented)
            for (alpha, beta) in self.pairs(i):
                for point in self.borders[i][alpha][beta]:
                    img = cv2.circle(image, (point[1], point[0]), 1, (1, 0, 0), -1)
        else:
            img = segmented.copy()
        
        if show_vertices:
            for point in self.vertices[i]:
                img = cv2.circle(img, (point[1], point[0]), 2, (0, 0, 1), -1)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.show()
        
        if return_img:
            return img