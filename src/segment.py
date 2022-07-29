import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import generic_filter
from scipy.optimize import minimize, leastsq
from scipy.spatial import ConvexHull
import skimage.segmentation as seg
import skimage.morphology as morph
import skimage.measure as measure
import skimage.draw as draw
from src.bwmorph import *
import pandas as pd
from scipy.spatial.distance import cdist

class VMSI_obj:
    def __init__(self):
        self.V_df = []
        self.C_df = []
        self.E_df = []

class Segmenter:
    def __init__(self, images = None, masks = None, very_far = 150, labelled=False):
        """
        :param masks: Numpy array - Segmented image with edges set to zero and cells set to non-zero. Edges must be 1px wide and 4-connected.
        :param very_far: Int - Maximum distance between two vertices connected by the same edge.
        :param labelled: Bool - Whether the segmented cells have been labelled.
        """
        self.images = []
        self.masks = []
        self.very_far = very_far

        if images is not None:
            self.images = images
        if masks is not None:
            self.masks = masks
        if not labelled:
            self.masks = measure.label(self.masks)

    def process_segmented_image(self, holes_mask=None):
        """
        Given a segmented mask, produce VMSI_obj for input into VMSI
        """

        # Process mask
        # Clear border (create external cell from all cells that run into image boundary)
        tmp1 = seg.clear_border(self.masks)
        # If we are specifying holes, also set cells bordering holes as external cell
        if holes_mask is not None:
            tmp5 = morph.binary_dilation(holes_mask, footprint=np.ones([5,5]))
            hole_adj_cells = np.unique(tmp5 * self.masks)[1:]
            tmp6 = np.isin(self.masks, hole_adj_cells)
            tmp1[tmp6] = 0
        tmp2 = ((self.masks - tmp1)>0).astype(int)
        tmp3 = tmp1 + tmp2

        # Find edge pixels that only separate external cells
        kernel = lambda neighborhood : len(set(neighborhood))
        tmp4 = generic_filter(tmp3, kernel, footprint=np.ones([3,3]))
        tmp1[np.logical_and(tmp1==0,tmp4<3)] = 1

        mask_tmp = tmp1

        # Relabel mask (may not be necessary in future, just for Matlab compatibility)
        mask_tmp = self.relabel(mask_tmp)

        # Create VMSI object to store vertex, cell and edge information
        obj = VMSI_obj()

        obj.C_df = self.find_cells(mask_tmp)
        obj.V_df, cc = self.find_vertices(mask_tmp, obj.C_df)
        obj.E_df = self.find_edges(obj, mask_tmp, cc)
        self.identify_holes(obj)
        return obj, mask_tmp

    def find_vertices(self, mask, C_df):
        V_df = pd.DataFrame(columns = ['coords','ncells','nverts','edges'])

        branchpoints = self.find_branch_points(mask==0)

        cc = measure.label(branchpoints, connectivity=2)
        v = np.array([np.flip(np.round(regionprops.centroid).astype(int)) for regionprops in measure.regionprops(cc)])
        a = np.array([regionprops.coords for regionprops in measure.regionprops(cc)])
        # regionprops returns the coordinates in numpy indexing rather than cartesian indexing - e.g.
        # (rows, cols) rather than (x, y) so flip and re-sort coordinates
        a = a[v[:,0].argsort()]
        v = v[v[:,0].argsort()]

        R = np.zeros([2,v.shape[0]])
        for i in range(v.shape[0]):

            vertex = v[i,:]
            # Flip again to convert back to numpy indexing
            ncells = mask[min(a[i][:,0])-1:max(a[i][:,0])+2, min(a[i][:,1])-1:max(a[i][:,1])+2]
            ncells = np.unique(ncells[ncells!=0])-1

            vertex_df = pd.DataFrame({'coords':[vertex],'ncells':[ncells],'nverts':[np.array([])],'edges':[np.array([])]})
            V_df = pd.concat([V_df, vertex_df], ignore_index=True)

            R[0, i] = vertex[0]
            R[1, i] = vertex[1]
        # Identify neighbour vertices
        adj = np.zeros([v.shape[0],v.shape[0]])

        D = np.add(np.tile(np.sum(np.multiply(R, R), axis=0), (v.shape[0],1)),
                   np.tile(np.sum(np.multiply(R, R), axis=0), (v.shape[0],1)).T) - 2*np.matmul(R.T, R)

        for V in range(len(V_df)):
            for cell in V_df.at[V, 'ncells']:
                C_df.at[cell, 'numv'] += 1
                C_df.at[cell, 'nverts'] = np.append(C_df.at[cell, 'nverts'], np.array([V]))

        for C in range(len(C_df)):
            # If cell has no vertices, assume it must border the external cell only
            if C_df.at[C, 'nverts'].size > 0:
                ncells = np.setdiff1d(np.unique(np.hstack(V_df.loc[C_df.at[C, 'nverts'], 'ncells'].tolist())), C)
            else:
                ncells = np.array([0])
            C_df.at[C, 'ncells'] = ncells

        for i in range(v.shape[0]):
            for j in range(i+1,v.shape[0]):
                if D[i,j] <= np.power(self.very_far, 2):
                    if np.intersect1d(V_df['ncells'].iloc[i], V_df['ncells'].iloc[j]).size >=2:
                        adj[i,j] = 1
                        adj[j,i] = 1
            V_df['nverts'].iloc[i] = np.where(adj[i,:]==1)[0]
        return V_df, cc

    def find_branch_points(self, skel):
        # Vectorized branch point finding; faster than convolving with filter

        skel = np.array(skel, dtype=int)

        branch_points = np.zeros(skel.shape)
        branch_points[1:skel.shape[0]-1,1:skel.shape[1]-1] = skel[2:skel.shape[0],1:skel.shape[1]-1] + skel[0:skel.shape[0]-2,1:skel.shape[1]-1] + \
                                   skel[1:skel.shape[0]-1,2:skel.shape[1]] + skel[1:skel.shape[0]-1,0:skel.shape[1]-2]
        branch_points = np.multiply(branch_points,skel)
        branch_points = branch_points >= 3
        return branch_points

    def find_cells(self, mask):
        # Identify cells, record region information
        C_df = pd.DataFrame(columns = ['centroids','nverts','numv','ncells','edges', 'area', 'holes',\
                                       'inertia', 'perimeter','feret_d', \
                                       'moments_hu','bbox','orientation'])

        # regionprops returns the co-ordinates in numpy indexing rather than cartesian indexing - e.g.
        # (rows, cols) rather than (x, y) so flip
        c = np.array([np.flip(regionprops.centroid) for regionprops in measure.regionprops(mask)])
        p = np.array([regionprops.perimeter for regionprops in measure.regionprops(mask)])
        ine = np.array([regionprops.inertia_tensor[np.triu_indices(2)] for regionprops in measure.regionprops(mask)])
        bbox = np.array([[regionprops.bbox[3]-regionprops.bbox[1],regionprops.bbox[2]-regionprops.bbox[0]] for regionprops in measure.regionprops(mask)])
        moments_hu = np.array([regionprops.moments_hu for regionprops in measure.regionprops(mask)])
        cell_props = pd.DataFrame(measure.regionprops_table(mask, properties=('label', \
                                                                              'feret_diameter_max','orientation','area')))

        # estimate very_far to be the half the maximum cell perimeter
        self.very_far = np.max(p[1:])/2

        for i in range(c.shape[0]):
            cell_df = pd.DataFrame({'centroids':[c[i,:]],'nverts':[np.array([])],'numv':0,'ncells':[np.array([])], 'edges':[np.array([])], \
                                    'area':cell_props.at[i,'area'], 'holes':False, 'inertia':[ine[i]], \
                                    'perimeter':p[i], \
                                    'feret_d':cell_props.at[i,'feret_diameter_max'], \
                                    'moments_hu':[moments_hu[i,:]],'bbox':[bbox[i,:]],'orientation':cell_props.at[i,'orientation']})
            C_df = pd.concat([C_df, cell_df], ignore_index=True)
        return C_df

    def identify_holes(self, obj):
        """
        Filter out labelled objects that have area greater than 2x the median area and are non-convex
        """
        areas = obj.C_df['area'].to_numpy()
        for i in range(obj.C_df.shape[0]):
            vcoords = np.array(obj.V_df.loc[obj.C_df.at[i, 'nverts'], 'coords'].tolist())
            if vcoords.shape[0] >= 3:
                hull = ConvexHull(vcoords)
                if hull.simplices.shape[0] < vcoords.shape[0] and obj.C_df.at[i, 'area'] > 2*np.median(areas):
                    obj.C_df.at[i, 'holes'] = True
            else:
                obj.C_df.at[i, 'holes'] = True
        return

    def relabel(self, mask):
        """
        If cells aren't sequenctially label, relabel them
        """
        new_mask = np.zeros(mask.shape, dtype=int)
        ids = np.sort(np.unique(mask))

        for i in range(1,len(ids)):
            new_mask[mask==ids[i]] = i
        return new_mask

    def find_edges(self, obj, mask, cc):
        E_df = pd.DataFrame(columns = ['pixels','verts','cells'])

        l_dat = mask
        b_dat = (l_dat == 0).astype(int)

        rv = np.vstack(obj.V_df['coords'])
        verts = np.zeros(b_dat.shape)
        verts[rv[:,1],rv[:,0]] = 1

        b_dat[np.where(verts == 1)] = 0
#        b_end  = b_dat * morph.dilation(verts, morph.disk(1))

        b_dat[cc != 0] = 0
#        b_end = (b_end * b_dat) + (self.endpoints(b_dat) * b_dat)
        # Not sure what the Matlab code is trying to accomplish but it doesn't seem to work so try another method
        b_end = self.endpoints(b_dat) * b_dat

        re = np.argwhere(b_end.T != 0)
        D = cdist(re, rv)

        b_l = measure.label(b_dat.T, connectivity=1).T
        end_labels = b_l[re[:,1],re[:,0]]
        b_props = measure.regionprops(b_l)

        for i in range(1, len(np.unique(b_l))):
            end_points = np.argwhere(end_labels==i)

            v1 = -1
            v2 = -1

            # Edges with 1 endpoint are generally 1-length; ignore these
            if len(end_points) == 2:
                v1 = np.argmin(D[end_points[0],:])
                v2 = np.argmin(D[end_points[1],:])
                if (v1 == v2):
                    sort1 = np.sort(D[end_points[0],:]).squeeze()
                    sort2 = np.sort(D[end_points[1],:]).squeeze()
                    if abs(sort1[0] - sort1[1]) <= np.sqrt(3):
                        v1 = np.argsort(D[end_points[0],:]).squeeze()[0]
                    elif abs(sort2[0] - sort2[1]) <= np.sqrt(3):
                        v2 = np.argsort(D[end_points[1],:]).squeeze()[0]

            if (v1 != -1) and (v2 != -1) and (v2 in obj.V_df.at[v1, 'nverts']) and ((v1 not in obj.C_df.at[0, 'nverts']) or (v2 not in obj.C_df.at[0, 'nverts'])):
                pix = np.ravel_multi_index(np.flip(b_props[i-1].coords.T), mask.shape[::-1])
                verts = np.array([v1, v2])
                cells = np.intersect1d(obj.V_df.at[v1, 'ncells'], obj.V_df.at[v2, 'ncells'])
                edge_df = pd.DataFrame({'pixels':[pix],'verts':[verts],'cells':[cells]})
                E_df = pd.concat([E_df, edge_df], ignore_index=True)

        # Edit V_df and C_df with edge information
        for v in range(0, len(obj.V_df)):
            for nv in obj.V_df.at[v, 'nverts']:
                edge_1 = np.argwhere((np.vstack(E_df['verts'])[:,0] == v)*(np.vstack(E_df['verts'])[:,1] == nv))
                edge_2 = np.argwhere((np.vstack(E_df['verts'])[:,1] == v)*(np.vstack(E_df['verts'])[:,0] == nv))

                if edge_1.size > 0:
                    obj.V_df.at[v, 'edges'] = np.append(obj.V_df.at[v, 'edges'], edge_1)
                elif edge_2.size > 0:
                    obj.V_df.at[v, 'edges'] = np.append(obj.V_df.at[v, 'edges'], edge_2)
                elif (v not in obj.C_df.at[0, 'nverts']) and (nv not in obj.C_df.at[0, 'nverts']):
                    # Create new edge
                    line = draw.line(obj.V_df.at[v, 'coords'][1], obj.V_df.at[v, 'coords'][0], obj.V_df.at[nv, 'coords'][1], obj.V_df.at[nv, 'coords'][0])
                    pix = np.ravel_multi_index(np.flip(line,axis=0), mask.shape[::-1])
                    verts = np.array([v, nv])
                    cells = np.intersect1d(obj.V_df.at[v, 'ncells'], obj.V_df.at[nv, 'ncells'])
                    edge_df = pd.DataFrame({'pixels':[pix],'verts':[verts],'cells':[cells]})
                    E_df = pd.concat([E_df, edge_df], ignore_index=True)
                    obj.V_df.at[v, 'edges'] = np.append(obj.V_df.at[v, 'edges'], len(E_df))
                else:
                    obj.V_df.at[v, 'edges'] = np.append(obj.V_df.at[v, 'edges'], np.array([-1]))

        for c in range(1, len(obj.C_df)):
            c_verts = obj.C_df.at[c, 'nverts']

            if len(c_verts) > 1:
                c_coords = np.vstack(obj.V_df.loc[c_verts, 'coords'].to_list())
                c_coords = c_coords - np.mean(c_coords, axis=0)

                # Sort vertices in clockwise direction
                theta = np.mod(np.arctan2(c_coords[:,1], c_coords[:,0]), 2*np.pi)
                c_verts = c_verts[np.argsort(theta)]
                c_verts = np.append(c_verts, c_verts[0])
                if c not in obj.C_df.at[0, 'ncells']:
                    for v in range(0, len(c_verts)-1):
                        if (c_verts[v+1] in obj.V_df.at[c_verts[v], 'nverts']):
                            obj.C_df.at[c, 'edges'] = np.append(obj.C_df.at[c, 'edges'], np.intersect1d(obj.V_df.at[c_verts[v], 'edges'], obj.V_df.at[c_verts[v+1], 'edges']))
                        else:
                            obj.C_df.at[c, 'edges'] = np.append(obj.C_df.at[c, 'edges'], -1)

        return E_df

    def endpoints(self, image):
        # Define endpoint as pixel with only 1 4-connected neighbor
        # This requires the skeletonized image to be 4-connnected
        image = image.astype(np.int)
        k = np.array([[0,1,0],[1,0,1],[0,1,0]])
        neighborhood_count = ndi.convolve(image,k, mode='constant', cval=1)
        neighborhood_count[~image.astype(np.bool)] = 0
        return neighborhood_count == 1

    def segment_image(self, diameter=None, channels=[0,0], use_model='default'):
        """
        :param diameter: estimated diameter (px) for cells in image. If not specified, this will be estimated from the image
        :param channels: channels containing membrane and nuclear staining of image. 0 - Grayscale, 1 - R, 2 - G, 3 - B
        :param use_model: which Cellpose neural network to use. 'default' - Cyto2, 'custom' - custom trained model.
        :return: segmented image
        """
        from cellpose import models, utils, plot
        image = self.images.copy()
        image = image.astype(float)

        # Assuming membrane staining instead of cytoplasm, invert image before segmenting with Cellpose
        def normalise_image(image):
            image_norm = image.copy()
            ub = np.percentile(image, 99)
            lb = np.percentile(image, 1)
            image_norm[image_norm>ub] = ub
            image_norm[image_norm<lb] = lb
            image_norm = np.divide(image_norm-lb, ub - lb)
            return image_norm

        if channels != [0,0]:
            image[:,:,channels[0]-1] = 1-normalise_image(image[:,:,channels[0]-1])
            image[:,:,channels[1]-1] = normalise_image(image[:,:,channels[1]-1])
        else:
            image = normalise_image(image)

        if use_model == 'default':
            model = models.Cellpose(model_type='cyto2')
        elif use_model == 'custom':
            import pathlib
            src_path = str(pathlib.Path(__file__).parent.resolve())
            modeldir = f'{src_path}/cellpose_models/cellpose_residual_on_style_on_concatenation_off_train_folder_2022_03_24_00_26_36.748195'
            model = models.Cellpose(model_dir=modeldir, net_avg=False)
        else:
            return "Invalid use_model option. Available models are 'default', 'custom'."

        masks, flows, styles, diams = model.eval(image, diameter=diameter, channels=channels, progress=True)
        segmented_image = masks

        return segmented_image