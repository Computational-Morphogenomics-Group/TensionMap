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
from scipy.sparse import coo_matrix
import pandas as pd


def sx_grad(p1, p2, q1x, q1y, q2x, q2y, rx, ry):
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
    Dx = np.array([p1, np.zeros(p1.shape), q1x-rx, -p2, np.zeros(p1.shape), -q2x+rx]).T
    return Dx

def ry_grad(p1, p2, q1y, q2y, ry):
    Dy = np.array([np.zeros(p1.shape),p1,q1y-ry,np.zeros(p1.shape),-p2,-q2y+ry]).T
    return Dy

def rho_x_grad(p1,p2,q1x,q2x):
    t2 = p1-p2
    t3 = np.divide(1,t2)
    t4 = np.divide(1,np.power(t2,2))
    t5 = p1*q1x
    t6 = t5-p2*q2x
    z = np.zeros(p1.shape)
    dRhoX = np.array([p1*t3,z,z,q1x*t3-t4*t6,-p2*t3,z,z,-q2x*t3+t4*t6]).T
    return dRhoX

def rho_y_grad(p1,p2,q1y,q2y):
    t2 = p1-p2
    t3 = np.divide(1,t2)
    t4 = np.divide(1,np.power(t2,2))
    t5 = p1*q1y
    t6 = t5-p2*q2y
    z = np.zeros(p1.shape)
    dRhoY = np.array([z,p1*t3,z,q1y*t3-t4*t6,z,-p2*t3,z,-q2y*t3+t4*t6]).T
    return dRhoY

def radius_grad(p1,p2,q1x,q1y,q2x,q2y,t1,t2):
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

def rho_x_hess(p1,p2,q1x,q2x):
    t2 = p1-p2
    t3 = np.divide(1,np.power(t2,2))
    t4 = p1*t3
    t5 = np.divide(1,t2)
    t6 = np.divide(1,np.power(t2,3))
    t7 = p1*q1x
    t11 = p2*q2x
    t8 = t7-t11
    t9 = t6*t8*2
    t10 = p2*t3
    t12 = q1x*t3
    t13 = q2x*t3
    t14 = -t9+t12+t13
    t15 = -t5-t10
    z = np.zeros_like(p1)
    Hx = np.array([z,z,z,-t4+t5,z,z,z,t4,\
      z,z,z,z,z,z,z,z,\
      z,z,z,z,z,z,z,z,\
      t5-p1*t3,z,z,t9-q1x*t3*2,t10,z,z,t14,\
      z,z,z,t10,z,z,z,t15,\
      z,z,z,z,z,z,z,z,\
      z,z,z,z,z,z,z,z,\
      t4,z,z,t14,t15,z,z,t9-q2x*t3*2]).T
    return Hx

def rho_y_hess(p1,p2,q1y,q2y):
    t2 = p1-p2
    t3 = np.divide(1,np.power(t2,2))
    t4 = p1*t3
    t5 = np.divide(1,t2)
    t6 = np.divide(1,np.power(t2,3))
    t7 = p1*q1y
    t11 = p2*q2y
    t8 = t7-t11
    t9 = t6*t8*2
    t10 = p2*t3
    t12 = q1y*t3
    t13 = q2y*t3
    t14 = -t9+t12+t13
    t15 = -t5-t10
    z = np.zeros_like(p1)
    Hy = np.array([z,z,z,z,z,z,z,z,\
      z,z,z,-t4+t5,z,z,z,t4,\
      z,z,z,z,z,z,z,z,\
      z,t5-p1*t3,z,t9-q1y*t3*2,z,t10,z,t14,\
      z,z,z,z,z,z,z,z,\
      z,z,z,t10,z,z,z,t15,\
      z,z,z,z,z,z,z,z,\
      z,t4,z,t14,z,t15,z,t9-q2y*t3*2]).T
    return Hy

def radius_hess(p1,p2,q1x,q1y,q2x,q2y,t1,t2):
    t4 = p1-p2
    t5 = q1x-q2x
    t6 = q1y-q2y
    t7 = np.divide(1,np.power(t4,2))
    t17 = q1x*2
    t18 = q2x*2
    t8 = t17-t18
    t9 = t1-t2
    t10 = t4*t9
    t11 = np.power(t5,2)
    t12 = np.power(t6,2)
    t13 = t11+t12
    t19 = p1*p2*t13
    t14 = t10-t19
    t15 = np.power(p1,2)
    t16 = np.power(p2,2)
    t22 = t7*t14
    t20 = np.divide(1,np.power(-t22,1.5))
    t21 = np.divide(1,np.power(t4,4))
    t23 = np.divide(1,np.sqrt(-t22))
    t24 = np.divide(1,np.power(t4,3))
    t25 = np.power(t8,2)
    t26 = q1y*2
    t27 = q2y*2
    t28 = t26-t27
    t29 = p1*p2*t8*t20*t24*(0.25)
    t30 = t14*t24*2
    t31 = t8*t15*t16*t20*t21*t28*(0.25)
    t32 = p2*t13
    t33 = -t1+t2+t32
    t34 = t7*t33
    t35 = t30+t34
    t36 = p1*p2*t7*t23
    t37 = np.power(t28,2)
    t38 = p1*p2*t20*t24*t28*(0.25)
    t39 = p1*t13
    t40 = t1-t2+t39
    t43 = t7*t40
    t41 = t30-t43
    t42 = t7*t23*(0.5)
    t44 = np.divide(1,t4)
    t45 = t20*t35*t44*(0.25)
    t46 = t42+t45
    t47 = p2*t7*t8
    t69 = p1*p2*t8*t24*2
    t48 = t47-t69
    t49 = t23*t48*(0.5)
    t50 = p2*t7*t28
    t70 = p1*p2*t24*t28*2
    t51 = t50-t70
    t52 = t23*t51*(0.5)
    t53 = t14*t21*6
    t54 = t15*t16*t20*t21*t25*(0.25)
    t55 = p2*t7*t8*t23*(0.5)
    t56 = p1*p2*t8*t23*t24
    t57 = p1*p2*t7*t8*t20*t35*(0.25)
    t58 = p1*t7*t8*t23*(0.5)
    t59 = p1*p2*t7*t8*t20*t41*(0.25)
    t60 = t15*t16*t20*t21*t37*(0.25)
    t61 = -t36+t60
    t62 = p2*t7*t23*t28*(0.5)
    t63 = p1*p2*t23*t24*t28
    t64 = p1*p2*t7*t20*t28*t35*(0.25)
    t65 = p1*t7*t23*t28*(0.5)
    t66 = p1*p2*t7*t20*t28*t41*(0.25)
    t67 = t7*t20*(0.25)
    t68 = -t42-t45
    t71 = t20*t41*t44*(0.25)
    t72 = t24*t33*2
    t73 = t7*t13
    t74 = t53+t72+t73-t24*t40*2
    t75 = t23*t74*(0.5)
    t76 = t20*t35*t41*(0.25)
    t77 = t75+t76
    t78 = p1*t7*t8
    t79 = t69+t78
    t80 = t23*t79*(0.5)
    t81 = p1*t7*t28
    t82 = t70+t81
    t83 = t23*t82*(0.5)
    t84 = t42+t71
    Hr = np.array([p1*p2*t7*1/np.sqrt(-t7*t14)-t15*t16*t20*t21*t25*(0.25),-t31,t29,\
          t49-p1*p2*t7*t8*t20*t35*(0.25),-t36+t54,t31,-t29,t59+t80,t8*t15*t16*t20*t21*t28*(-0.25),\
          t36-t15*t16*t20*t21*t37*(0.25),t38,t52-p1*p2*t7*t20*t28*t35*(0.25),t31,t61,-t38,t66+t83,\
          t29,t38,t7*t20*(-0.25),t46,-t29,-t38,t67,-t42-t71,t55-p1*p2*t8*t23*t24-p1*p2*t7*t8*t20*t35*(0.25),\
          t62-p1*p2*t23*t24*t28-p1*p2*t7*t20*t28*t35*(0.25),t46,\
          t20*np.power(t35,2)*(-0.25)-t23*(t53+t24*t33*4)*(0.5),-t55+t56+t57,-t62+t63+t64,t68,t77,\
          t54-p1*p2*t7*t23,t31,-t29,-t49+t57,t36-t54,-t31,t29,-t59-t80,t31,t61,-t38,-t52+t64,-t31,t36-t60,\
          t38,-t66-t83,-t29,-t38,t67,t68,t29,t38,-t67,t84,t56+t58+t59,t63+t65+t66,-t42-t20*t41*t44*(0.25),\
          t77,-t56-t58-t59,-t63-t65-t66,t84,t20*np.power(t41,2)*(-0.25)-t23*(t53-t24*t40*4)*(0.5)]).T
    return Hr

def const_hess(p1,p2,q1x,q1y,q2x,q2y):
    t2 = q1x*2
    t3 = q2x*2
    t4 = t2-t3
    t5 = p1*p2*2
    t6 = q1y*2
    t7 = q2y*2
    t8 = t6-t7
    t9 = q1x-q2x
    t10 = q1y-q2y
    t11 = p2*t4
    t12 = p2*t8
    t13 = p1*t4
    t14 = p1*t8
    t15 = np.power(t9,2)
    t16 = np.power(t10,2)
    t17 = -t15-t16
    Hc = np.array([p1*p2*-2,np.zeros_like(p1),np.zeros_like(p1),-p2*t4,t5,np.zeros_like(p1),np.zeros_like(p1),\
             -t13,np.zeros_like(p1),-t5,np.zeros_like(p1),-p2*t8,np.zeros_like(p1),t5,np.zeros_like(p1),-t14,\
             np.zeros_like(p1),np.zeros_like(p1),np.zeros_like(p1),np.ones_like(p1),np.zeros_like(p1),np.zeros_like(p1),\
             np.zeros_like(p1),-np.ones_like(p1),-p2*t4,-p2*t8,np.ones_like(p1),np.zeros_like(p1),t11,t12,-np.ones_like(p1),\
             t17,t5,np.zeros_like(p1),np.zeros_like(p1),t11,-t5,np.zeros_like(p1),np.zeros_like(p1),t13,np.zeros_like(p1),\
             t5,np.zeros_like(p1),t12,np.zeros_like(p1),-t5,np.zeros_like(p1),t14,np.zeros_like(p1),np.zeros_like(p1),np.zeros_like(p1),\
             -np.ones_like(p1),np.zeros_like(p1),np.zeros_like(p1),np.zeros_like(p1),np.ones_like(p1),-p1*t4,-p1*t8,-np.ones_like(p1),\
             t17,t13,t14,np.ones_like(p1),np.zeros_like(p1)]).T
    return Hc

class VMSI():
    
    def __init__(self, vertices, cells, edges, width=500, height=500):
        self.vertices = vertices
        self.cells = cells
        self.edges = edges
        self.width = width
        self.height = height

        # Mark fourfold vertices
        self.vertices['fourfold'] = [(np.shape(nverts)[0] != 3) for nverts in self.vertices['nverts']]

        # Initialize new columns
        self.edges['radius'] = np.zeros(len(self.edges))
        self.edges['rho'] = tuple((0,0) for _ in range(len(self.edges)))
        self.edges['fitenergy'] = np.zeros(len(self.edges))

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
        self.bulk_cells = np.array(range(1,len(self.cells)))

        boundary_cells = np.unique(self.cells.at[0, 'ncells'])

        # Remove boundary cells and cells surrounded by boundary cells from bulk cells
        self.bulk_cells = self.bulk_cells[np.isin(self.bulk_cells, boundary_cells, invert=True)]
        bad_cells = np.array([])
        for cell in self.bulk_cells:
            if np.sum(np.isin(self.cells.at[cell, 'ncells'], self.bulk_cells)) == 0:
                bad_cells = np.append(bad_cells, cell)
        self.bulk_cells = self.bulk_cells[np.isin(self.bulk_cells, bad_cells, invert=True)]
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
        # Todo: filter edges to identify bad edges - is this necessary?

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

            r[i] = np.mean(np.power(np.array([np.linalg.norm(r1 - rho[i]), np.linalg.norm(r2 - rho[i])]), 2))
            r_flat[i] = p[np.where(self.dC[i,:] == 1)] * p[np.where(self.dC[i,:] == -1)] * q_sq[i]

        dP = np.matmul(self.dC, p)
        r = np.multiply(r, np.power(dP, 2))

        A = np.multiply(self.dC.T, dP).T
        b = r_flat - r

        theta = np.linalg.lstsq(np.vstack((A, np.ones(A.shape[1]))),np.concatenate((b, np.array([0]))))[0]

        return theta


    def initial_minimization(self):

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

        scale = 0.5 * (np.mean(np.linalg.norm(t1_0, axis=1)) + np.mean(np.linalg.norm(t2_0, axis=1)))

        # Energy function
        def energy(x):
            x = x.reshape(x0.shape, order='F')
            q = x[:,0:2]
            p = x[:,2]

            b = np.matmul(self.dC, np.multiply(q.T,p).T)
            delta_p = np.matmul(self.dC, p)

            t1 = np.divide((b - (np.multiply(r1.T,delta_p).T)).T,np.linalg.norm(b - (np.multiply(r1.T,delta_p).T), axis=1)).T
            t2 = np.divide((b - (np.multiply(r2.T,delta_p).T)).T,np.linalg.norm(b - (np.multiply(r2.T,delta_p).T), axis=1)).T

            E = 0.5 * np.mean(np.power(np.sum(t1 * tau_1, axis=1), 2) +
                              np.power(np.sum(t2 * tau_2, axis=1), 2))
            return E

        def e_jac(x):
            x = x.reshape(x0.shape, order='F')
            q = x[:,0:2]
            p = x[:,2]

            b = np.matmul(self.dC, np.multiply(q.T,p).T)
            delta_p = np.matmul(self.dC, p)

            t1 = np.divide((b - (np.multiply(r1.T,delta_p).T)).T,np.linalg.norm(b - (np.multiply(r1.T,delta_p).T), axis=1)).T
            t2 = np.divide((b - (np.multiply(r2.T,delta_p).T)).T,np.linalg.norm(b - (np.multiply(r2.T,delta_p).T), axis=1)).T

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
            return dE

        # Define constraints and bounds
        # nonlinear constraint
        def nonlinear_con(x):
            x = x.reshape(x0.shape, order='F')
            q = x[:,0:2]
            p = x[:,2]

            b = np.matmul(self.dC, np.multiply(q.T,p).T)
            delta_p = np.matmul(self.dC, p)

            l1 = np.linalg.norm(b - (np.multiply(r1.T,delta_p).T), axis=1)
            l2 = np.linalg.norm(b - (np.multiply(r2.T,delta_p).T), axis=1)

            E = 0.5*(np.mean(l1) + np.mean(l2)) - scale
            return E

        def nlc_jac(x):
            x = x.reshape(x0.shape, order='F')
            q = x[:,0:2]
            p = x[:,2]

            b = np.matmul(self.dC, np.multiply(q.T,p).T)
            delta_p = np.matmul(self.dC, p)

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

            dE = np.bincount(rows, weights=np.ravel(dE/self.dC.shape[0],order='F'))
            return dE


        nonlincon = scipy.optimize.NonlinearConstraint(nonlinear_con, 0, 0, jac=nlc_jac)

        # linear constraint
        Aeq = np.concatenate((np.zeros(x0.shape[0]*2), np.ones(x0.shape[0])))
        lincon = scipy.optimize.LinearConstraint(Aeq, np.mean(p0)*p0.shape[0], np.mean(p0)*p0.shape[0])

        constraints = [nonlincon, lincon]

        bounds = scipy.optimize.Bounds(lb = np.concatenate((-np.inf*np.ones(x0.shape[0]), -np.inf*np.ones(x0.shape[0]), 0.001*np.ones(x0.shape[0]))),
                                       ub = np.concatenate((np.inf*np.ones(x0.shape[0]), np.inf*np.ones(x0.shape[0]), 1000*np.ones(x0.shape[0]))))

        res = scipy.optimize.minimize(energy,np.ravel(x0, order='F'),jac=e_jac,method='trust-constr',bounds=bounds,
                                   constraints=constraints,options={'maxiter':2000}, tol=1e-6)
        x = res.x
        x = x.reshape(x0.shape, order='F')

        q = x[:,0:2]
        p = x[:,2]

        self.generate_circular_arcs()

        theta0 = self.estimate_theta(x)

        def theta_energy(theta):
            dP = p[self.cell_pairs[:,0]] - p[self.cell_pairs[:,1]]
            dT = theta[self.cell_pairs[:,0]] - theta[self.cell_pairs[:,1]]
            dQ = q[self.cell_pairs[:,0],:] - q[self.cell_pairs[:,1],:]
            QL = np.sum(np.power(dQ, 2), axis=1)

            rho = np.divide(np.matmul(self.dC,np.multiply(p, q.T).T).T, dP).T
            r_sq = np.divide(((p[self.cell_pairs[:,0]] * p[self.cell_pairs[:,1]] * QL) - (dP * dT)),np.power(dP, 2))
            r_sq[r_sq<0] = 0

            r = np.sqrt(r_sq)

            delta_x = np.subtract(rho[:,0],self.edgearc_x.T).T
            delta_y = np.subtract(rho[:,1],self.edgearc_y.T).T

            dMag = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))

            E = 0.5 * np.mean(np.sum(np.power(np.subtract(dMag.T, r).T, 2), axis=1))
            return E

        def theta_e_jac(theta):
            dP = p[self.cell_pairs[:,0]] - p[self.cell_pairs[:,1]]
            dT = theta[self.cell_pairs[:,0]] - theta[self.cell_pairs[:,1]]
            dQ = q[self.cell_pairs[:,0],:] - q[self.cell_pairs[:,1],:]
            QL = np.sum(np.power(dQ, 2), axis=1)

            rho = np.divide(np.matmul(self.dC,np.multiply(p, q.T).T).T, dP).T
            r_sq = np.divide(((p[self.cell_pairs[:,0]] * p[self.cell_pairs[:,1]] * QL) - (dP * dT)),np.power(dP, 2))
            ind_z = np.where(r_sq<0)
            r_sq[ind_z] = 0

            r = np.sqrt(r_sq)

            delta_x = np.subtract(rho[:,0],self.edgearc_x.T).T
            delta_y = np.subtract(rho[:,1],self.edgearc_y.T).T

            dMag = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))
            d = np.subtract(dMag.T, r).T

            avg_d = np.sum(d, axis=1)
            dR = radius_grad_theta(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],0], q[self.cell_pairs[:,0],1], q[self.cell_pairs[:,1],0], q[self.cell_pairs[:,1],1], theta[self.cell_pairs[:,0]], theta[self.cell_pairs[:,1]])
            dR[ind_z,:] = 0

            dE = np.divide(-np.multiply(avg_d, dR.T).T,self.dC.shape[0])
            rows = np.concatenate([self.cell_pairs[:,0], self.cell_pairs[:,1]])

            dE = np.bincount(rows, weights=np.ravel(dE,order='F'))
            return dE

        if (theta_energy(np.zeros_like(theta0)) < theta_energy(theta0)):
            theta0 = np.zeros_like(theta0)

        # linear constraints
        Aeq = np.ones((1, len(self.involved_cells)))
        eq_lincon = scipy.optimize.LinearConstraint(Aeq, 0, 0)

        dP = p[self.cell_pairs[:,0]] - p[self.cell_pairs[:,1]]
        dQ = q[self.cell_pairs[:,0],:] - q[self.cell_pairs[:,1],:]
        QL = np.sum(np.power(dQ, 2), axis=1)

        A = np.multiply(dP, self.dC.T).T
        b = p[self.cell_pairs[:,0]] * p[self.cell_pairs[:,1]] * QL
        ineq_lincon = scipy.optimize.LinearConstraint(A,-np.inf,b)

        constraints = [eq_lincon, ineq_lincon]

        res = scipy.optimize.minimize(theta_energy,theta0,jac=theta_e_jac,method='trust-constr',
                                        constraints=constraints,options={'maxiter':2000}, tol=1e-6)
        theta = res.x
        return x, theta

    
    def fit(self):
        
        """
        
        Perform the minimization of equation 5 with respect to the variables (q, z, p)
        
        """
        self.prepare_data()

        print("Initial minimization")
        # Perform initial minimization for p, q and theta
        x0, theta0 = self.initial_minimization()
        X0 = np.vstack([x0.T, theta0]).T

        print("Main minimization")

        def energy(X):
            X = X.reshape(X0.shape, order='F')

            q = X[:,0:2]
            p = X[:,2]
            theta = X[:,3]

            dP = p[self.cell_pairs[:,0]] - p[self.cell_pairs[:,1]]
            dT = theta[self.cell_pairs[:,0]] - theta[self.cell_pairs[:,1]]
            dQ = q[self.cell_pairs[:,0],:] - q[self.cell_pairs[:,1],:]
            QL = np.sum(np.power(dQ, 2), axis=1)

            rho = np.divide(np.matmul(self.dC,np.multiply(p, q.T).T).T, dP).T
            r_sq = np.divide(((p[self.cell_pairs[:,0]] * p[self.cell_pairs[:,1]] * QL) - (dP * dT)),np.power(dP, 2))
            r_sq[r_sq < 0] = 0

            r = np.sqrt(r_sq)

            delta_x = np.subtract(rho[:,0],self.edgearc_x.T).T
            delta_y = np.subtract(rho[:,1],self.edgearc_y.T).T

            dMag = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))

            E = 0.5 * np.mean(np.sum(np.power(np.subtract(dMag.T, r).T, 2), axis=1))
            return E

        def e_jac(X):
            X = X.reshape(X0.shape, order='F')

            q = X[:,0:2]
            p = X[:,2]
            theta = X[:,3]

            dP = p[self.cell_pairs[:,0]] - p[self.cell_pairs[:,1]]
            dT = theta[self.cell_pairs[:,0]] - theta[self.cell_pairs[:,1]]
            dQ = q[self.cell_pairs[:,0],:] - q[self.cell_pairs[:,1],:]
            QL = np.sum(np.power(dQ, 2), axis=1)

            rho = np.divide(np.matmul(self.dC,np.multiply(p, q.T).T).T, dP).T
            r_sq = np.divide(((p[self.cell_pairs[:,0]] * p[self.cell_pairs[:,1]] * QL) - (dP * dT)),np.power(dP, 2))
            ind_z = np.where(r_sq<=0)
            r_sq[ind_z] = 0

            r = np.sqrt(r_sq)

            delta_x = np.subtract(rho[:,0],self.edgearc_x.T).T
            delta_y = np.subtract(rho[:,1],self.edgearc_y.T).T

            dMag = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))
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
            return dE

        def load_hessian(h):
            # It looks like Z should always be constant
            Z = 8
            pF = int(Z*(Z+1)/2)
            delta = 4
            rows = np.zeros(pF * self.dC.shape[0])
            cols = np.zeros(pF * self.dC.shape[0])
            vals = np.zeros(pF * self.dC.shape[0])

            n = 0
            for i in range(8):
                for j in range(i,8):
                    I = n*self.dC.shape[0] + np.arange(0,self.dC.shape[0])
                    rows[I] = I

                    if i < Z/2:
                        rows[I] = self.cell_pairs[:,0] + i*self.dC.shape[1]
                    else:
                        rows[I] = self.cell_pairs[:,1] + (i-delta)*self.dC.shape[1]

                    if j < Z/2:
                        cols[I] = self.cell_pairs[:,0] + j*self.dC.shape[1]
                    else:
                        cols[I] = self.cell_pairs[:,1] + (j-delta)*self.dC.shape[1]

                    vals[I] = h[:,i,j]
                    n += 1

            H = coo_matrix((vals[vals!=0], (rows[vals!=0],cols[vals!=0])), shape=(int(self.dC.shape[1]*Z/2),int(self.dC.shape[1]*Z/2)))
            return H

        def e_hess(X):
            X = X.reshape(X0.shape, order='F')

            q = X[:,0:2]
            p = X[:,2]
            theta = X[:,3]

            dP = p[self.cell_pairs[:,0]] - p[self.cell_pairs[:,1]]
            dT = theta[self.cell_pairs[:,0]] - theta[self.cell_pairs[:,1]]
            dQ = q[self.cell_pairs[:,0],:] - q[self.cell_pairs[:,1],:]
            QL = np.sum(np.power(dQ, 2), axis=1)

            rho = np.divide(np.matmul(self.dC,np.multiply(p, q.T).T).T, dP).T
            r_sq = np.divide(((p[self.cell_pairs[:,0]] * p[self.cell_pairs[:,1]] * QL) - (dP * dT)),np.power(dP, 2))
            ind_z = np.where(r_sq<=0)
            nind_z = np.where(r_sq>0)
            r_sq[ind_z] = 0

            r = np.sqrt(r_sq)

            delta_x = np.subtract(rho[:,0],self.edgearc_x.T).T
            delta_y = np.subtract(rho[:,1],self.edgearc_y.T).T

            dMag = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))
            delta_x = np.divide(delta_x, dMag)
            delta_y = np.divide(delta_y, dMag)

            d = np.subtract(dMag.T, r).T

            dRhoX = rho_x_grad(p[self.cell_pairs[:,0]],p[self.cell_pairs[:,1]],q[self.cell_pairs[:,0],0],q[self.cell_pairs[:,1],0])
            dRhoY = rho_y_grad(p[self.cell_pairs[:,0]],p[self.cell_pairs[:,1]],q[self.cell_pairs[:,0],1],q[self.cell_pairs[:,1],1])
            dR = radius_grad(p[self.cell_pairs[:,0]],p[self.cell_pairs[:,1]],q[self.cell_pairs[:,0],0],q[self.cell_pairs[:,0],1],q[self.cell_pairs[:,1],0],q[self.cell_pairs[:,1],1],theta[self.cell_pairs[:,0]],theta[self.cell_pairs[:,1]])
            dR[ind_z] = 0

            r_ratio = np.divide(r, dMag.T).T

            dNormXX = np.sum(np.multiply(r_ratio,np.multiply(delta_x,delta_x)),axis=1)
            dNormXY = np.sum(np.multiply(r_ratio,np.multiply(delta_x,delta_y)),axis=1)
            dNormYY = np.sum(np.multiply(r_ratio,np.multiply(delta_y,delta_y)),axis=1)

            dRhoX = dRhoX.reshape(1,dRhoX.shape[0],dRhoX.shape[1])
            dRhoY = dRhoY.reshape(1,dRhoY.shape[0],dRhoY.shape[1])
            dR = dR.reshape(1,dR.shape[0],dR.shape[1])

            dRhoXT = np.transpose(dRhoX, (2,0,1))
            dRhoYT = np.transpose(dRhoY, (2,0,1))
            dRT = np.transpose(dR, (2,0,1))

            H = np.multiply(dRhoXT, np.transpose(np.multiply(dNormXX, np.transpose(dRhoX, (1,0,2)).T),(1,0,2))) + \
                np.multiply(dRhoXT, np.transpose(np.multiply(dNormXY, np.transpose(dRhoY, (1,0,2)).T),(1,0,2))) + \
                np.multiply(dRhoYT, np.transpose(np.multiply(dNormXY, np.transpose(dRhoX, (1,0,2)).T),(1,0,2))) + \
                np.multiply(dRhoYT, np.transpose(np.multiply(dNormYY, np.transpose(dRhoY, (1,0,2)).T),(1,0,2)))

            dNormX = np.sum(delta_x, axis=1)[nind_z]
            dNormY = np.sum(delta_y, axis=1)[nind_z]

            H = H - np.multiply(dRT, np.transpose(np.multiply(dNormX, np.transpose(dRhoX, (1,0,2)).T),(1,0,2))) - \
                np.multiply(dRT, np.transpose(np.multiply(dNormY, np.transpose(dRhoY, (1,0,2)).T),(1,0,2))) - \
                np.multiply(dRhoXT, np.transpose(np.multiply(dNormX, np.transpose(dR, (1,0,2)).T),(1,0,2))) - \
                np.multiply(dRhoYT, np.transpose(np.multiply(dNormY, np.transpose(dR, (1,0,2)).T),(1,0,2)))

            nPix = (np.ones(self.dC.shape[0])*self.avg_edge_length)[nind_z]
            H = H + np.multiply(dRT, np.transpose(np.multiply(nPix, np.transpose(dR, (1,0,2)).T),(1,0,2)))

            avg_ratio = np.sum(np.divide(d, dMag),axis=1)
            H = H + np.multiply(dRhoXT, np.transpose(np.multiply(avg_ratio, np.transpose(dRhoX, (1,0,2)).T),(1,0,2))) + \
                np.multiply(dRhoYT, np.transpose(np.multiply(avg_ratio, np.transpose(dRhoY, (1,0,2)).T),(1,0,2)))
            H = np.divide(np.transpose(H, (2,0,1)),self.dC.shape[0])

            H = load_hessian(H)
            H = H + H.T
            return H


        def nonlinear_con(X):
            X = X.reshape(X0.shape, order='F')

            q = X[:,0:2]
            p = X[:,2]
            theta = X[:,3]

            return (np.matmul(self.dC,p) * np.matmul(self.dC, theta)) - (p[self.cell_pairs[:,0]] * p[self.cell_pairs[:,1]] * np.sum(np.power(np.matmul(self.dC, q), 2), axis=1))

        def nlc_jac(X):
            X = X.reshape(X0.shape, order='F')

            q = X[:,0:2]
            p = X[:,2]
            theta = X[:,3]

            dP = np.matmul(self.dC,p)
            dT = np.matmul(self.dC, theta)
            dQ = np.matmul(self.dC, q)
            QL = np.sum(np.power(dQ, 2), axis=1)

            gX = np.multiply(self.dC.T, -2*p[self.cell_pairs[:,0]]*p[self.cell_pairs[:,1]]*dQ[:,0]).T
            gY = np.multiply(self.dC.T, -2*p[self.cell_pairs[:,0]]*p[self.cell_pairs[:,1]]*dQ[:,1]).T
            gTh = np.multiply(self.dC.T, dP).T
            gP = np.multiply(self.dC.T, dT).T - np.divide(np.multiply(QL*p[self.cell_pairs[:,0]]*p[self.cell_pairs[:,1]],np.abs(self.dC).T).T,p)
            dE = np.array([gX,gY,gP,gTh]).T
            return dE

        def nlc_hess(X, v):
            """
            :param X: values
            :param v: ndarray containing lagrange multipliers
            :return: hessian
            """

            X = X.reshape(X0.shape, order='F')

            q = X[:,0:2]
            p = X[:,2]
            theta = X[:,3]

            dP = p[self.cell_pairs[:,0]] - p[self.cell_pairs[:,1]]
            dT = theta[self.cell_pairs[:,0]] - theta[self.cell_pairs[:,1]]
            dQ = q[self.cell_pairs[:,0],:] - q[self.cell_pairs[:,1],:]
            QL = np.sum(np.power(dQ, 2), axis=1)

            rho = np.divide(np.matmul(self.dC,np.multiply(p, q.T).T).T, dP).T
            r_sq = np.divide(((p[self.cell_pairs[:,0]] * p[self.cell_pairs[:,1]] * QL) - (dP * dT)),np.power(dP, 2))
            ind_z = np.where(r_sq<=0)
            nind_z = np.where(r_sq>0)
            r_sq[ind_z] = 0

            r = np.sqrt(r_sq)

            delta_x = np.subtract(rho[:,0],self.edgearc_x.T).T
            delta_y = np.subtract(rho[:,1],self.edgearc_y.T).T

            dMag = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))
            delta_x = np.divide(delta_x, dMag)
            delta_y = np.divide(delta_y, dMag)

            d = np.subtract(dMag.T, r).T

            dNormX = np.sum(np.multiply(delta_x, d),axis=1)
            dNormY = np.sum(np.multiply(delta_y, d),axis=1)
            d_avg = np.sum(d, axis=1)[nind_z]

            hRhoX = rho_x_hess(p[self.cell_pairs[:,0]],p[self.cell_pairs[:,1]],q[self.cell_pairs[:,0],0],q[self.cell_pairs[:,1],0])
            hRhoY = rho_y_hess(p[self.cell_pairs[:,0]],p[self.cell_pairs[:,1]],q[self.cell_pairs[:,0],1],q[self.cell_pairs[:,1],1])
            hR = radius_hess(p[self.cell_pairs[:,0]],p[self.cell_pairs[:,1]],q[self.cell_pairs[:,0],0],q[self.cell_pairs[:,0],1],q[self.cell_pairs[:,1],0],q[self.cell_pairs[:,1],1],theta[self.cell_pairs[:,0]],theta[self.cell_pairs[:,1]])
            hCon = const_hess(p[self.cell_pairs[:,0]],p[self.cell_pairs[:,1]],q[self.cell_pairs[:,0],0],q[self.cell_pairs[:,0],1],q[self.cell_pairs[:,1],0],q[self.cell_pairs[:,1],1])

            hRhoX = hRhoX.reshape(hRhoX.shape[0],8,8)
            hRhoY = hRhoY.reshape(hRhoY.shape[0],8,8)
            hR = hR.reshape(hR.shape[0],8,8)
            hCon = hCon.reshape(hCon.shape[0],8,8)

            H = np.multiply(hRhoX.T, dNormX).T + np.multiply(hRhoY.T, dNormY).T - \
                np.multiply(hR.T, d_avg).T + np.multiply(hCon.T, v).T

            H = load_hessian(H)
            H = H + H.T
            return H

        nonlincon = scipy.optimize.NonlinearConstraint(nonlinear_con,lb=-np.inf,ub=0,jac=nlc_jac,hess=nlc_hess)

        Aeq = np.vstack((np.concatenate((np.zeros(2*theta0.shape[0]), np.ones(theta0.shape[0])/theta0.shape[0], np.zeros(theta0.shape[0]))),
                         np.concatenate((np.zeros(3*theta0.shape[0]), np.ones(theta0.shape[0])/theta0.shape[0]))))
        beq = np.array([np.mean(X0[:,2]), np.mean(X0[:,3])])
        lincon = scipy.optimize.LinearConstraint(Aeq, lb=beq, ub=beq)

        constraints= [nonlincon, lincon]
        bounds = scipy.optimize.Bounds(lb=np.concatenate((-np.inf*np.ones(2*theta0.shape[0]), 0.001*np.ones(theta0.shape[0]), -np.inf*np.ones(theta0.shape[0]))),
                                       ub=np.concatenate((np.inf*np.ones(2*theta0.shape[0]), 1000*np.ones(theta0.shape[0]), np.inf*np.ones(theta0.shape[0]))))

        res = scipy.optimize.minimize(energy,np.ravel(X0,order='F'),jac=e_jac,hess=e_hess,method='trust-constr',constraints=constraints,
                                      bounds=bounds,options={'maxiter':2000}, tol=1e-6)
        X = res.x
        X = X.reshape(X0.shape, order='F')

        q = X[:,0:2]
        p = X[:,2]
        theta = X[:,3]

        # compute the tensions
        self.get_tensions(q, p, theta)
        
        return q, z, p


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
