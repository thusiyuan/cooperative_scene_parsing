"""
Created on April, 2018

@author: Siyuan Huang

visualize code for SUNRGBD
"""


import matplotlib.pyplot as plt
import numpy as np
from random import random as rand
from mpl_toolkits.mplot3d import Axes3D
import mayavi.mlab as mlab


def show_2dboxes(im, bdbs, color_list=[], random_color=True, scale=1.0):
    """
       Visualize the bounding boxes with the image

       Parameters
       ----------
       im : numpy array (W, H, 3)
       bdbs : list of dicts
           Keys: {'x1', 'y1', 'x2', 'y2', 'classname'}
           The (x1, y1) position is at the top left corner,
           the (x2, y2) position is at the bottom right corner
        color_list: list of colors
    """
    plt.cla()
    plt.axis('off')
    plt.imshow(im)
    for i, bdb in enumerate(bdbs):
        if bdb is None:
            continue
        bbox = np.array([bdb['x1'], bdb['y1'], bdb['x2'], bdb['y2']]) * scale
        if random_color is False:
            color = color_list[i]
        else:
            color = (rand(), rand(), rand())
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, edgecolor=color, linewidth=2.5)
        plt.gca().add_patch(rect)
        plt.gca().text(bbox[0], bbox[1], '{:s}'.format(bdb['classname']), bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
    plt.show()
    return im


# def show_2dboxes_8pts(im, points, bdbs)
#

def plot_world_point(ax, p1, p2, color='r-'):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color)


def plot_cuboid(ax, p1, p2, p3, p4, p5, p6, p7, p8, color='r-'):
    plot_world_point(ax, p1, p2, color)
    plot_world_point(ax, p2, p3, color)
    plot_world_point(ax, p3, p4, color)
    plot_world_point(ax, p4, p1, color)
    plot_world_point(ax, p5, p6, color)
    plot_world_point(ax, p6, p7, color)
    plot_world_point(ax, p7, p8, color)
    plot_world_point(ax, p8, p5, color)
    plot_world_point(ax, p1, p5, color)
    plot_world_point(ax, p2, p6, color)
    plot_world_point(ax, p3, p7, color)
    plot_world_point(ax, p4, p8, color)
    return p1, p2, p3, p4, p5, p6, p7, p8


def show_3d_box(boxes):
    """
    :param box: 8 x 3 numpy array
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    for box in boxes:
        plot_cuboid(ax, box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], 'r-')
    plt.show()


def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=1, draw_text=True, text_scale=(1,1,1), color_list=None):
    ''' Draw 3D bounding boxes
        Modified from https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/kitti_object.py
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%n, scale=text_scale, color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    #mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(float(int(hex[i:i + hlen / 3], 16)) / 255 for i in range(0, hlen, hlen / 3))


def object_color(obj_id, if_rgb, if_random):
    obj_color = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
     "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
     "#8dd3c7", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5",
     "#ffed6f", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999", "#621e15",
     "#e59076", "#128dcd", "#083c52", "#64c5f2", "#61afaf", "#0f7369", "#9c9da1", "#365e96", "#983334", "#77973d",
     "#5d437c", "#36869f", "#d1702f", "#8197c5", "#c47f80", "#acc484", "#9887b0", "#2d588a", "#58954c", "#e9a044",
     "#c12f32", "#723e77", "#7d807f", "#9c9ede", "#7375b5", "#4a5584", "#cedb9c", "#b5cf6b", "#8ca252", "#637939",
     "#e7cb94", "#e7ba52", "#bd9e39", "#8c6d31", "#e7969c", "#d6616b", "#ad494a", "#843c39", "#de9ed6", "#ce6dbd",
     "#a55194", "#7b4173", "#000000", "#0000FF"]
    length = len(obj_color)
    if if_random:
        obj_id = np.random.randint(length)
    if if_rgb:
        return hex_to_rgb(obj_color[obj_id][1:])
    else:
        return obj_color[obj_id]


def show_whole_room(layout, obj_vertexes):
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    fig = draw_gt_boxes3d(layout, fig, draw_text=False, color=(1, 0, 0),
                          line_width=3)
    fig = draw_gt_boxes3d(obj_vertexes, fig, draw_text=False,
                          color_list=[object_color(1, True, True) for _ in range(obj_vertexes.shape[0])], line_width=2)
    mlab.view(azimuth=0)
    # mlab.view(azimuth=0, elevation=180-roll)
    # cam = fig.scene.camera
    # cam.position = [0, 0, 0]
    mlab.show()
    print 'show whole room'