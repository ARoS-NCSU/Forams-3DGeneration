import bpy
import os
import numpy as np
import json
import sys


def loadNextForam(file_loc, sample_id):
    #return boundingbox
    bpy.ops.import_scene.obj(filepath=file_loc)
    obj_list = [o for o in bpy.context.selected_objects]
    if len(obj_list) > 1:
        print('ERROR TOO MANY OBJECTS PER OBJ FILE')
    obj_list[0].name = sample_id
    for ob in bpy.context.selected_objects:
        ob.select = False

def loadAllForams(sample_ids):
    data_folder = os.path.abspath('../data/synthetic/')
    for sample_id in sample_ids:
        next_file = os.path.join(data_folder, sample_id + '.obj')
        next_dims = loadNextForam(next_file, sample_id)

def shiftAllForams(sample_ids, dims, dims_present, final_size):
    max_bbox = [1.5*final_size]*3
    all_dims = np.ones(3, dtype=np.uint8)
    for i in range(len(dims)):
        all_dims[i] = dims[i]
    flat_idcs = np.arange(len(sample_ids), dtype=np.uint32)
    for fi, sample_id in zip(flat_idcs, sample_ids):
        idcs = np.unravel_index(fi, all_dims)
        offset = np.multiply(idcs,max_bbox)
        translateScaleObj(sample_id, offset, final_size)
    addArrowLabels(all_dims, dims_present, max_bbox)

def addArrowLabels(all_dims, dims_present, max_bbox):
    labels_base = ['Number Chambers', 'Growth', 'Distance', 'Planar Angle', 'Z Angle']
    labels = [labels_base[i] for i,val in enumerate(dims_present) if val]
    rotations = [[0,1.5708,0],[-1.5708,0,0],[0,0,0]]
    for i,dim in enumerate(all_dims):
        if dim==1:
            break
        next_loc = np.zeros(3)
        next_loc[i] = max_bbox[i]*(all_dims[i])
        bpy.ops.object.empty_add(type='SINGLE_ARROW', rotation=rotations[i], location=next_loc)
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.scene.objects.active = None
        bpy.data.objects['Empty'].select = True
        bpy.context.scene.objects.active = bpy.data.objects['Empty']
        bpy.context.active_object.name = labels[i]
        bpy.context.object.show_name = True
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.scene.objects.active = None


def translateScaleObj(obj_name, offset, final_size):
    o = bpy.data.objects[obj_name]
    bpy.ops.object.select_all( action = 'DESELECT' )
    o.select = True
    bpy.context.scene.objects.active = o
    largest_side = max(o.dimensions)
    scale_val = final_size/largest_side
    bpy.ops.transform.resize(value = [scale_val,scale_val,scale_val])
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY',center='BOUNDS')
    o.location = offset
    #bpy.ops.transform.translate(value = offset)
    bpy.ops.object.select_all( action = 'DESELECT' )

def runVisualization(arglist):
    if 'json' in arglist[0]:
        with open(arglist[0],'r') as f:
            arg_dict = json.load(f)
        sample_ids = arg_dict['samples']
        dims = arg_dict['dim_shape']
        dims_present = arg_dict['dims']
        loadAllForams(sample_ids)
        shiftAllForams(sample_ids, dims, dims_present, 0.2)
    else:
        loadAllForams([arglist[0]])


argv = sys.argv
arglist = argv[argv.index("--") + 1:]
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
runVisualization(arglist)
