############################################################################################################
# Oi = centre of the chamber, and origin of the reference system as well
# Ui = aperture point -- describes the location of the chamber aperture
# Vi = growth vector -- it is attached to the aperture Ui and points to the centre of the new aperture Oi+1
# reference growth axis = it is a base direction for the growth vector Vi
# lmax = maximum length of Vi
# phi = devaition angle
# beta = angle by which Vi is rotated around the reference axis
# GF = Chamber Growth Factor (kx = ky = kz = GF)
# TF = Translation Factor
#
# Steps:
# 1. Take Oi as centre and build shape around it. Call the intersection of the shape and y_axis as U0 (first aperture).
# 2. Calculate Vi- direction (Oi-Ui) and scale the length (s*lmax). Add devaition and rotation.
# 3. Draw the next chamber taking Vi end point as the centre of next chamber.
# 4. Caluclate shortest exterior point on next chamber from the current chamber. Make it next aperture.
# 5. Repeat from 2-5
############################################################################################################

import bpy
import os
import glob
import numpy as np
import json
from math import radians, degrees
import math
import random
import mathutils
import sys
import itertools

def deselectAll():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.scene.objects.active = None

def joinChambers(cham_map):
    deselectAll()
    is_set = False
    for cham_name in cham_map.keys():
        cham_map[cham_name].select = True
        if not is_set:
            bpy.context.scene.objects.active = cham_map[cham_name]
            is_set = True
    bpy.ops.object.join()
    return bpy.context.active_object


def union(orig_obj, union_obj, base_scale):
    deselectAll()
    orig_obj.select = True
    bpy.context.scene.objects.active = orig_obj
    mod = get_boolean_modifier(orig_obj, union_obj, 'UNION', base_scale)
    bpy.ops.object.modifier_apply(apply_as='DATA',modifier=mod.name)
    bpy.context.scene.update()
    deselectAll()
    cleanObj(orig_obj)

def removeObj(rem_obj):
    bpy.data.objects.remove(rem_obj, do_unlink=True)
    bpy.context.scene.update()

def subtract(base_obj, diff_obj, base_scale):
    """
    Subtract diff_obj from orig_obj. If boolean modifier requires perturbation
    to not fail, orig obj is translated along offset_dir.
    """
    deselectAll()
    base_obj.select = True
    bpy.context.scene.objects.active = base_obj
    mod = get_boolean_modifier(base_obj, diff_obj, 'DIFFERENCE', base_scale)
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier=mod.name)
    bpy.context.scene.update()
    deselectAll()
    cleanObj(base_obj)

def get_boolean_modifier(base_obj, mod_obj, modifier_name, base_scale):
    mod = base_obj.modifiers.new(name=modifier_name,type='BOOLEAN')
    mod.object = mod_obj
    mod.solver = 'BMESH'
    mod.operation = modifier_name
    failures = 0
    base_size = mod_obj.scale.copy()
    base_size = base_size / base_scale
    test_factors = [0.05 * (i+1) for i in range(20)]
    test_factors.sort(key=lambda p: abs(p-base_scale))
    test_factors.pop(0)
    while not isSuccessfulModifier(base_obj) and len(test_factors)>0:
        base_obj.modifiers.remove(mod)
        new_factor = test_factors.pop(0)
        mod_obj.scale = base_size * new_factor
        #print(base_scale, new_factor, mod_obj.scale)
        bpy.context.scene.update()
        mod = base_obj.modifiers.new(name=modifier_name,type='BOOLEAN')
        mod.object = mod_obj
        mod.solver = 'BMESH'
        mod.operation = modifier_name
    return mod

def isSuccessfulModifier(mod_obj):
    if mod_obj.type=='MESH':
        mesh = mod_obj.data
        mesh_mod = mod_obj.to_mesh(bpy.context.scene,True, 'RENDER')
        if len(mesh.vertices) > 0:
            if len(mesh_mod.vertices)/len(mesh.vertices) < 0.3:
                bpy.data.meshes.remove(mesh_mod)
                return False
    else:
        raise Exception('Expected modifying object to have type MESH')
    bpy.data.meshes.remove(mesh_mod)
    return True

def cleanObj(base_obj):
    deselectAll()
    base_obj.select = True
    bpy.context.scene.objects.active = base_obj
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.remove_doubles()
    bpy.ops.mesh.normals_make_consistent()
    bpy.ops.object.editmode_toggle()
    deselectAll()

def duplicate(base_obj):
    new_obj = base_obj.copy()
    new_obj.data = base_obj.data.copy()
    bpy.context.scene.objects.link(new_obj)
    deselectAll()
    return new_obj

def hollow(solid_chamber, wall_thickness, shape):
    hollowing_obj = duplicate(solid_chamber)
    hollowing_obj.scale = [shape[0]*(1-wall_thickness), shape[1]*(1-wall_thickness), shape[2]*(1-wall_thickness)]
    subtract(solid_chamber, hollowing_obj, 1)
    bpy.context.scene.update()
    deselectAll()
    return hollowing_obj

def finalClean(base_obj):
    deselectAll()
    base_obj.select = True
    bpy.context.scene.objects.active = base_obj
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.remove_doubles()
    bpy.ops.mesh.normals_make_consistent()
    bpy.context.object.data.use_auto_smooth = True
    bpy.ops.object.editmode_toggle()

def createChamber(pos, scale, axis):
    tmp = mathutils.Matrix(axis)
    new_axes_q = tmp.to_quaternion()
    deselectAll()
    bpy.ops.mesh.primitive_ico_sphere_add(view_align=False, enter_editmode=False)
    chamber = bpy.context.active_object
    orig_axes_q = chamber.matrix_world.to_quaternion()
    quaternion_rot = orig_axes_q.rotation_difference(new_axes_q)
    chamber.scale = scale
    chamber.location += mathutils.Vector((pos[0],pos[1],pos[2]))
    chamber.rotation_quaternion = quaternion_rot
    bpy.context.scene.update()
    chamber.modifiers.new('MySubsurf','SUBSURF')
    for mod in chamber.modifiers:
        if mod.type == 'SUBSURF':
            mod.levels = 2
            bpy.ops.object.modifier_apply(apply_as='DATA', modifier=mod.name)
    bpy.ops.object.shade_smooth()
    deselectAll()
    return chamber

def saveObjWithGt(save_loc, sample_id, gt_centers, gt_shapes,gt_axes, gt_dict, gt_apertures):
    gt_dict.update({'centers':gt_centers, 'shapes':gt_shapes, 'axes':gt_axes, 'apertures':gt_apertures})
    with open(os.path.abspath(os.path.join(save_loc, sample_id + '_gt.json')), 'w') as F:
        F.write(json.dumps(gt_dict))
    print(os.path.abspath(os.path.join(save_loc, sample_id + '.obj')))
    bpy.ops.export_scene.obj(filepath=os.path.abspath(os.path.join(save_loc, sample_id + '.obj')), use_smooth_groups=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    create_watertight(gt_centers, gt_shapes, gt_axes)
    bpy.ops.export_scene.obj(filepath=os.path.abspath(os.path.join(save_loc, sample_id + '_wt.obj')), use_smooth_groups=True)

def create_watertight(centers, shapes, axes):
    base_foram = None
    for center, shape, axes in zip(centers, shapes, axes):
        if not base_foram:
            base_foram = createChamber(center[:3], shape, axes)
        else:
            new_chamber = createChamber(center[:3], shape, axes)
            union(base_foram, new_chamber, 1)
            removeObj(new_chamber)


def createForamUsingMorphospace(num_chamb, shape, TF, phi, beta, scaling, wall_thickness, aperture_scale, sample_id, save_loc):
    """
    Maintain two surface groups. The final resulting group and the diff group.
    The final resulting group has each chamber as a hollow chamber.
    The hollow chambers are created by diffs with the diff group which is no hollow chambers.
    Following 3D alg from "2D and 3D Numerical Models of the Growth of Foraminiferal Shells"
    Note we work in homogeneous coordinates
    """
    
    chamber_map = {}
    test_inner = None
    
    Oi = np.array([0,0,0])
    base_axes = [[1,0,0],[0,1,0],[0,0,1]]

    chamber_base = 'next_chamber'
    
    foram_obj_name = chamber_base + '0'
    new_chamber = createChamber(Oi[:3], shape, base_axes)
    chamber_map[foram_obj_name] = new_chamber
    
    diff_obj = hollow(new_chamber, wall_thickness, shape)

    gt_shapes = [shape]
    gt_centers = [Oi.tolist()]
    gt_apertures = []

    gt_axes = []
    axes = base_axes
    gt_axes.append(base_axes)

    shape.sort()
    shape[0], shape[1] = shape[1], shape[0]
    intended_aperture = np.array([0, shape[1], 0])
    Ai = getNextAperture(intended_aperture, new_chamber, shape) # Ai is in world coordinate
    gt_apertures.append(Ai.tolist())

    Uis = [Ai,Oi,np.array([0,0,1])] # Uis is in world coordinate

    for i in range(1,num_chamb):
        print('CHAMBER: {}'.format(i))
        new_chamber_name = chamber_base + str(i)

        #Vi is the direction vector (needs to be added to aperture of prev chamb)
        Vi, axes = getPerturbedOriginWithAxes(Uis, phi, beta, base_axes)
        print('ORIGIN VECTOR {}'.format(Vi))
        Vi = scaleGrowthVector(Vi, TF, shape[1])
        print('SCALE VECTOR {}'.format(Vi))
        #gt_apertures.append(Vi.tolist())

        print("Previous Aperture: {}".format(Ai))
        print("VECTOR: {}".format(Vi))
        Oi = Vi + Uis[0]
        print("NEW ORIGIN: {}".format( Oi))
        #print("ORIGIN OUTSIDE FORAM: {}".format(Oi))


        shape = np.multiply(shape, scaling[:3])

        gt_shapes.append(shape.tolist())
        gt_centers.append(Oi.tolist())
        gt_axes.append(axes.tolist())
        new_chamber = createChamber(Oi[:3], shape, axes)
        chamber_map[new_chamber_name] = new_chamber
        old_chamber_name = chamber_base + str(i-1)
        min_idx = np.argmin(shape)
        tmp_shape = shape[min_idx]*aperture_scale
        #print('APERTURE SHAPE: ', tmp_shape)
        aperture_to_subtract = createChamber(Uis[0][:3], [tmp_shape, tmp_shape,tmp_shape], axes)
        next_diff_obj = hollow(new_chamber, wall_thickness, shape)
        subtract(new_chamber, diff_obj, 1)
        subtract(chamber_map[old_chamber_name], aperture_to_subtract, aperture_scale)
        union(diff_obj, next_diff_obj, 1)

        # finding closest point from last aperture on current mesh
        #test_vec = Ai[:3] - ((gt_centers[-2][:3] - Ai[:3]) + (gt_centers[-1][:3] - Ai[:3]))
        #Ai = getNextAperture(test_vec, new_chamber_name, shape) # Ai is in world coordinate
        Ai = getNextAperture(Ai, new_chamber, shape) # Ai is in world coordinate
        gt_apertures.append(Ai.tolist())

        removeObj(next_diff_obj)
        removeObj(aperture_to_subtract)

        Uis.pop()
        Uis.insert(0, Ai)

    old_chamber_name = chamber_base + str(num_chamb-1)
    min_idx = np.argmin(shape)
    tmp_shape = shape[min_idx]*aperture_scale
    aperture_to_subtract = createChamber(Uis[0][:3], [tmp_shape, tmp_shape,tmp_shape], axes)
    subtract(chamber_map[old_chamber_name], aperture_to_subtract, aperture_scale)
    removeObj(aperture_to_subtract)

    removeObj(diff_obj)
    foram_obj = joinChambers(chamber_map)
    finalClean(foram_obj)
    input_dict = {'number of chambers':num_chamb, 'shape':list(shape), 'TF':TF, 'phi':phi, 'beta':beta, 'scaling':list(scaling), 'wall thickness':wall_thickness, 'aperture scale': aperture_scale}
    saveObjWithGt(save_loc, sample_id, gt_centers, gt_shapes,gt_axes, input_dict, gt_apertures)

def getNextAperture(previous_aperture, chamber, shape):
    to_local_coord = chamber.matrix_world.inverted()
    local_aperture = to_local_coord * mathutils.Vector(previous_aperture)
    (hit, loc, norm, fi) = chamber.closest_point_on_mesh(local_aperture[:3])
    to_world_coord = chamber.matrix_world
    world_loc = to_world_coord * loc
    return np.array([world_loc[0],world_loc[1],world_loc[2]])

def getPerturbedOriginWithAxes(Uis, phi, beta, base_axes):
    v1 = unitVector(Uis[0]-Uis[1])
    v2 = unitVector(Uis[2]-Uis[1])
    vec_scale = homogeneousNorm(Uis[0]-Uis[1])
    #v1 = [round(i, 6) for i in v1[:3]]
    #v2 = [round(i, 6) for i in v2[:3]]
    #beta = beta + (2 * math.pi * random.uniform(-0.014, 0.014))
    #pos_neg = 1 if random.random() < 0.5 else -1
    #phi = phi + (pos_neg * math.acos(random.uniform(0.996,1)))
    cross = getNormedCrossProduct(v2, v1)
    first_rot = getRotationMat(cross, phi)
    second_rot = getRotationMat(v1, beta)
    origin = np.matmul(second_rot, np.matmul(first_rot,v1))
    if abs(homogeneousNorm(origin) -1) > 0.00000001:
        raise Exception('Origin not a unit vector! Vector changed from {} to {}'.format(v1, origin))
    #origin = origin * vec_scale
    axes = np.matmul(second_rot, np.matmul(first_rot, base_axes))
    return origin, axes

def scaleGrowthVector(Vi, TF, cham_radius):
    ret_Vi = Vi * TF * cham_radius
    #ret_Vi = Vi * TF
    #ret_Vi[3] = 1
    return ret_Vi

def getTransformUpdates(translate, curr_vec, transformed_vec, btrw, wctfc):
    transl_mat = np.array([[1,0,0,translate[0]],[0,1,0,translate[1]],[0,0,1,translate[2]],[0,0,0,1]])
    if np.linalg.matrix_rank([curr_vec, transformed_vec])==1:
        if np.linalg.norm(curr_vec + transformed_vec) == 0:
            transl_mat = np.array([[1,0,0,-translate[0]],[0,1,0,-translate[1]],[0,0,1,-translate[2]],[0,0,0,1]])
        next_transform = transl_mat
    else:
        cross = getNormedCrossProduct(curr_vec, transformed_vec)
        #note cw is away from transformed_vec. ccw is towards
        angle = angleBetween(curr_vec[:3], transformed_vec[:3])
        next_rotation = homogeneousNorm(transformed_vec[:3]) * getRotationMat(cross, angle)
        next_transform = np.matmul(transl_mat,next_rotation)

    # btrw = inv(A) x inv(B) x inv(C) x inv(D)
    next_btrw = np.matmul(btrw, np.linalg.inv(next_transform))
    # wctfc = D x C x B x A
    next_wctfc = np.matmul(next_transform, wctfc)
    return next_transform, next_btrw, next_wctfc

def getNormedCrossProduct(v1,v2):
    cross_unnorm = np.cross(v1[:3],v2[:3])
    if homogeneousNorm(cross_unnorm) == 0:
            #to 3 dim
            return np.zeros(3)
            #return np.zeros(4)
    cross = cross_unnorm / homogeneousNorm(cross_unnorm)
    #to 3 dim
    #return np.append(cross,[1])
    return cross

def homogeneousNorm(v):
    return np.linalg.norm(v[:3])

def unitVector(v):
    return v / homogeneousNorm(v)

def angleBetween(v1,v2):
    """
    Returns angle in radians.
    """
    v1_u = unitVector(v1)
    v2_u = unitVector(v2)
    return np.arccos(np.dot(v1_u, v2_u))

def getRotationMat(v, angle):
    """
    return:  cos(angle)*I + sin(angle)*cross_product_matrix(u) + (1-cos(angle))*outer_product(u,u)
    """
    K = np.array([[0, -v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    rotation = np.identity(3) + (np.sin(angle) * K) + (np.matmul(K,K)*(1-np.cos(angle)))
    #to 3 dim
    rot_mat = rotation
    #print('ROTATION MATRIX')
    #print(rot_mat)
    #rot_mat = np.vstack((np.hstack((rotation,[[0],[0],[0]])),[0,0,0,1]))
    return rot_mat

def getSyntheticFolder(root_folder):
    data_path = os.path.abspath(os.path.join(root_folder,'data'))
    if os.path.exists(os.path.abspath(os.path.join(data_path,'synthetic'))):
        return os.path.abspath(os.path.join(data_path,'synthetic'))
    return None

def getSampleId(data_folder):
    ct_file = os.path.abspath(os.path.join(data_folder,'counter.txt'))
    try:
        with open( ct_file, 'r' ) as fle:
            sample_id = int( fle.readline() ) + 1
    except FileNotFoundError:
        sample_id = 0

    with open( ct_file, 'w' ) as fle:
        fle.write( str(sample_id) )
    return str(sample_id)
    
def updateSampleId(data_folder, update_val):
    ct_file = os.path.abspath(os.path.join(data_folder,'counter.txt'))
    with open( ct_file, 'w' ) as fle:
        fle.write( str(update_val) )
    return

def generate_from_cmd(arglist):
    arg_dict = json.loads(arglist[0])
    dims = arg_dict['dims']
    increments = arg_dict['increments']
    data_folder = getSyntheticFolder(arg_dict['root_folder'])
    TF = arg_dict['TF']
    phi = arg_dict['phi']
    num_chamb = arg_dict['num_cham']
    beta = arg_dict['beta']
    shape = arg_dict['shape']
    #shape = [arg_dict['shape_base']]*3
    #scaling = (arg_dict['growth'], arg_dict['growth'], arg_dict['growth'])
    scaling = arg_dict['growth']
    final_size = arg_dict['final_size']
    wall_thickness = 0.05
    idcs = [i for i,val in enumerate(dims) if val]
    aperture_scale = arg_dict['aperture_scale']
    #for ui
    #shape_base = ((final_size * 0.5) / (scaling**(num_chamb-1)))/1000
    #for cmdline
    #shape_base = arg_dict['shape_base']
    if len(idcs) == 0:
        #shape = [shape_base]*3
        sample_id = getSampleId(data_folder)
        createForamUsingMorphospace(num_chamb, shape, TF, radians(phi), radians(beta), (scaling,scaling,scaling), wall_thickness, aperture_scale, sample_id, data_folder)
    else:
        argument_list = [num_chamb, scaling, TF, phi, beta]
        grid_folder = os.path.join(data_folder, 'grid')
        grid_id = getSampleId(grid_folder)
        sample_ids = []
        base_sample_id = int(getSampleId(data_folder))
        all_dim_vals = []
        dim_shape = [0]*len(idcs)
        for i,idx in enumerate(idcs):
            next_dim = []
            next_val = argument_list[idx]
            while next_val - dims[idx] <= 0.0000001:
                next_dim.append(next_val)
                next_val += increments[idx]
                dim_shape[i] += 1
            all_dim_vals.append(next_dim)
        sample_prod = 1
        for mult in dim_shape:
            sample_prod = sample_prod * mult
        sample_ids = list(range(base_sample_id, base_sample_id + sample_prod))
        updateSampleId(data_folder, base_sample_id + sample_prod+1)
        saveGridFile(grid_folder,grid_id,sample_ids, dim_shape, arg_dict)
        for sample_id, (reset_ct, dim_vals) in zip(sample_ids, enumerate(itertools.product(*all_dim_vals))):
            print(sample_id, reset_ct, dim_vals)
            for i, val in enumerate(dim_vals):
                argument_list[idcs[i]] = val
            #shape = [shape_base]*3
            #print(argument_list)
            createForamUsingMorphospace(argument_list[0], shape, argument_list[2], radians(argument_list[3]), radians(argument_list[4]), (argument_list[1],argument_list[1],argument_list[1]), wall_thickness, aperture_scale, str(sample_id), data_folder)
            reset_blend()
        #saveGridFile(grid_folder,grid_id,sample_ids, dim_shape, arg_dict)

def saveGridFile(grid_folder,grid_id,sample_ids, dim_shape, arg_dict):
    arg_dict['dim_shape'] = dim_shape
    arg_dict['samples'] = sample_ids
    with open(os.path.abspath(os.path.join(grid_folder, grid_id + '.json')), 'w') as f:
        f.write(json.dumps(arg_dict))

def reset_blend():
    #bpy.ops.wm.read_factory_settings()

    for scene in bpy.data.scenes:
        for obj in scene.objects:
            scene.objects.unlink(obj)

    # only worry about data in the startup scene
    for bpy_data_iter in (
            bpy.data.objects,
            bpy.data.meshes,
            bpy.data.lamps,
            bpy.data.cameras,
    ):
        for id_data in bpy_data_iter:
            bpy_data_iter.remove(id_data)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=True)

#when running from ui
#print('here')
argv = sys.argv
arglist = argv[argv.index("--") + 1:]

#when running testing from cmd line
#arglist = ['{"root_folder": "E:\\\\PhD\\\\Forams\\\\synthetic_foram_model", "num_cham": 11, "shape_base": 0.05668934240362812, "TF": 0.6, "phi": 0.0, "beta": 0.0, "growth": 1.05, "final_size": 125, "dims": [null], "increments": [null]}']
#arglist = ['{"root_folder": "/home/turner/Forams/synthetic/synthetic_foram_model", "num_cham": 12, "shape_base": 0.04037639572246145, "TF": 0.85, "phi": 65.0, "beta": 10.0, "growth": 1.2, "final_size": 600, "dims": [null, null, 0.95, null, null], "increments": [null, null, 0.05, null, null], "aperture_scale": 0.25}']

print(arglist)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
generate_from_cmd(arglist)
