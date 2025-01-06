import bpy
import os
import numpy as np
import json
import sys
import random
import csv

# Load an object

# Attach the material
# Set keyframes
# Render
# Save to file
# https://christianjmills.com/Create-a-Shape-Key-Motion-Graphic-With-the-Blender-Python-API/
# rotations
# http://extremelearning.com.au/evenly-distributing-points-on-a-sphere/

def renderForam(file_loc, sample_id, material_name, rotations, save_loc):
    if os.path.isfile(os.path.join(save_loc, '{}_rotations.csv'.format(material_name))):
        return
    bpy.ops.import_scene.obj(filepath=file_loc)
    bpy.context.scene.render.resolution_percentage=20
    obj_list = [o for o in bpy.context.selected_objects]
    if len(obj_list) > 1:
        print('ERROR TOO MANY OBJECTS PER OBJ FILE')
    foram = obj_list[0]
    centerAndSmoothObj(foram)

    mat = bpy.data.materials.get(material_name)
    nodes = mat.node_tree.nodes
    mat_output = nodes.get('Material Output')
    if foram.data.materials:
        foram.data.materials[0] = mat
    else:
        foram.data.materials.append(mat)
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.compression = 0

    bpy.ops.object.select_all( action = 'DESELECT' )
    foram.select_set(True)
    bpy.context.view_layer.objects.active = foram
    foram.rotation_mode = 'QUATERNION'
    rotations = getRotationAnimation(rotations)
    for i,orientation in enumerate(rotations):
        foram.rotation_quaternion = orientation
        bpy.context.scene.render.filepath = os.path.join(save_loc, '{}_{}.PNG'.format(material_name, i))
        bpy.ops.render.render(write_still=True)
    with open(os.path.join(save_loc, '{}_rotations.csv'.format(material_name)),'w') as f:
            write_csv = csv.writer(f)
            write_csv.writerow(['q_x', 'q_y', 'q_z', 'q_w'])
            write_csv.writerows([[q[0], q[1], q[2], q[3]] for q in rotations])
    bpy.ops.object.delete()

def getRandomQuaternion():
    s = random.random()
    s1 = np.sqrt(1- s)
    s2 = np.sqrt(s)
    t1 = 2*np.pi * random.random()
    t2 = 2*np.pi * random.random()
    w = np.cos(t2) * s2
    x = np.sin(t1) * s1 
    y = np.cos(t1) * s1 
    z = np.sin(t2) * s2 
    return np.array([x, y, z, w])

def hamilton(p, q):
    w = p[3]*q[3] - p[0]*q[0] - p[1]*q[1] - p[2]*q[2]
    i = p[3]*q[0] + p[0]*q[3] + p[1]*q[2] - p[2]*q[1]
    j = p[3]*q[1] - p[0]*q[2] + p[1]*q[3] + p[2]*q[0]
    k = p[3]*q[2] + p[0]*q[1] - p[1]*q[0] + p[2]*q[3]
    return np.array([i,j,k,w])

def getQuatToLoc(x,y,z):
    q = [0,0,0,0]
    v1 = [1,0,0]
    v2 = [x,y,z]
    v2 = v2 / np.sqrt(np.dot(v2,v2))
    w = np.dot(v1,v2)
    if w < -0.99999:
        v3 = [0,1,0]
        axis = np.cross(v3, v2)
        rad = np.pi * 0.5
        scale = np.sin(rad)
        q[:3] = scale*axis
        q[3] = np.cos(rad)
    elif w > 0.99999:
        q = [0,0,0,1]
    else:
        r_vec = np.cross(v1,v2)
        q[:3] = list(r_vec)
        q[3] = np.sqrt(np.dot(v1,v1)*np.dot(v2,v2)) + w
    q = q / np.sqrt(np.dot(q,q))
    return q

def getRotationAnimation(num_orientations):
    i = np.arange(0, num_orientations, dtype=float) + 0.5
    phi = np.arccos(1 - 2*i/num_orientations)
    goldenRatio = (1 + 5**0.5)/2
    theta = 2 * np.pi * i / goldenRatio
    xs, ys, zs = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
    q_rots = []
    q_rand = getRandomQuaternion()
    for x,y,z in zip(xs,ys,zs):
        q = getQuatToLoc(x,y,z)
        rot_q = hamilton(q_rand,q)
        q_rots.append(rot_q)
    return q_rots
    

def renderAllForams(sample_ids, rotations, material_name, root_folder):
    data_folder = os.path.join(root_folder, 'data', 'synthetic')
    for sample_id in sample_ids:
        save_loc = os.path.join(root_folder, 'data', 'renders', '{}'.format(sample_id))
        next_file = os.path.join(data_folder, '{}.obj'.format(sample_id))
        print('Rendering to: {}'.format(save_loc))
        os.makedirs(save_loc, exist_ok=True)
        renderForam(next_file, sample_id, material_name, rotations, save_loc)

def centerAndSmoothObj(foram):
    bpy.ops.object.select_all( action = 'DESELECT' )
    foram.select_set(True)
    bpy.context.view_layer.objects.active = foram
    #Scaled to 3x the um size for data. Acutal should be 0.000001
    foram.scale = (0.000003, 0.000003, 0.000003)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY',center='BOUNDS')
    foram.location = (0,0,0)
    #o.modifiers.new('MySubsurf','SUBSURF')
    #for mod in bpy.context.active_object.modifiers:
    #    if mod.type == 'SUBSURF':
    #        mod.levels = 2
    #        bpy.ops.object.modifier_apply(modifier=mod.name)
    #bpy.ops.object.shade_smooth()
    bpy.ops.object.select_all( action = 'DESELECT' )

def setupGPU():
    bpy.data.scenes[0].render.engine = "CYCLES"
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "OPTIX"
    bpy.context.scene.cycles.device = "GPU"
    use_gpu = [False,True]
    cuda_gpu_idx = 0
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    for d in bpy.context.preferences.addons['cycles'].preferences.devices:
        print(d["name"], d["id"])
        if "OptiX" in d["id"]:
            if use_gpu[cuda_gpu_idx]:
                d["use"] = 1
            else:
                d["use"] = 0
            cuda_gpu_idx += 1
        else:
            d["use"] = 0
        print("We are using " if d["use"] else "Not using: ", d["name"])
    #for d in opencl:
    #    d["use"] = 0
    #    print("We are using " if d["use"] else "Not using: ", d["name"])

def render(arglist):
    arg_dict = json.loads(arglist[0])
    root_folder = arg_dict['root_folder']
    sample_ids = arg_dict['samples']
    rotations = int(arg_dict['rotations'])
    material_name = arg_dict['material_name']
    setupGPU()
    renderAllForams(sample_ids, rotations, material_name, root_folder)


argv = sys.argv
arglist = argv[argv.index("--") + 1:]
#args = {"samples": ["test_render_obj"],"rotations": 8, "material_name":"Forabot"}
#arglist = json.dumps(args)
######For later reference. Don't want to select all because we don't want to delete our materials...
######bpy.ops.object.select_all(action='SELECT')
######bpy.ops.object.delete(use_global=False)
render(arglist)

