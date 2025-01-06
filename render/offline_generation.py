import subprocess
import os
import json
import glob

def subsetSamples(all_samples, chunk_size):
    for i in range(0, len(all_samples), chunk_size):
        yield all_samples[i:i+chunk_size]

def renderBlender(blender_render_location, list_of_objs, num_rotations, root_folder, used_gpu):
        blend_file_loc = os.path.abspath(os.path.join(os.getcwd(),'forabot_render_final.blend'))
        chunk_size = 500
        for subset in subsetSamples(list_of_objs, chunk_size):
            renderBlenderMaterial(blender_render_location, subset, num_rotations, blend_file_loc, 'Forabot', root_folder, used_gpu)
        #blend_file_loc = os.path.abspath(os.path.join(os.getcwd(),'microfluidics_render_final.blend'))  
        #renderBlenderMaterial(blender_render_location,list_of_objs, num_rotations, blend_file_loc, 'Microfluidics', root_folder, used_gpu)
        
def renderBlenderMaterial(blender_render_location, list_of_objs, num_rotations, blend_file_loc, material_name, root_folder, used_gpu):
    render_script = os.path.abspath(os.path.join(os.getcwd(),'generate_renders_fixed_gpu.py'))
    arg_dict = {}
    arg_dict['root_folder'] = root_folder
    arg_dict['samples'] = list_of_objs
    arg_dict['rotations'] = num_rotations
    arg_dict['material_name'] = material_name
    arg_dict['used_gpu'] = used_gpu
    arg_list = ['--', json.dumps(arg_dict)]

    print([blender_render_location, "-b", blend_file_loc, "--background", "--python", render_script] + arg_list)
    isShell = not os.name == 'posix'
    process = subprocess.run([blender_render_location, "-b", blend_file_loc, "--background", "--python", render_script] + arg_list, shell=isShell)
    
def callBlender(blender_loc, num_cham,shape,TF,phi,beta,growth,final_size,aperture_scale,dims=[None], increments=[None]):
        custom_loc = os.path.abspath(os.path.join(os.getcwd(),'morphospace.py'))
        arg_dict = {}
        arg_dict['root_folder'] = os.path.abspath('..')
        arg_dict['num_cham'] = num_cham
        arg_dict['shape'] = shape
        arg_dict['TF'] = TF
        arg_dict['phi'] = phi
        arg_dict['beta'] = beta
        arg_dict['growth'] = growth
        arg_dict['final_size'] = final_size
        arg_dict['dims'] = dims
        arg_dict['increments'] = increments
        arg_dict['aperture_scale'] = aperture_scale
        arg_list = ['--', json.dumps(arg_dict)]

        print([blender_loc, "--background","--python",custom_loc] + arg_list)
        isShell = not os.name == 'posix'
        process = subprocess.run([blender_loc, "--background", "--python",custom_loc] + arg_list, shell=isShell)
        
def getSamples(grid_file=None):
    if grid_file is None:
        initialdir = "../data/synthetic/grid/*.json"
        grid_loc = os.path.abspath(initialdir)
        list_of_files = glob.glob(grid_loc)
        grid_file_loc = max(list_of_files, key=os.path.getctime)
    else:
        grid_file_loc = os.path.abspath("../data/synthetic/grid/{}.json".format(grid_file))
    with open(grid_file_loc,'r') as f:
        arg_dict = json.load(f)
    sample_ids = arg_dict['samples']
    return sample_ids
    
def getHeterohelix():
    num_cham = 16
    shape = [10.0,10.0,10.0]
    growth = 1.1
    TF = -0.4
    phi = 55
    beta = 0.0
    #num chambers, growth rate, Tf, Phi, Beta
    dims = [None, 1.15, -0.1, 90, None]
    increments = [None, 0.01, 0.01, 1, None]
    aperture_scale = 0.45
    return num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale

def getGuembelitriella():
    num_cham = 16
    shape = [10.0,10.0,10.0]
    growth = 1.1
    TF = -0.24
    phi = 75
    beta = 61.0
    #num chambers, growth rate, Tf, Phi, Beta
    dims = [None, 1.3, -0.12, 111, 81]
    increments = [None, 0.02, 0.02, 3, 2]
    aperture_scale = 0.45
    return num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale

def getPachyderma1():
    num_cham = 16
    shape = [10.0,10.0,10.0]
    growth = 1.1
    TF = 0.45
    phi = 110
    beta = -30.0
    #num chambers, growth rate, Tf, Phi, Beta
    dims = [None, 1.15, 0.6, 125, -10]
    increments = [None, 0.01, 0.02, 2, 2]
    aperture_scale = 0.45
    return num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale
    
def getPachyderma2():
    num_cham = 16
    shape = [10.0,10.0,10.0]
    growth = 1.1
    TF = 0.45
    phi = 125
    beta = -20
    #num chambers, growth rate, Tf, Phi, Beta
    dims = [None, 1.15, 0.6, 145, -5]
    increments = [None, 0.01, 0.02, 2, 2]
    aperture_scale = 0.45
    return num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale


def getDutertrei1():
    num_cham = 16
    shape = [10.0,10.0,10.0]
    growth = 1.1
    TF = 0.45
    phi = 110
    beta = 10.0
    #num chambers, growth rate, Tf, Phi, Beta
    dims = [None, 1.15, 0.6, 125, 30]
    increments = [None, 0.02, 0.02, 2, 2]
    aperture_scale = 0.45
    return num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale
    
def getDutertrei2():
    num_cham = 16
    shape = [10.0,10.0,10.0]
    growth = 1.1
    TF = 0.45
    phi = 125
    beta = 5
    #num chambers, growth rate, Tf, Phi, Beta
    dims = [None, 1.15, 0.6, 145, 20]
    increments = [None, 0.02, 0.02, 2, 2]
    aperture_scale = 0.45
    return num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale

def getDutertrei3():
    num_cham = 16
    shape = [10.0,10.0,10.0]
    growth = 1.11
    TF = 0.45
    phi = 110
    beta = 10.0
    #num chambers, growth rate, Tf, Phi, Beta
    dims = [None, 1.15, 0.6, 125, 30]
    increments = [None, 0.02, 0.02, 2, 2]
    aperture_scale = 0.45
    return num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale

def getDutertrei4():
    num_cham = 16
    shape = [10.0,10.0,10.0]
    growth = 1.11
    TF = 0.45
    phi = 125
    beta = 5
    #num chambers, growth rate, Tf, Phi, Beta
    dims = [None, 1.15, 0.6, 145, 20]
    increments = [None, 0.02, 0.02, 2, 2]
    aperture_scale = 0.45
    return num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale

def getPelagica():
    num_cham = 16
    shape = [10.0,10.0,10.0]
    growth = 1.1
    TF = 0.45
    phi = 130
    beta = -3.0
    #num chambers, growth rate, Tf, Phi, Beta
    dims = [None, 1.15, 0.6, 140, 3]
    increments = [None, 0.01, 0.01, 1, 1]
    aperture_scale = 0.45
    return num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale

def getUvula():
    num_cham = 16
    shape = [10.0,10.0,10.0]
    growth = 1.05
    TF = 0.1
    phi = 0
    beta = -5
    #num chambers, growth rate, Tf, Phi, Beta
    dims = [None, 1.1, 0.6, 15, 5]
    increments = [None, 0.01, 0.05, 2, 1]
    aperture_scale = 0.45
    return num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale

def getBulloides():
    num_cham = 16
    shape = [10.0,10.0,10.0]
    growth = 1.1
    TF = -0.8
    phi = 70
    beta = 146
    #num chambers, growth rate, Tf, Phi, Beta
    dims = [None, 1.2, -0.6, 75, 161]
    increments = [None, 0.01, 0.02, 1, 1]
    aperture_scale = 0.45
    return num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale
        
if __name__ == '__main__':

    root_folder = '/home/trichmo/Forams/2022/synthetic_foram_model/'
    blender_generation_location = '/usr/local/blender/blender-2.79b-linux-glibc219-x86_64/blender'
    blender_render_location = '/usr/local/blender/blender-3.2.0-linux-x64/blender'

    num_renders = 6
    final_size = 1 #TODO: remove final_size from morphospace expected input. Not used.
    
    #num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale = getHeterohelix()
    #num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale = getGuembelitriella()
    #num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale = getPachyderma1()
    #num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale = getPachyderma2()
    #num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale = getDutertrei1()
    #num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale = getDutertrei2()
    #num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale = getDutertrei3()
    #num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale = getDutertrei4()
    #num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale = getPelagica()
    #num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale = getUvula()
    #num_cham, shape, growth, TF, phi, beta, dims, increments, aperture_scale = getBulloides()
    
    #callBlender(blender_generation_location, num_cham,shape,TF,phi,beta,growth,final_size,aperture_scale,dims, increments)
    
    grid_loc = "25"
    used_gpu = 0

    sample_ids = getSamples(grid_loc)
    renderBlender(blender_render_location, sample_ids, num_renders, root_folder, used_gpu)

