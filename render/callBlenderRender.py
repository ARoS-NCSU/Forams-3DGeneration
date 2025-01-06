import subprocess
import os
import json

def callBlenderRender(self,obj_locations, rotations, material_name):
    if material_name == 'Forabot':
        blend_file_loc = custom_loc = os.path.abspath(os.path.join(os.getcwd(), '..','forabot_render.blend1'))
    elif material_name == 'Microfluidics':
        blend_file_loc = custom_loc = os.path.abspath(os.path.join(os.getcwd(),'..','microfluidics.blend'))
    else
        raise Exception('Unknown Material: {}'.format(material_name))    
    blender_loc = ''
    custom_loc = os.path.abspath(os.path.join(os.getcwd(),'generate_renders.py'))
    arg_dict = {}
    arg_dict['root_folder'] = os.path.abspath('..')
    arg_dict['samples'] = obj_locations
    arg_dict['rotations'] = 8
    arg_dict['material_name'] = 'Forabot'
    arg_list = ['--', json.dumps(arg_dict)]

    print([blender_loc, "-b", blend_file_loc, "--background", "--python",custom_loc] + arg_list)
    isShell = not os.name == 'posix'
    process = subprocess.run([blender_loc, "-b", blend_file_loc, "--background", "--python",custom_loc] + arg_list, shell=isShell)
    
    
if __name__ == '__main__':
    obj_locations = []
    rotations = 8
    material_name = 'Forabot'
    callBlenderRender(self,obj_locations, rotations, material_name)
