from tkinter import *
from tkinter import ttk, messagebox, filedialog
import subprocess
import os
import json
import glob

class UserInterface:
    def __init__(self):
        self.root = Tk()
        self.root.protocol("WM_DELETE_WINDOW", self._onClose)
        self.root.title("Forams Morphology")
        self.createBlenderFileWindow()
        self.createBlenderRenderFileWindow()
        self._setupWindow()

    def createBlenderFileWindow(self):
        if os.name == 'posix':
            initialdir = "/opt/blender-2.78/"
            filetypes = [("Blender executable","*")]
        else:
            initialdir = "C:\\Program Files\\Blender Foundation"
            filetypes = [("Blender executable","*.exe")]
        self.blender_location =  filedialog.askopenfilename(parent = self.root, title='Select Blender 2.78 Executable', initialdir = initialdir,filetypes = filetypes)
        
    def createBlenderRenderFileWindow(self):
        if os.name == 'posix':
            initialdir = "/snap/bin/"
            filetypes = [("Blender executable","*")]
        else:
            initialdir = "C:\\Program Files\\Blender Foundation"
            filetypes = [("Blender executable","*.exe")]
        self.blender_render_location =  filedialog.askopenfilename(parent = self.root, title='Select Blender 3.1 Executable', initialdir = initialdir,filetypes = filetypes)

    def _onClose(self):
        self.root.destroy()

    def _setupWindow(self):
        #Selection for number of chambers: int values 1 to 50
        #Selection for total outer size of Chambers: 50um to 1mm custom list
        #Need to decide on chamber shape change
        #Selection for chamber growth direction length: float -1 to 1 by 0.05
        #Input for angle: -180 to 180
        #Input for beta angle -180 to 180

        planar_angle_cmd = self.root.register(self._isPlanarAngle)
        beta_angle_cmd = self.root.register(self._isBetaAngle)
        cham_cmd = self.root.register(self._isChamberCt)
        rot_cmd = self.root.register(self._isRotationCt)
        size_cmd = self.root.register(self._isSize)
        dir_cmd = self.root.register(self._isDirection)
        growth_cmd = self.root.register(self._isGrowth)
        aperture_cmd = self.root.register(self._isAperture)

        size_tuple = (50,100,125,150,175,200,225,250,275,300,325,350,375,400,
                            425,450,475,500,525,550,575,600,700,800,900,1000)

        self.value_label_var = StringVar()
        self.value_label_var.set('Values')
        self.should_render_forabot_val = IntVar()
        self.should_render_microfluidics_val = IntVar()
        
        self.value_label = ttk.Label(self.root, textvariable=self.value_label_var)     
        self.num_chamber_label = ttk.Label(self.root,
                                text="Number of Chambers")
        self.size_label = ttk.Label(self.root,
                                text="Shape of first foram chamber - x,y,z (um)")
        self.growth_label = ttk.Label(self.root,
                                text="Chamber Growth Rate")
        self.direction_label = ttk.Label(self.root,
                                text="Distance Between Chambers")
        self.planar_angle_label = ttk.Label(self.root,
                                text="Angle in plane")
        self.z_angle_label = ttk.Label(self.root,
                                text="Angle towards Z-axis")
        self.aperture_label = ttk.Label(self.root, text="Aperture scale")
        self.should_render_forabot_label = ttk.Label(self.root, text="Render Forabot")
        self.should_render_microfluidics_label = ttk.Label(self.root, text="Render Microfluidics")
        self.num_rotations_label = ttk.Label(self.root, text="Number of Render Rotations")

        self.num_chambers = ttk.Spinbox(self.root, from_=1, to=50,
                                validate='all', validatecommand=(cham_cmd,'%P'))
        self.sizex = ttk.Spinbox(self.root, from_=1, to=30,
                                validate='all', validatecommand=(size_cmd,'%P'))
        self.sizey = ttk.Spinbox(self.root, from_=1, to=30,
                                validate='all', validatecommand=(size_cmd,'%P'))
        self.sizez = ttk.Spinbox(self.root, from_=1, to=30,
                                validate='all', validatecommand=(size_cmd,'%P'))
        self.growth = ttk.Spinbox(self.root, from_=1, to=2, validate='all',
                                validatecommand=(growth_cmd,'%P'), increment=0.05)
        self.direction = ttk.Spinbox(self.root, from_=-1, to=1, increment=0.05,
                                validate='all', validatecommand=(dir_cmd, '%P'))
        self.planar_angle = ttk.Entry(self.root, validate='all',
                                validatecommand=(planar_angle_cmd, '%P', '%i'))
        self.z_angle = ttk.Entry(self.root, validate='all',
                                validatecommand=(beta_angle_cmd, '%P', '%i'))
        self.aperture_scale = ttk.Spinbox(self.root, from_=0.05, to=1, increment=0.05,
                                validate='all', validatecommand=(aperture_cmd,'%P'))
        self.should_render_forabot = Checkbutton(self.root, var=self.should_render_forabot_val)
        self.should_render_microfluidics = Checkbutton(self.root, var=self.should_render_microfluidics_val)
        self.num_rotations = ttk.Spinbox(self.root, from_=1, to=30, validate='all', validatecommand=(rot_cmd, '%P'))
        self.submit_button = ttk.Button(self.root, text="Submit",
                                command=self._generateForam)

        self.value_label.grid(row=0, column=1)
        self.num_chamber_label.grid(row=1, column=0)
        self.size_label.grid(row=2, column=0)
        self.growth_label.grid(row=3, column=0)
        self.direction_label.grid(row=4, column=0)
        self.planar_angle_label.grid(row=5, column=0)
        self.z_angle_label.grid(row=6, column=0)
        self.aperture_label.grid(row=7, column=0)
        self.should_render_forabot_label.grid(row=8, column=0)
        self.should_render_microfluidics_label.grid(row=8, column=2)
        self.num_rotations_label.grid(row=9, column=0)

        self.num_chambers.grid(row=1, column=1)
        self.sizex.grid(row=2, column=1)
        self.sizey.grid(row=2, column=2)
        self.sizez.grid(row=2, column=3)
        self.growth.grid(row=3, column=1)
        self.direction.grid(row=4, column=1)
        self.planar_angle.grid(row=5, column=1)
        self.z_angle.grid(row=6, column=1)
        self.aperture_scale.grid(row=7, column=1)
        self.should_render_forabot.grid(row=8, column=1)
        self.should_render_microfluidics.grid(row=8, column=3)
        self.num_rotations.grid(row=9, column=1)
        self.submit_button.grid(row=10, column=1)

        self.num_chambers.focus_force()

    def _isBetaAngle(self, val, idx_s):
        idx = int(idx_s)
        if idx == 0:
            return self._isInRange(val, 180, -180, float, True) or val == '-'
        return self._isInRange(val, 180, -180, float, True)
     
    def _isPlanarAngle(self, val, idx_s):
        idx = int(idx_s)
        return self._isInRange(val, 180, 0, float, True)

    def _isChamberCt(self, val):
        return self._isInRange(val, 51, 0, int)
        
    def _isRotationCt(self, val):
        return self._isInRange(val, 31, 0, int)

    def _isSize(self,val):
        return self._isInRange(val, 31, 0, int)

    def _isDirection(self,val):
        return self._isInRange(val, 1, -1, float, True)

    def _isAperture(self, val):
        return self._isInRange(val, 1,0.05, float)

    def _isGrowth(self,val):
        return self._isInRange(val, 2.0000001, 0.999999, float)

    def _isInRange(self, val,high,low,dtype, inclusive=False):
        try:
            a = dtype(val)
            if inclusive:
                return a>=low and a<=high
            else:
                return a>low and a<high
        except ValueError:
            return not val

    def _generateForam(self):
        #final_size = int(self.size.get())
        final_size=0
        shape = [float(self.sizex.get()), float(self.sizey.get()), float(self.sizez.get())]
        num_cham = int(self.num_chambers.get())
        growth = float(self.growth.get())
        TF = float(self.direction.get())
        phi = float(self.planar_angle.get())
        beta = float(self.z_angle.get())
        aperture_scale = float(self.aperture_scale.get())
        #shape_base = ((final_size * 0.5) / (growth**(num_cham-1)))/1000
        self._callBlender(num_cham,shape,TF,phi,beta,growth,final_size,aperture_scale)
        #add in _renderBlender call for the last obj
        self._runSingleVisualization()

    def _runSingleVisualization(self):
        initialdir = "../data/synthetic/*.obj"
        sample_loc = os.path.abspath(initialdir)
        list_of_files = glob.glob(sample_loc)
        sample_file_loc = max(list_of_files, key=os.path.getctime)
        sample_filename = os.path.basename(sample_file_loc)
        sample_id = sample_filename.split('_')[0]
        print(sample_id)
        self._renderBlender([sample_id], self.num_rotations.get(), os.path.abspath('..'))
        custom_loc = os.path.abspath('visualizeMorphospace.py')
        arg_list = ['--', sample_id]
        isShell = not os.name == 'posix'
        process = subprocess.run([self.blender_location, "--python",custom_loc] + arg_list, shell=isShell)


    def _callBlender(self,num_cham,shape,TF,phi,beta,growth,final_size,aperture_scale,dims=[None], increments=[None]):
        blender_loc = self.blender_location
        custom_loc = os.path.abspath(os.path.join(os.getcwd(),'morphospace.py'))
        #run_str = '"{}" --background --python "{}" -- '.format(blender_loc,custom_loc)
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
        #arg_list = '\' {'
        #for key in arg_dict.keys():
        #    run_str = run_str + '"{}": "{}", '.format(key,arg_dict[key])
        #arg_list = ' '.join(map(str, arg_list))
        #run_str = run_str[:-2] + '}\''
        #new_output = run_str
        #print(new_output)
        #process = subprocess.run(new_output, shell=True)

        print([blender_loc, "--background","--python",custom_loc] + arg_list)
        isShell = not os.name == 'posix'
        process = subprocess.run([blender_loc, "--background", "--python",custom_loc] + arg_list, shell=isShell)
        
    def _renderBlender(self, list_of_objs, num_rotations, root_folder):
        print('RENDERING')
        if self.should_render_forabot_val.get():
            print('RENDERING FORABOT')
            blend_file_loc = os.path.abspath(os.path.join(os.getcwd(),'forabot_render_final.blend'))
            self._renderBlenderMaterial(list_of_objs, num_rotations, blend_file_loc, 'Forabot', root_folder)
        if self.should_render_microfluidics_val.get():
            print('RENDERING MICROFLUIDICS')
            blend_file_loc = os.path.abspath(os.path.join(os.getcwd(),'microfluidics_render_final.blend'))  
            self._renderBlenderMaterial(list_of_objs, num_rotations, blend_file_loc, 'Microfluidics', root_folder)
        
    def _renderBlenderMaterial(self, list_of_objs, num_rotations, blend_file_loc, material_name, root_folder):
        render_script = os.path.abspath(os.path.join(os.getcwd(),'generate_renders.py'))
        arg_dict = {}
        arg_dict['root_folder'] = root_folder
        arg_dict['samples'] = list_of_objs
        arg_dict['rotations'] = num_rotations
        arg_dict['material_name'] = material_name
        arg_list = ['--', json.dumps(arg_dict)]

        print([self.blender_render_location, "-b", blend_file_loc, "--background", "--python", render_script] + arg_list)
        isShell = not os.name == 'posix'
        process = subprocess.run([self.blender_render_location, "-b", blend_file_loc, "--background", "--python", render_script] + arg_list, shell=isShell)

if __name__ == '__main__':
    ui = UserInterface()
    ui.root.mainloop()
