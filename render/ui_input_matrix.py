from tkinter import *
from tkinter import ttk, messagebox, filedialog
import subprocess
import os
import json
from ui_input import UserInterface
from info_dialog import GridInfo
import time
import glob

class UserInterfaceMatrix(UserInterface):
    def __init__(self):
        super().__init__()

        planar_angle_cmd = self.root.register(self._isPlanarAngle)
        beta_angle_cmd = self.root.register(self._isBetaAngle)
        cham_cmd = self.root.register(self._isChamberCt)
        size_cmd = self.root.register(self._isSize)
        dir_cmd = self.root.register(self._isDirection)
        growth_cmd = self.root.register(self._isGrowth)


        self.value_label_var.set('Start')
        self.end_label = ttk.Label(self.root, text='End')
        self.increment_label = ttk.Label(self.root, text='Increment')

        self.value_label.grid(row=0, column=1)
        self.end_label.grid(row=0, column=2)
        self.increment_label.grid(row=0, column=3)

        self.next_num_chambers = ttk.Spinbox(self.root, from_=1, to=50,
                                validate='all', validatecommand=(cham_cmd,'%P'))
        self.next_growth = ttk.Spinbox(self.root, from_=1, to=2, validate='all',
                                validatecommand=(growth_cmd,'%P'), increment=0.05)
        self.next_direction = ttk.Spinbox(self.root, from_=-1, to=1, increment=0.05,
                                validate='all', validatecommand=(dir_cmd, '%P'))
        self.next_planar_angle = ttk.Entry(self.root, validate='all',
                                validatecommand=(planar_angle_cmd, '%P', '%i'))
        self.next_z_angle = ttk.Entry(self.root, validate='all',
                                validatecommand=(beta_angle_cmd, '%P', '%i'))

        self.next_num_chambers.grid(row=1, column=2)
        self.next_growth.grid(row=3, column=2)
        self.next_direction.grid(row=4, column=2)
        self.next_planar_angle.grid(row=5, column=2)
        self.next_z_angle.grid(row=6, column=2)

        ang_inc_cmd = self.root.register(self._isAngleIncrement)
        unit_inc_cmd = self.root.register(self._isUnitIncrement)

        self.chamb_increment = ttk.Spinbox(self.root, from_=1, to=10,
                                validate='all', validatecommand=(cham_cmd,'%P'))
        self.growth_increment = ttk.Spinbox(self.root, from_=0, to=1, validate='all',
                                validatecommand=(unit_inc_cmd,'%P'), increment=0.01)
        self.direction_increment = ttk.Spinbox(self.root, from_=0, to=1, increment=0.01,
                                validate='all', validatecommand=(unit_inc_cmd, '%P'))
        self.planar_increment = ttk.Spinbox(self.root, from_=-90, to=90,
                                validate='all', validatecommand=(ang_inc_cmd,'%P'))
        self.z_increment = ttk.Spinbox(self.root, from_=-90, to=90,
                                validate='all', validatecommand=(ang_inc_cmd,'%P'))

        self.chamb_increment.grid(row=1, column=3)
        self.growth_increment.grid(row=3, column=3)
        self.direction_increment.grid(row=4, column=3)
        self.planar_increment.grid(row=5, column=3)
        self.z_increment.grid(row=6, column=3)

        self.cham_dim = IntVar()
        self.growth_dim = IntVar()
        self.direction_dim = IntVar()
        self.planar_dim = IntVar()
        self.z_angle_dim = IntVar()

        self.is_next_num_chambers = Checkbutton(self.root, var=self.cham_dim)
        self.is_next_growth = Checkbutton(self.root, var=self.growth_dim)
        self.is_next_direction = Checkbutton(self.root, var=self.direction_dim)
        self.is_next_planar_angle = Checkbutton(self.root, var=self.planar_dim)
        self.is_next_z_angle = Checkbutton(self.root, var=self.z_angle_dim)

        self.is_next_num_chambers.grid(row=1, column=4)
        self.is_next_growth.grid(row=3, column=4)
        self.is_next_direction.grid(row=4, column=4)
        self.is_next_planar_angle.grid(row=5, column=4)
        self.is_next_z_angle.grid(row=6, column=4)

        self.run_selected_grid = ttk.Button(self.root, text="Open Grid",
                                command=self._openSelectedGrid)

        self.run_selected_grid.grid(row=10, column=2)


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
        # = ((final_sizes * 0.5) / (growth**(num_cham-1)))/1000
        total_dim = 0
        if False: #sum([int(i.get()) for i in [self.cham_dim, self.growth_dim, self.direction_dim, self.planar_dim, self.z_angle_dim]])>3:
            messagebox.showerror(title='Too Many Dimensions', message='Must have 3 or fewer checked')
        else:
            dims = [None]*5
            increments = [None] * 5
            if self.cham_dim.get():
                tmp = self.next_num_chambers.get()
                tmp2 = self.chamb_increment.get()
                if tmp:
                    dims[0] = int(tmp)
                    if tmp2:
                        increments[0] = int(tmp2)
                    else:
                        self._incrementInvalid()
                        return
                else:
                    self._dimensionInvalid()
                    return
            if self.growth_dim.get():
                tmp = self.next_growth.get()
                tmp2 = self.growth_increment.get()
                if tmp:
                    dims[1] = float(tmp)
                    if tmp2:
                        increments[1] = float(tmp2)
                    else:
                        self._incrementInvalid()
                        return
                else:
                    self._dimensionInvalid()
                    return
            if self.direction_dim.get():
                tmp = self.next_direction.get()
                tmp2 = self.direction_increment.get()
                if tmp:
                    dims[2] = float(tmp)
                    if tmp2:
                        increments[2] = float(tmp2)
                    else:
                        self._incrementInvalid()
                        return
                else:
                    self._dimensionInvalid()
                    return
            if self.planar_dim.get():
                tmp = self.next_planar_angle.get()
                tmp2 = self.planar_increment.get()
                if tmp:
                    dims[3] = float(tmp)
                    if tmp2:
                        increments[3] = float(tmp2)
                    else:
                        self._incrementInvalid()
                        return
                else:
                    self._dimensionInvalid()
                    return
            if self.z_angle_dim.get():
                tmp = self.next_z_angle.get()
                tmp2 = self.z_increment.get()
                if tmp:
                    dims[4] = float(tmp)
                    if tmp2:
                        increments[4] = float(tmp2)
                    else:
                        self._incrementInvalid()
                        return
                else:
                    self._dimensionInvalid()
                    return
            self._callBlender(num_cham,shape,TF,phi,beta,growth,final_size,aperture_scale,dims,increments)
            idcs = [i for i,val in enumerate(dims) if val]
            print(len("Number of Indices: {}".format(idcs)))
            if len(idcs) >= 1:
                initialdir = "../data/synthetic/grid/*.json"
                grid_loc = os.path.abspath(initialdir)
                list_of_files = glob.glob(grid_loc)
                grid_file_loc = max(list_of_files, key=os.path.getctime)
                with open(grid_file_loc,'r') as f:
                    arg_dict = json.load(f)
                sample_ids = arg_dict['samples']
                self._renderBlender(sample_ids, self.num_rotations.get(), os.path.abspath('..'))
                self._openLastGrid(grid_file_loc)
            elif len(idcs) == 0:
                self._runSingleVisualization()
            
    def _populateDims(self, str_val):
        return float(tmp) if not tmp else None

    def _dimensionInvalid(self):
        messagebox.showerror(title='Invalid Dimension', message='Each checked box must have 2 associated values')

    def _incrementInvalid(self):
        messagebox.showerror(title='Invalid Increment', message='Ensure each used increment value is populated')

    def _isUnitIncrement(self,val):
        return self._isInRange(val, 1, 0, float)

    def _isAngleIncrement(self,val):
        return self._isInRange(val, 360, 0, float, True)

    def _openLastGrid(self, grid_file_loc):
        self._runGrid(grid_file_loc)

    def _openSelectedGrid(self):
        initialdir = "../data/synthetic/grid"
        filetypes = [("Grid Visualization","*.json")]
        griddir = os.path.join(os.getcwd(),initialdir)
        grid_file_loc =  filedialog.askopenfilename(parent = self.root, title='Select Grid to Visualize', initialdir = griddir,filetypes = filetypes)
        self._showGridParams(grid_file_loc)
        self._runGrid(grid_file_loc)

    def _runGrid(self, grid_file_loc):
        custom_loc = os.path.abspath('visualizeMorphospace.py')
        arg_list = ['--', grid_file_loc]
        isShell = not os.name == 'posix'
        process = subprocess.run([self.blender_location, "--python",custom_loc] + arg_list, shell=isShell)

    def _showGridParams(self, file_loc):
        with open(file_loc, 'r') as f:
            params = json.load(f)
        info_dialog = GridInfo(self.root, params)
        self.root.update()
        self.root.update()


if __name__ == '__main__':
    ui = UserInterfaceMatrix()
    ui.root.mainloop()
