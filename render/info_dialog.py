from tkinter import *

class GridInfo:

    def __init__(self, parent, params):
        self.top = Toplevel()

        chambers = params['num_cham']
        shape = params['shape']
        growth  = params['growth']
        TF = params['TF']
        phi = params['phi']
        beta = params['beta']
        dims = params['dims']
        increments = params['increments']
        #final_size = shape_base*1000*(growth**(chambers-1))*2

        cham_addition = "To {} By {}".format(dims[0], increments[0]) if dims[0] else None
        growth_addition = "To {} By {}".format(dims[1], increments[1]) if dims[1] else None
        distance_addition = "To {} By {}".format(dims[2], increments[2]) if dims[2] else None
        phi_addition = "To {} By {}".format(dims[3], increments[3]) if dims[3] else None
        beta_addition = "To {} By {}".format(dims[4], increments[4]) if dims[4] else None

        Label(self.top, text="Number of Chambers {} {}".format(str(chambers),cham_addition)).pack()
        Label(self.top, text="Shape of Foram Chambers {}".format(str(shape))).pack()
        Label(self.top, text="Growth Rate {} {}".format(str(growth),growth_addition)).pack()
        Label(self.top, text="Distance Between Chambers {} {}".format(str(TF),distance_addition)).pack()
        Label(self.top, text="Angle in Plane {} {}".format(str(phi),phi_addition)).pack()
        Label(self.top, text="Angle towards Z-axis {} {}".format(str(beta),beta_addition)).pack()

        b = Button(self.top, text="OK", command=self.ok)
        b.pack(pady=5)

    def ok(self, event=None):
        self.top.destroy()
