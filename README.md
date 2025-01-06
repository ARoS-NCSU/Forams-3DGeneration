# synthetic_foram_model
The purpose of this repo is to generate realistic 3D models of foraminifera.

Link for Numerical Models (Page 674):
https://link.springer.com/content/pdf/10.1007%2F3-540-44860-8.pdf 

Visualization:
http://www.eforams.org/index.php?title=VirtuaLab

![Morphospace Tree](http://www.eforams.org/img_auth.php/1/10/Morphotree02a.jpg)

Data: http://www.endlessforams.org/

Classes:
- Pelagica 7k samples
- Heterohelix 7k samples  
- Bulloides 11k samples 
- Uvula 11k samples
- Dutertrei 10k samples  
- Pachyderma 10k samples
- Guembelitriella 11k samples

## Synthetic Data Generation

Add texture images in ```data/texture/``` folder.

Run the [morphospace.py](render/morphospace.py) file in [Blender 2.79][1] 

Run the [callBlenderRender.py](render/callBlenderRender.py) file in [Blender 3.2.0][2]

Notes: Set edit->preferences->system->"cycles render device" to CUDA and select the gpu to use gpu rendering

The generated forams are using micron values. Both the json and the obj file is treated in that manner. To scale the object to actual micron size for rendering, multiply by 1e-6.

## Train Model

Preprocess the data.

```python preprocessing.py```

Start the server with:

```python -m visdom.server```

Then, go to the following address in browser:

```http://localhost:8097```

Set ```train_path_real```, ```train_path_synthetic```, ```test_path_real``` in [opt.py](model/opt.py) to the correct path.
Domain Adapation network can be turned on by setting  ```DOMAIN_ADAPTATION_HEAD``` to ```True```.

```python train.py```


[1]: https://download.blender.org/release/Blender2.79/
[2]: https://download.blender.org/release/Blender3.2/
