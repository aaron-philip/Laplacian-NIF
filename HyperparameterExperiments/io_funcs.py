"""
This file contains many of the useful functions regularly used to format or use
training data as well as tools to generate data (last two)

"""
from sys import path
path.append("/mnt/scratch/philipaa/tddft-emulation/nif")
from scipy.io import FortranFile
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
import math

def gen_sample(f,batch_size,specific=None):
    """
    Inputs: 
    f: file path
    batch_size: number of consecutively stored entries desired
    specific: allows user to specify the index to draw from. Not all that useful when 
    data is shuffled

    Returns: Sample from specified dataset. 
    """
    if specific is not None:
        I = f['Input'][specific:specific+batch_size]
        O = f['Output'][specific:specific+batch_size]

    else:
        ndata = f['Input'].shape[0]
        beg = random.randint(0,ndata*0.8-batch_size-1)
        I = f['Input'][beg:beg+batch_size] #np.zeros((batch_size, 6915))
        O = f['Output'][beg:beg+batch_size] #np.zeros((batch_size,4))

    return I,O

def generator(f, batch_size):
    """
    Function used to draw training data from a h5 dataset in batches when training

    Inputs: 
    f: file path to dataset
    batch_size: consecutive entries to draw

    Returns: Input and corresponding output pair
    """
    while True:
        batch_index = 0
        batches_list = list(range(int(math.ceil(float(f['Input'].shape[0]) / batch_size))))

        while batch_index < len(batches_list):
            ndata = f['Input'].shape[0]

            # Note that samples are always taken from first 80% of dataset to enable validation testing
            beg = random.randint(0,int(ndata*0.8-batch_size-1))
            I = f['Input'][beg:beg+batch_size] 
            O = f['Output'][beg:beg+batch_size] 
            batch_index += 1
            yield (I,O)

def format_input(rho0, current0, path):
    """
    Returns the array that should be passed to trained model for predictions.
    Useful for examining model training (eg. printing a prediction)
    Designed specifically for systems that had sensors regularly distributed over the space

    Inputs: 
    rho0, current0: can be obtained with read_tdd
    path: path to the scaling factors .npy file setup during dataset generation. 
    """
    field_t0 = np.empty([24, 24, 24, 4])
    field_t0[:,:,:,0] = rho0
    field_t0[:,:,:,1:4] = current0

    shift, scale, _,_ = np.load(path)
    
    field_normed = (field_t0 - shift) / scale 
    res = 2

    # included length used by SKY3D code for reference, but input is normalized to [-1,1]
    x = np.arange(-11.5, 12.5, 1) / 11.5
    y = np.arange(-11.5, 12.5, 0.25) / 11.5
    z = np.arange(-11.5, 12.5, 0.25) / 11.5

    coursened = field_normed[::res,::res,::res]
    temp_inp = np.append(coursened.flatten(), np.array([x[0], y[0], z[0]]))

    # Create input for a central slice (for visulization)
    In = np.empty((96, 96, 6915))
    for i in range(0,96):
        for j in range(0,96):
            In[i,j] = temp_inp.copy()
            In[i,j,-3:] = [x[11], y[i], z[j]]
    return In

def format_input_sensor6(rho0, current0, path):
    """
    Returns the array that should be passed to trained model for predictions.
    Useful for examining model training (eg. printing a prediction)
    Designed specifically for the 6 sensor restoration models

    Inputs: 
    rho0, current0: can be obtained with read_tdd
    path: path to the scaling factors .npy file setup during dataset generation. 
    """
    field_t0 = np.empty([24, 24, 24, 4])
    field_t0[:,:,:,0] = rho0
    field_t0[:,:,:,1:4] = current0

    x_shift, x_scale = np.load(path)
    field_t0 = (field_t0 - x_shift) / x_scale

    # included length used by SKY3D code for reference, but input is normalized to [-1,1]
    x = np.arange(-11.5, 12.5, 1) / 11.5
    y = np.arange(-11.5, 12.5, 0.25) / 11.5
    z = np.arange(-11.5, 12.5, 0.25) / 11.5

    PNET_coords = get_subcube_face_centers(field_t0, 3)
    SNET_coords = sub_cube_points(field_t0, 6)

    # ParameterNet sensors
    coursened = []
    [coursened.append(field_t0[i]) for i in PNET_coords]
    coursened = np.array(coursened)
    temp_inp = np.append(coursened.flatten(), np.array([x[0], y[0], z[0]]))

    # Create input for a central slice (for visulization)
    In = np.empty((96, 96, 27))
    for i in range(0,96):
        for j in range(0,96):
            In[i,j] = temp_inp.copy()
            In[i,j,-3:] = [x[11], y[i], z[j]]
    return In

def read_tdd(filename, dir):
    """
    Extracts rho, current, and (x,y,z) stored in FortranFiles for usage. 
    Inputs: 
   
    filename: name of tdd file. 
    dir: directory the tdd file is stored in
    """

    tdd_file = FortranFile(dir+filename,'r')

    record = tdd_file.read_record('i4', 'f8', 'i4', 'i4', 'i4')
    iter = record[0][0]
    time = record[1][0]
    ncolx = record[2][0]
    ncoly = record[3][0]
    ncolz = record[4][0]

    record = tdd_file.read_record('f8', 'f8', 'f8', 'f8', np.dtype(('f8', (ncolx,1))), np.dtype(('f8', (ncoly,1))), np.dtype(('f8', (ncolz,1))))

    dx = record[0]
    dy = record[1]
    dz = record[2]
    wxyz = record[3]

    x = record[4].T[0]
    y = record[5].T[0]
    z = record[6].T[0]

    record = tdd_file.read_record('c16','b','c')
    record = tdd_file.read_reals(np.dtype(('f8', (ncolx, ncoly, ncolz))))
    rho = record.T
    record = tdd_file.read_record('c16','b','c')
    record = tdd_file.read_reals(np.dtype(('f8', (ncolx,ncoly, ncolz,3))))
    current = record
    return rho, current, (x,y,z)

def plotXSlice(rho, coord, x_slice, log=False):
    """
    Useful for examining cross-sections of density

    Inputs: 
    rho: density field (obtain from read_tdd)
    coord: coordinates of the field to graph against accordingly
    x_slice: the index of the slice that is being examined. 
    log: Examine on a logarithmic scale to observe numerical instability
    """
    if log:
        r =np.log(rho[x_slice])
    else: 
        r = rho[x_slice]
    pos = plt.imshow(r,vmin=np.min(r), vmax = np.max(r), 
                extent =[coord[1][0], coord[1][-1], coord[2][0], coord[2][-1]],
                interpolation='nearest', cmap='rainbow', origin ='lower') 
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title("Density at Plane: X = %s" % x_slice)
    plt.colorbar(pos)

# returns an array of coordinates 
def get_subcube_face_centers(grid, half_height):
    """
    This is how the indices of the 6 sensors is obtained.

    Inputs: 
    grid: The grid being studied/trained to. The assumption currently is that the grid is cubical
    but this can easily be modified.
    
    half_height: Essentially half of how "tall" the cube in which the nucleus is insribed in 
    should be
    """
    # Compute the indices of the center of the faces of the sub-cube
    center_indices = []
    center = grid.shape[0] // 2  # Middle of the mesh
    center_indices.append((center, center - half_height, center))  # Face 1
    center_indices.append((center, center + half_height, center))  # Face 2
    center_indices.append((center - half_height, center, center))  # Face 3
    center_indices.append((center + half_height, center, center))  # Face 4
    center_indices.append((center, center, center - half_height))  # Face 5
    center_indices.append((center, center, center + half_height))  # Face 6
    
    return center_indices

def sub_cube_points(mesh, sub_cube_height):
    """
    Used to sample points inside the "subcube" at full resolution 
    while sampling exterior points of the cube on the mesh at 2x resolution in each 
    dimension (used for 6 sensor models)

    Inputs: 
    mesh: can just be the rho mesh
    sub_cube_height: height of cube to sample fully within
    """

    # Get the dimensions of the mesh
    x_dim, y_dim, z_dim,_ = mesh.shape

    # Find the center of the mesh
    x_center = x_dim // 2
    y_center = y_dim // 2
    z_center = z_dim // 2

    # Find the bounds of the sub-cube
    x_min = max(0, x_center - sub_cube_height)
    x_max = min(x_dim, x_center + sub_cube_height)
    y_min = max(0, y_center - sub_cube_height)
    y_max = min(y_dim, y_center + sub_cube_height)
    z_min = max(0, z_center - sub_cube_height)
    z_max = min(z_dim, z_center + sub_cube_height)

    # Sample points inside the sub-cube at a higher resolution
    sub_cube_points = []
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            for z in range(z_min, z_max):
                sub_cube_points.append((x, y, z))

    # Sample points outside the sub-cube at a lower resolution
    non_sub_cube_points = []
    for x in range(0, x_dim, 2):
        for y in range(0, y_dim, 2):
            for z in range(0, z_dim, 2):
                if x < x_min or x >= x_max or y < y_min or y >= y_max or z < z_min or z >= z_max:
                    non_sub_cube_points.append((x, y, z))

    # Combine the sub-cube and non-sub-cube points
    all_points = sub_cube_points + non_sub_cube_points

    return all_points
