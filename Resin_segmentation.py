
import time
import numpy as np
from skimage import io
from skimage.morphology import dilation
from skimage.filters import gaussian
from skimage.measure import find_contours
from skimage.segmentation import active_contour
step = 2
init_num = 256
print('Initiliaze')
img_init = io.imread('./init%d_2.png'%(init_num), as_grey=True)
img_init = dilation(img_init)
init = find_contours(img_init, 0.5)
init = init[0][::50]
init = init[::-1,::-1]
init = init[init[:,0].argsort()]
#init_int = np.array(init, dtype=np.int32)
#img_init = np.zeros_like(img_init)
#color = 1
#for i in [-1,0,1]:
#    for j in [-1,0,1]:
#        img_init[init_int[:,1]+i,init_int[:,0]+j] = color
#io.imsave('img_init.bmp', img_init)
print(init)
print(np.shape(init))
for i in range(init_num-step,init_num-20*step,-step):
    print('Load image', i)
    img = io.imread('./../OCT_image/1026_with_glass_bead_before_irradiation/Filename_%04d.png'%(i))
    img_org = np.sum(img[:,:,:3], axis=2)/3
    img_org = gaussian(img, 3)
    
    print('Snake')
    time_start = time.time()
    snake = active_contour(img_org, init, alpha=0.001, beta=0.001, w_line=0, w_edge=5, max_iterations=500, bc='free')
    time_end = time.time()
    print('time= %f'%(time_end-time_start))
    
    snake_int = np.array(snake, dtype=np.int32)
    img_snake = np.array(img)[:,:,:3]
    color = [0,0,255]
    color = 0
    for j in [-1,0,1]:
        for k in [-1,0,1]:
            img_snake[snake_int[:,1]+j,snake_int[:,0]+k] = color
    io.imsave('img_snake_file%d.png'%(i), img_snake)
    #print(snake)
    #print(np.shape(snake))
    
    init = np.copy(snake)
    print()
    
print('Done')