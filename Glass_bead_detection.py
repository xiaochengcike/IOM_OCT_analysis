import numpy as np
import cv2
image_list = []
for i in range(198,203):
    num = i
    print(i)
    
    org_image = cv2.imread('./../OCT_image/1026_with_glass_bead_before_irradiation/Filename_%04d.png'%(num), cv2.IMREAD_GRAYSCALE)[::2,::2]
    image = np.array(org_image)
    cv2.imshow('org_image', image)
#    cv2.waitKey(0)
    
    ''' dft '''
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.array(np.log(np.abs(fshift))*10, dtype=np.uint8)
    cv2.imshow('dft', magnitude_spectrum)
#    cv2.waitKey(0)
    
    ''' dft filter '''
    rows, cols = image.shape
    crow, ccol = rows//2 , cols//2
        ### cross ###
    #width = 20
    #newimage = np.zeros_like(fshift)
    #newimage[:, ccol-width:ccol+width] = fshift[:, ccol-width:ccol+width]
    #newimage[crow-width:crow+width, :] = fshift[crow-width:crow+width, :]
    #newimage2 = np.zeros_like(magnitude_spectrum)
    #newimage2[:, ccol-width:ccol+width] = magnitude_spectrum[:, ccol-width:ccol+width]
    #newimage2[crow-width:crow+width, :] = magnitude_spectrum[crow-width:crow+width, :]
        ### square ###
    #width = 200
    #newimage = np.zeros_like(fshift)
    #newimage[crow-width:crow+width, ccol-width:ccol+width] = fshift[crow-width:crow+width, ccol-width:ccol+width]
    #newimage2 = np.zeros_like(magnitude_spectrum)
    #newimage2[crow-width:crow+width, ccol-width:ccol+width] = magnitude_spectrum[crow-width:crow+width, ccol-width:ccol+width]
        ### line ###
    width = 3
    newimage = np.array(fshift)
    newimage[crow-width:crow+width, :ccol-width] = 0
    newimage[crow-width:crow+width, ccol+width:] = 0
    newimage2 = np.array(magnitude_spectrum)
    newimage2[crow-width:crow+width, :ccol-width] = 0
    newimage2[crow-width:crow+width, ccol+width:] = 0
    #width = 1
    #newimage = np.array(fshift)
    #newimage[:crow-width, ccol-width:ccol+width] = 0
    #newimage[crow+width:, ccol-width:ccol+width] = 0
    #newimage2 = np.array(magnitude_spectrum)
    #newimage2[:crow-width, ccol-width:ccol+width] = 0
    #newimage2[crow+width:, ccol-width:ccol+width] = 0
        ### line removed part ###
    #width = 1
    #newimage = np.zeros_like(fshift)
    #newimage[crow-width:crow+width, :ccol-width] = fshift[crow-width:crow+width, :ccol-width]
    #newimage[crow-width:crow+width, ccol+width:] = fshift[crow-width:crow+width, :ccol-width]
    #newimage2 = np.zeros_like(magnitude_spectrum)
    #newimage2[crow-width:crow+width, :ccol-width] = magnitude_spectrum[crow-width:crow+width, :ccol-width]
    #newimage2[crow-width:crow+width, ccol+width:] = magnitude_spectrum[crow-width:crow+width, :ccol-width]
        ### circle ###
    #width = 100**2
    #newimage = np.array(fshift)
    #newimage2 = np.array(magnitude_spectrum)
    #H = np.shape(image)[0]
    #W = np.shape(image)[1]
    #for i in range(H):
    #    for j in range(W):
    #        if ((i-crow)**2+(j-ccol)**2)>width:
    #            newimage[i,j]=0
    #            newimage2[i,j]=0
    #width = 80**2
    #newimage = np.array(fshift)
    #newimage2 = np.array(magnitude_spectrum)
    #H = np.shape(image)[0]
    #W = np.shape(image)[1]
    #for i in range(H):
    #    for j in range(W):
    #        if ((i-crow)**2+(j-ccol)**2)<width:
    #            newimage[i,j]=0
    #            newimage2[i,j]=0
    cv2.imshow('dft filter', newimage2)
#    cv2.waitKey(0)
    
    ''' idft '''
    f_ishift = np.fft.ifftshift(newimage)
    image = np.fft.ifft2(f_ishift)
    image[image>255] = 255
    image = np.array(np.abs(image), dtype=np.uint8)
    cv2.imshow('idft', image)
#    cv2.waitKey(0)
    
    ''' blur '''
    itera = 3
    for i in range(itera):
    #    image = cv2.blur(image, ksize=(3,3))
    #    image = cv2.medianBlur(image, ksize=3)
        image = cv2.GaussianBlur(image, ksize=(3,3), sigmaX = 3*3)
    cv2.imshow('blur', image)
#    cv2.waitKey(0)
    
    ''' histogram equalization all'''
    #image = cv2.equalizeHist(image)
    #cv2.imshow('histogram equalization all', image)
    #cv2.waitKey(0)
    
    ''' histogram equalization vertical'''
    #image = np.array(image*0.1, dtype=np.uint8)
    #W = np.shape(image)[1]
    #for i in range(W):
    #    image[:,i] = np.reshape(cv2.equalizeHist(image[:,i]), [np.shape(image)[0]])
    #cv2.imshow('histogram equalization', image)
    #cv2.waitKey(0)
    
    ''' histogram equalization horizontal'''
    #H = np.shape(image)[0]
    #for i in range(H):
    #    image[i,:] = np.reshape(cv2.equalizeHist(image[i,:]), [np.shape(image)[1]])
    #cv2.imshow('histogram equalization horizontal', image)
    #cv2.waitKey(0)
    
    ''' threshold '''
    #thres = 170
    #image[image>thres] = 255
    #image[image<=thres] = 0
#    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 251, 0)
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    #print(ret)
    cv2.imshow('threshold', image)
#    cv2.waitKey(0)
    
    ''' erosion dilation '''
    #itera = 2
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    #image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=itera)
    #kernel = np.array([[1,1,1],[1,1,1],[1,1,1]], np.uint8)
    #image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=itera)
    #cv2.imshow('erosion dilation', image)
    #cv2.waitKey(0)
    
    ''' ROI '''
    buf_image = np.ones_like(image, dtype=np.uint8)
    buf_image = buf_image*255
    #buf_image[128:640,200:1208] = image[128:640,200:1208]
    buf_image[64:320,100:604] = image[64:320,100:604]
    image = buf_image
    image_list.append(np.copy(buf_image))
    cv2.imshow('ROI', image)
#    cv2.waitKey(0)
    
    ''' find contours '''
    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    shape = [np.shape(image)[0], np.shape(image)[1], 3]
    image = np.zeros(shape, np.uint8)
    image[:,:,0] = org_image
    image[:,:,1] = org_image
    image[:,:,2] = org_image
    
    ''' all draw'''
    #image = cv2.drawContours(image, contours, -1, (0,255,255), thickness = 2)
    
    ''' conditional drawing'''
    L = np.shape(contours)[0]
    for i in range(L):
        if cv2.contourArea(contours[i])>128 and cv2.contourArea(contours[i])<4096:
            image = cv2.drawContours(image, contours, i, (0,255,255), thickness=1)
    
    ''' show result '''
    #cv2.rectangle(image, (200,128), (1208,640), (0,0,255), thickness=2)
    cv2.imshow('result', image)
    cv2.waitKey(0)
    comb_image = np.zeros((np.shape(image)[0],np.shape(image)[1]*2,np.shape(image)[2]),dtype=np.uint8)
    comb_image[:,:np.shape(image)[1],0] = buf_image
    comb_image[:,:np.shape(image)[1],1] = buf_image
    comb_image[:,:np.shape(image)[1],2] = buf_image
    comb_image[:,np.shape(image)[1]:np.shape(image)[1]*2] = image
    cv2.imwrite('./%d_combined.jpg'%(num), comb_image)
    cv2.destroyAllWindows()
num = num-2
print(num)
org_image = cv2.imread('./../OCT_image/1026_with_glass_bead_before_irradiation/Filename_%04d.png'%(num), cv2.IMREAD_GRAYSCALE)[::2,::2]
out_image = np.zeros_like(image_list[0], dtype=np.float)
for im in image_list:
    out_image = out_image + im
out_image = np.array(out_image/5, dtype=np.uint8)

cv2.imwrite('%d_average.jpg'%(num), out_image)
cv2.waitKey(0)

''' blur '''
itera = 3
for i in range(itera):
    out_image = cv2.GaussianBlur(out_image, ksize=(3,3), sigmaX = 3*3)
cv2.imwrite('%d_blur.jpg'%(num), out_image)
cv2.waitKey(0)

''' threshold '''
ret, out_image = cv2.threshold(out_image, 0, 255, cv2.THRESH_OTSU)
cv2.imwrite('%d_threshold.jpg'%(num), out_image)
cv2.waitKey(0)

''' find contours '''
out_image, contours, hierarchy = cv2.findContours(out_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
shape = [np.shape(out_image)[0], np.shape(out_image)[1], 3]
out_image = np.zeros(shape, np.uint8)
out_image[:,:,0] = org_image
out_image[:,:,1] = org_image
out_image[:,:,2] = org_image

''' conditional drawing'''
L = np.shape(contours)[0]
for i in range(L):
    if cv2.contourArea(contours[i])>32 and cv2.contourArea(contours[i])<2048:
        out_image = cv2.drawContours(out_image, contours, i, (0,255,255), thickness=1)
    
''' show result '''
cv2.imwrite('%d_result.jpg'%(num), out_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
