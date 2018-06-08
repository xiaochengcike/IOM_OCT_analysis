# IOM_OCT_analysis

- **Segmentation for resin part using active contour.**  
  ![skimage.segmentation.active_contour].
  
  [skimage.segmentation.active_contour]: http://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.active_contour

- **Detection for glass bead and denoising using Fourier transform.**  
  Process:  
  Image -> DFT -> Filter -> IDFT -> Gaussian Blur -> Otsu Threshold -> ROI -> Find Contours
