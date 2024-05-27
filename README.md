# L_Pancorbo_Automatic_Segmentation_of_Ventricular_Volumes_from_Cardiac_Optoacoustic_Data
Repository for Semester Project "Automatic Segmentation of Ventricular Volumes from Cardiac Optoacoustic Data" done at the Multiscale Functional and Molecular Imaging Lab (ETH Zürich) under supervision of Dr. Çağla Özsoy, Dr. Luis Deán-Ben & Prof. Daniel Razansky

## Requirements
The following packages are required to run this project:

- numpy
- matplotlib
- ipywidgets
- scipy
- skimage
- Feret
- tqdm (optional)

You can install these packages using pip:

```bash
pip install numpy matplotlib ipywidgets scipy scikit-image Feret
```

## Example Code
You can find an example of how to use this project in the `example.ipynb` notebook. The code snippet below demonstrates a basic usage:

```bash
propagated_mask = restrictedPropagation(volume_4d, t_slice=170, dimension=1, denoised_method='tvl1', active_contours_method='acwe')
```
To perform the segmentation of the whole volume at time 170 in the sagittal view (dimension=1).
