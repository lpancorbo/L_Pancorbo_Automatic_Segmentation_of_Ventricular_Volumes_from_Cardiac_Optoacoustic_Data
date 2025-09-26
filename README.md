# Automatic Segmentation of Ventricular Volumes from Cardiac Optoacoustic Data
Repository containing the code used in my semester project at the Multiscale Functional and Molecular Imaging Lab (ETH Zürich) under supervision of Dr. Çağla Özsoy, Dr. Luis Deán-Ben & Prof. Daniel Razansky.

## Environment
To set up the environment, follow the steps below.

```bash
# Create a new conda environment
conda create -n ventricular-seg python=3.10

# Activate the environment
conda activate ventricular-seg

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

## Demo
You can find an example of how to use this project in [notebooks/example.ipynb](notebooks/example.ipynb). The code snippet below demonstrates a basic usage:

```bash
propagated_mask = restrictedPropagation(volume_4d, t_slice=170, dimension=1, denoised_method='tvl1', active_contours_method='acwe')
```
To perform the segmentation of the whole volume at time 170 in the sagittal view (dimension=1).