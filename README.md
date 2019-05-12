**# [Patch based segmentation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4745344/)**

Patch-based methods have been shown to be an effective approach for labeling brain structures (and other body structures) In general, these approaches label each voxel of a target image by comparing the image patch, centered on the voxel with patches from an atlas library, and assigning the most probable label according to the closest matches. Often, a localized search window centered around the target voxel is used. Various patch-based label fusion procedures have been proposed and were shown to produce accurate and robust segmentation. Patch-based techniques have recently demonstrated high performance in various computer vision tasks, including texture synthesis, in painting, and super-resolution.
In our model, we took 30*30 patches of MRI images. 
![patch](https://user-images.githubusercontent.com/48405935/57585769-3d4ba080-750a-11e9-9002-3aff03105012.jpg)
