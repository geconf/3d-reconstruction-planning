# Motion Planning for 3D Model Reconstruction

Final Project for WPI's RBE550-Motion Planning done in collaboration with https://elpislab.org/. We built 3D object models using a robot manipulator and an Intel Realsense D435 RGBD camera.

Relevant works:

```
 @inproceedings{zhong2024expansiongrr,
   title={Expansion-GRR: Efficient Generation of Smooth Global Redundancy Resolution Roadmaps},
   author={Zhong, Li and Chamzas},
   booktitle={IROS},
   year={2024},
 }

## Installation

* Clone the repository:

```shell
git clone https://github.com/geconf/planning-3d-reconstruction
```

* Create and load the conda environment:

```shell
conda env create -f environment.yml
conda activate plan-3d-recon
```

* Build GRR Roadmaps

```shell
cd Expansion-GRR
python redundancy.py <robot> <rotation_type> # e.g. ur10 rot_variable_yaw
```

## Visualization

```shell
cd ..
python main.py
```
