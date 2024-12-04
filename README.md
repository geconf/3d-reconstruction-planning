# Motion Planning for 3D Model Reconstruction

Final Project for WPI's RBE550-Motion Planning done in collaboration with https://elpislab.org/. We built 3D object models using a robot manipulator and an Intel Realsense D435 RGBD camera.

Relevant works:

```
 @inproceedings{zhong2024expansiongrr,
   title={Expansion-GRR: Efficient Generation of Smooth Global Redundancy Resolution Roadmaps},
   author={Zhong, L. and Chamzas, C.},
   booktitle={IROS},
   year={2024},
 }

@article{calli2015benchmarkingycb,
  title={Benchmarking in Manipulation Research: The YCB Object and Model Set and Benchmarking Protocols},
  author={Calli, B. and Walsman, A. and Singh, A. and Srinivasa, S. and Abbeel, P. and Dollar, A. M.},
  booktitle={IEEE Robotics and Automation Magazine},
  year={2015},
}

@article{calli2015yalecmuberkeley,
  title={Yale-CMU-Berkeley dataset for robotic manipulation research},
  author={Calli, B. and Singh, A. and Bruce, J. and Walsman, A. and Konolige, K. and Srinivasa, S. and Abbeel, P. and Dollar, A. M.},
  booktitle={The International Journal of Robotics Research},
  year={2017},
}

@inproceedings{calli2015ycbobject,
  title={The YCB Object and Model Set: Towards Common Benchmarks for Manipulation Research},
  author={Calli, B. and Singh, A. and Walsman, A. and Srinivasa, S. and Abbeel, P. and Dollar, A. M.},
  booktitle={ICAR},
  year={2015},
}
```

## Installation

* Clone the repository:

```shell
git clone https://github.com/geconf/planning-3d-reconstruction
cd ./planning-3d-reconstruction
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

---
RBE 550 - Motion Planning 2024 taught by Professor Constantinos Chamzas at Worcester Polytechnic Institute Robotics Engineering Department
