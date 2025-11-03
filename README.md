## Running Instructions

### Copy dataset
In the root folder, run:
```bash
cp -r data/dataset1/ src/Vx/dataset1/
cp -r data/dataset2/ src/Vx/dataset2/
cp -r data/dataset3/ src/Vx/dataset3/
```
replace x with the version you wish to run e.g. V1,V2..

## Running V1
#### Make libraries:
```
make lib
```
#### Make example3:
```
make example3
```

#### Run example and generate profile (uses dataset3 if not specified):
```
make run DATASET=dataset1|dataset2|dataset3 
```

#### Clean:
```
make clean
```

## Running V2 / V3

### CPU:

#### Compile CPU implementation
```
make cpu
```
#### Run CPU implementation
```
make run_cpu DATASET=dataset1|dataset2|dataset3 
```
#### Profile for CPU with gprof 
```
make profile_cpu 
```

### GPU:

#### Compile GPU implementation
```
make gpu
```
#### Run GPU implementation
```
make run_gpu DATASET=dataset1|dataset2|dataset3 
```
#### Profile for GPU with nsys
```
make profile_gpu DATASET=dataset1|dataset2|dataset3 
```
## HD Dataset Used
To access our HD dataset, visit: [valley](https://www.kaggle.com/datasets/hiyamrehan/valley-hd) on kaggle/hiyam-rehan
