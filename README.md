## Running Instructions

### Copy dataset
In the root folder, run:
```bash
cp -r data/dataset1/ src/V1/dataset1/
cp -r data/dataset2/ src/V1/dataset2/
cp -r data/dataset3/ src/V1/dataset3/
```

### Make libraries:
```
make lib
```
### Make example3:
```
make example3
```

### Run example and generate profile (uses dataset3 if not specified):
```
make run DATASET=dataset1|dataset2|dataset3 
```

### Clean:
```
make clean
```
