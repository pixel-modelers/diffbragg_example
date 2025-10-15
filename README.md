1. First setup a simtbx environment as shown [here](https://smb.slac.stanford.edu/~dermen/easybragg/)

2. Ensure DIALS is in the above environment

```
# ensure you are in the correct CONDA env:
python -c "import simtbx"
mamba install -c conda-forge dials -y
```

3. Install the scorer:

```
git clone https://github.com/pixel-modelers/score_trainer.git
cd score_trainer
pip install -e .
score.getMod
```

4. Get an image mask, an MTZ file, a stills process PHIL file, and the images from [SBgrid dataset 747](https://data.sbgrid.org/dataset/747/)

```
wget https://smb.slac.stanford.edu/~dermen/dbex/747_mask.pkl
wget https://smb.slac.stanford.edu/~dermen/dbex/scaled.mtz 
wget https://smb.slac.stanford.edu/~dermen/dbex/stills_proc.phil
wget https://smb.slac.stanford.edu/~dermen/dbex/refGeom.expt
```

5. Run `dials.stills_process` to get an initial starting model. Note we are running stills process on rotation data, so expect some inaccuracy:

```
dials.stills_process  stills_proc.phil  747/*cbf input.reference_geometry=refGeom.expt output_dir=sp.proc dispatch.integrate=False
```

Note, optionally install mpi to speed up processing:

```
mamba install -c conda-forge openmpi mpi4py
mpirun -n 10 dials.stills_process  stills_proc.phil  747/*cbf input.reference_geometry=refGeom.expt output_dir=sp.proc mp.method=mpi dispatch.integrate=False
```

6. Install the `dbex` repo

```
git clone https://github.com/pixel-modelers/diffbragg_example.git 
cd diffbragg_example
pip install -e .
```

To fit an image, look for the `*indexed.refl` and `*refined.expt` files in the stills_process output folder. Each pair of files represents multiple images, and the `%s_indexed.expt` and `%s_refined.refl` files go hand-in-hand, with the `.expt` containing the crystal/detector/beam models and the `.refl` containing the meta data describing the Bragg reflections used to fit the model (e.g. regions of interest on the detector).

Fit an image with `dbex.refine_one` by providing a pair of .expt / .refl files with a corresponding experiment index (as each pair of files represents multiple images). 

```
DIFFBRAGG_USE_CUDA=1 python -m dbex.refine_one -e sp.proc/idx-0001_refined.expt  -r sp.proc/idx-0001_indexed.refl  -i 0 -o test.h5 -m 747_mask.pkl  -z scaled.mtz 
```

To visualize the results run e.g.

```
python -m dbex.look test.h5
```



