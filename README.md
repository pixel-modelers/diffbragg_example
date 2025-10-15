* First setup a simtbx environment as shown [here](https://smb.slac.stanford.edu/~dermen/easybragg/)

* Ensure DIALS is in the above environment

```
# ensure you are in the correct CONDA env:
python -c "import simtbx"
mamba install -c conda-forge dials -y
```

* Install the scorer:

```
git clone https://github.com/pixel-modelers/score_trainer.git
cd score_trainer
pip install -e .
score.getMod
```

* Get an image mask, an MTZ file, a stills process PHIL file, and the images from [SBgrid dataset 747](https://data.sbgrid.org/dataset/747/)

```
wget https://smb.slac.stanford.edu/~dermen/dbex/747_mask.pkl
wget https://smb.slac.stanford.edu/~dermen/dbex/scaled.mtz 
wget https://smb.slac.stanford.edu/~dermen/dbex/stills_proc.phil
wget https://smb.slac.stanford.edu/~dermen/dbex/refGeom.expt
```

* Run `dials.stills_process` to get an initial starting model

```
dials.stills_process  stills_proc.phil  747/*cbf input.reference_geometry=refGeom.expt output_dir=sp.proc 
```

Note, optionally install mpi to speed up processing:

```
mamba install -c conda-forge openmpi mpi4py
mpirun -n 10 dials.stills_process  stills_proc.phil  747/*cbf input.reference_geometry=refGeom.expt output_dir=sp.proc mp.method=mpi
```

