from argparse import ArgumentParser

ap = ArgumentParser()

ap.add_argument(
    '-e','--exptName',
    type=str,
    required=True,
    help="Path to the DIALS experiment list (.json) file."
)
ap.add_argument(
    '-r','--reflName',
    type=str,
    required=True,
    help="Path to the DIALS reflection table (.refl) file."
)
ap.add_argument(
    '-i','--exptIdx',
    type=int,
    required=True,
    help="The integer index (id) of the experiment to be refined from the experiment list and reflection table."
)
ap.add_argument("-o", "--outFile", type=str, required=True)
ap.add_argument("-m", "--maskFile", type=str, required=True)
ap.add_argument("-z", "--mtzFile", type=str, required=True)
ap.add_argument("-c", "--mtzCol", type=str, default="F,SIGF")
args = ap.parse_args()
devid = 0


import h5py
import numpy as np
from scipy.optimize import minimize


from score_trainer import roi_check
from dbex.run_diffbragg import run_diffbragg
from dbex.data_load import DataLoad


DL=DataLoad(args)

# run model refinement

Bragg = run_diffbragg(DL, devId=devid)

# <><><><><><>
# TODO define a Bragg array from the PyTorch best-fit
# Bragg =
# thats the same shape as the DL.data
# <><><><><><><

# compare model to data
CHECKER = roi_check.roiCheck()

# optional mimization function to optimize a per-ROI scale factor
def func(x, CHECKER, bragg_im, bg_im, dat_im):
    bragg_scale = x[0]
    score = CHECKER.score(dat_im, bragg_scale**2*bragg_im + bg_im)
    resid = 1-score
    return resid

data_subims = []
bg_subims = []
bragg_subims = []
opt_bragg_scales = []
opt_bg_scales = []
opt_offsets = []
mask_subims = []
model_subims = []
scores = []
for i_sb, (pid,  (x1,x2,y1,y2)) in enumerate(zip(DL.pids, DL.bbox)):
    Y, X = np.indices((y2-y1, x2-x1))
    bg_im = DL.background_image[pid, y1:y2, x1:x2]
    assert not np.any(np.isnan(bg_im))
    x = slice(x1,x2,1)
    y = slice(y1,y2,1)
    dat_im = DL.data[pid,y, x]
    bragg_im = Bragg[pid, y, x]
    min_out = minimize(func, x0=[1], args=(CHECKER, bragg_im, bg_im, dat_im), method="Nelder-Mead")
    if min_out.success:
        opt_bragg_scale = min_out['x'][0]**2
        #opt_bg_scale = min_out['x'][1]**2
        #opt_offset = min_out['x'][2]
    else:
        opt_bragg_scale = 1
        #opt_bg_scale = 1
        #opt_offset = 0

    mod_im =bg_im + opt_bragg_scale*bragg_im
    score = CHECKER.score(dat_im, mod_im)
    print("roi=%d : score= %.1f" % (i_sb, score*100))
    model_subims.append(mod_im)
    data_subims.append( dat_im)
    bg_subims.append(bg_im)
    bragg_subims.append(bragg_im)

    opt_bragg_scales.append(opt_bragg_scale)
    #opt_bg_scales.append(opt_bg_scale)
    #opt_offsets.append(opt_offset)

    scores.append(score)

with h5py.File(args.outFile, "w") as h:
    h.create_dataset("score", data=scores)
    h.create_dataset("bragg_scale", data=opt_bragg_scales)

    for i in range(len(scores)):
        h.create_dataset("data/roi%d" % i, data=data_subims[i])
        h.create_dataset("model/roi%d" % i, data=model_subims[i])
        h.create_dataset("bragg/roi%d" % i, data=bragg_subims[i])
        h.create_dataset("bg/roi%d" % i, data=bg_subims[i])

print("Average score:", 100*np.mean(scores), "+-", 100*np.std(scores))
print("Fraction of spots well modeled= %.1f%%" % (100*sum([s >= 0.5 for s in scores])/len(scores), ) )
print(f"Visualize using `python -m dbex.look {args.outFile}`")
