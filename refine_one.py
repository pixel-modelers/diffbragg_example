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
args = ap.parse_args()
devid = 0

config = open("xtal_refine.phil", "r").read()
config2 = open("fhkl_refine.phil", "r").read()

import logging
import numpy as np
from copy import deepcopy
import os

from libtbx.phil import parse
from simtbx.diffBragg import hopper_utils
from simtbx.modeling.forward_models import diffBragg_forward
from simtbx.command_line.hopper import phil_scope
from simtbx.diffBragg import utils
from simtbx.diffBragg import hopper_io
from dxtbx.model import ExperimentList
from dials.array_family import flex


from score_trainer import roi_check


# parameters object that controls hopper refinement
params = phil_scope.fetch(sources=[parse(config)]).extract()
params.fix.Ndef=False
params2 = phil_scope.fetch(sources=[parse(config2)]).extract()
for prm in (params,params2):
    prm.simulator.structure_factors.mtz_name=args.mtzFile
    prm.roi.hotpixel_mask=args.maskFile


F = utils.open_mtz(params.simulator.structure_factors.mtz_name, params.simulator.structure_factors.mtz_column)
F = F.generate_bijvoet_mates()
Finds = F.indices()
Famp = F.data()
Fmap = {h:amp for h,amp in zip(Finds, Famp)}

# experiment list and reflection table
Expt = ExperimentList.from_file(args.exptName)[args.exptIdx]

Refs = flex.reflection_table.from_file(args.reflName)
Refs = Refs.select(Refs['id'] == args.exptIdx)

# get the background 

data = utils.image_data_from_expt(Expt)
bbox,pids,tilt_coefs,_,_ = utils.get_roi_background_and_selection_flags(
    Refs, data, use_robust_estimation=False, shoebox_sz=14,
    pad_for_background_estimation=5, hotpix_mask=data < 0)


# run model refinement
logger = logging.getLogger("diffBragg.main")
logger.setLevel(logging.DEBUG)


def detector_refinement(model_df, Expt, params):
    new_El = ExperimentList()
    new_El.append(Expt)
    new_El.as_file("_geom_ref.expt")
    Refs.as_file("_geom_ref.refl")
    Refs['id'] = flex.int(len(Refs),0)
    model_df["geom_exp"] = "_geom_ref.expt"
    model_df["geom_ref"] = "_geom_ref.refl"
    model_df["geom_exp_idx"] = 0
    model_df.to_pickle("_geom_ref.pkl")
    from simtbx.diffBragg.refiners import geometry
    with open("_geom_groups.txt", "w") as o:
        for i_p in range(len(Expt.detector)):
            o.write("%d %d\n" % (i_p, i_p))
    params3 = deepcopy(params)
    params3.fix.Fhkl=True
    params3.refiner.panel_group_file="_geom_groups.txt"
    params3.geometry.fix.panel_rotations=[0,0,0]
    params3.geometry.fix.panel_translations=[0,0,1]
    params3.geometry.input_pkl="_geom_ref.pkl"
    params3.geometry.save_state_freq=100000
    params3.filter_during_refinement.enable=False
    params3.outdir="_geom.out"
    params3.max_process=1
    params3.geometry.optimize=True
    geometry.geom_min(params3)
    new_det = ExperimentList.from_file("_geom.out/diffBragg_detector.expt")[0].detector
    Expt.detector = new_det
    return Expt



for cycle in range(5):

    # REFINEMENT OF XTAL
    ref_out = hopper_utils.refine(Expt, Refs, params, return_modeler=True, free_mem=True, gpu_device=devid)
    Expt, _, Modeler, SIM, x = ref_out
    mdl_parm = hopper_utils.get_param_from_x(x, Modeler, as_dict=True)

    for prm in [params, params2]:
        prm.init.Nabc = mdl_parm['Na'], mdl_parm['Nb'], mdl_parm['Nc']
        prm.init.Ndef = mdl_parm['Nd'], mdl_parm['Ne'], mdl_parm['Nf']
        prm.init.G = mdl_parm['scale']

    #model_df = hopper_io.save_to_pandas(x, Modeler, SIM, args.exptName, params2, Expt, 0,
    #                                    args.reflName, None, 0, write_expt=False, write_pandas=False,
    #                                    exp_idx=args.exptIdx)


    # REFINEMENT OF FHKL
    ref_out2 = hopper_utils.refine(Expt, Refs, params2, return_modeler=True, free_mem=True, gpu_device=devid)
    _, _, Modeler2, SIM2, x2 = ref_out2
    Fidx_to_asu = {i: hkl for hkl,i in SIM2.asu_map_int.items()}
    refined = np.where(SIM2.Fhkl_scales != 1)[0]
    scale_facs = SIM2.Fhkl_scales[refined]
    hkls = [Fidx_to_asu[i] for i in refined]
    new_amps = {}
    for i in refined:
        asu = Fidx_to_asu[i]
        if asu in Fmap:
            scale = SIM2.Fhkl_scales[i]
            new_amp = np.sqrt(scale)*Fmap[asu]
            new_amps[asu] = new_amp
            
    for i_hkl, hkl in enumerate(Finds):
        if hkl in new_amps:
            Famp[i_hkl] = new_amps[hkl]

    F2 = F.customized_copy(data=Famp)
    F2.as_mtz_dataset(column_root_label="F").mtz_object().write("_temp.mtz")
    params.simulator.structure_factors.mtz_name = "_temp.mtz"
    params.simulator.structure_factors.mtz_column = "F(+),SIGF(+),F(-),SIGF(-)"
    params2.simulator.structure_factors.mtz_name = "_temp.mtz"
    params2.simulator.structure_factors.mtz_column = "F(+),SIGF(+),F(-),SIGF(-)"
    Fmap = {h:amp for h,amp in zip(F2.indices(), F2.data())}
    params.filter_during_refinement.enable=False
    params2.filter_during_refinement.enable=False

    model_df = hopper_io.save_to_pandas(x2, Modeler2, SIM2, args.exptName, params2, Expt, 0,
                                        args.reflName, None, 0, write_expt=False, write_pandas=False,
                                        exp_idx=args.exptIdx)

    # REFINEMENT OF DETECTOR
    Expt = detector_refinement(model_df, Expt, params2)

    #new_El = ExperimentList()
    #new_El.append(Expt)
    ##from IPython import embed;embed()
    #new_El.as_file("_geom_ref.expt")
    #Refs.as_file("_geom_ref.refl")
    #Refs['id'] = flex.int(len(Refs),0)
    #model_df["geom_exp"] = "_geom_ref.expt"
    #model_df["geom_ref"] = "_geom_ref.refl"
    #model_df["geom_exp_idx"] = 0
    #model_df.to_pickle("_geom_ref.pkl")
    #from simtbx.diffBragg.refiners import geometry
    #with open("_geom_groups.txt", "w") as o:
    #    for i_p in range(len(Expt.detector)):
    #        o.write("%d %d\n" % (i_p, i_p))
    #params3 = deepcopy(params2)
    #params3.fix.Fhkl=True
    #params3.refiner.panel_group_file="_geom_groups.txt"
    #params3.geometry.fix.panel_rotations=[0,0,0]
    #params3.geometry.fix.panel_translations=[0,0,1]
    #params3.geometry.input_pkl="_geom_ref.pkl"
    #params3.geometry.save_state_freq=100000
    #params3.filter_during_refinement.enable=False
    #params3.outdir="_geom.out"
    #params3.max_process=1
    #params3.geometry.optimize=True
    #geometry.geom_min(params3)
    #new_det = ExperimentList.from_file("_geom.out/diffBragg_detector.expt")[0].detector
    #Expt.detector = new_det


    #params2.outdir=args.outDir
    #os.makedirs(params2.outdir, exist_ok=True)
    #Modeler2.exper_name = args.exptName
    #Modeler2.refl_name = args.reflName
    #Modeler2.save_up(x2, SIM2, rank=0, i_shot=0)
    #Modeler2.clean_up(SIM2)

energies = [utils.ENERGY_CONV / Expt.beam.get_wavelength()]
fluxes = [SIM2.D.flux]
mdl_parm = hopper_utils.get_param_from_x(x2, Modeler2, as_dict=True)
Bragg = diffBragg_forward(
    Expt.crystal, Expt.detector, Expt.beam, F2, 
    energies, fluxes,
    oversample=SIM2.D.oversample, 
    Ncells_abc=(mdl_parm['Na'], mdl_parm['Nb'], mdl_parm['Nc']),
    Ncells_def=(mdl_parm['Nd'], mdl_parm['Ne'], mdl_parm['Nf']),
    beamsize_mm=SIM2.D.beamsize_mm, device_Id=devid,
    spot_scale_override=mdl_parm['scale'],
    cuda=True,
    num_phi_steps=SIM2.D.phisteps, delta_phi=SIM2.D.phistep_deg, 
    spindle_axis=SIM2.D.spindle_axis, no_Nabc_scale=True
)


# compare model to data
CHECKER = roi_check.roiCheck()


from scipy.optimize import minimize
def func(x, CHECKER, bragg_im, bg_im, dat_im):
    bragg_scale = x[0]
    bg_scale = x[1]
    offset = x[2]
    score = CHECKER.score(dat_im, bragg_scale**2*bragg_im + offset+ bg_scale**2*bg_im)
    resid = 1-score
    return resid


import h5py
import numpy as np
data_subims = []
bg_subims = []
bragg_subims = []
opt_bragg_scales = []
opt_bg_scales = []
opt_offsets = []
mask_subims = []
scores = []
for (x_coef, y_coef, offset), pid,  (x1,x2,y1,y2) in zip(tilt_coefs, pids, bbox) :
    Y,X = np.indices((y2-y1, x2-x1))
    bg_im = (X+x1)*x_coef + (Y+y1)*y_coef + offset
    x = slice(x1,x2,1)
    y = slice(y1,y2,1)
    dat_im = data[pid,y, x]
    bragg_im = Bragg[pid, y, x]
    #min_out = minimize(func, x0=[1,1,0], args=(CHECKER, bragg_im, bg_im, dat_im), method="Nelder-Mead")
    #if min_out.success:
    #    opt_bragg_scale = min_out['x'][0]**2
    #    opt_bg_scale = min_out['x'][1]**2
    #    opt_offset = min_out['x'][2]
    #else:
    opt_bragg_scale = 1
    opt_bg_scale = 1
    opt_offset = 0

    mod_im = opt_bg_scale*bg_im + opt_bragg_scale*bragg_im + opt_offset
    score = CHECKER.score(dat_im, mod_im)
    print(score)
    data_subims.append( dat_im)
    bg_subims.append(bg_im)
    bragg_subims.append(bragg_im)

    opt_bragg_scales.append(opt_bragg_scale)
    opt_bg_scales.append(opt_bg_scale)
    opt_offsets.append(opt_offset)

    scores.append(score)

with h5py.File(args.outFile, "w") as h:
    h.create_dataset("score", data=scores)
    h.create_dataset("bragg_scale", data=opt_bragg_scales)
    h.create_dataset("bg_scale", data=opt_bg_scales)
    h.create_dataset("offset", data=opt_offsets)

    for i in range(len(scores)):
        h.create_dataset("data/roi%d" % i, data=data_subims[i])
        h.create_dataset("bragg/roi%d" % i, data=bragg_subims[i])
        h.create_dataset("bg/roi%d" % i, data=bg_subims[i])


print("Average score:", np.mean(scores), "+-", np.std(scores))
