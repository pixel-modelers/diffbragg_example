
from copy import deepcopy
import logging
import os

import numpy as np

from dials.array_family import flex
from dxtbx.model import ExperimentList
from libtbx.phil import parse
from simtbx.command_line.hopper import phil_scope
from simtbx.diffBragg import hopper_utils, utils, hopper_io
from simtbx.modeling.forward_models import diffBragg_forward

# TODO : replace the generated files herein with generic tempfiles

def detector_refinement(model_df, Expt, Refs, params):
    new_El = ExperimentList()
    new_El.append(Expt)
    new_El.as_file("_geom_ref.expt")
    Refs.as_file("_geom_ref.refl")
    Refs['id'] = flex.int(len(Refs), 0)
    model_df["geom_exp"] = "_geom_ref.expt"
    model_df["geom_ref"] = "_geom_ref.refl"
    model_df["geom_exp_idx"] = 0
    model_df.to_pickle("_geom_ref.pkl")
    from simtbx.diffBragg.refiners import geometry
    with open("_geom_groups.txt", "w") as o:
        for i_p in range(len(Expt.detector)):
            o.write("%d %d\n" % (i_p, i_p))
    params_geom = deepcopy(params)
    params_geom.fix.Fhkl = True
    params_geom.refiner.panel_group_file = "_geom_groups.txt"
    params_geom.geometry.fix.panel_rotations = [0, 0, 0]
    params_geom.geometry.fix.panel_translations = [0, 0, 1]
    params_geom.geometry.input_pkl = "_geom_ref.pkl"
    params_geom.geometry.save_state_freq = 100000
    params_geom.filter_during_refinement.enable = False
    params_geom.outdir = "_geom.out"
    params_geom.max_process = 1
    params_geom.geometry.optimize = True
    geometry.geom_min(params_geom)
    new_det = ExperimentList.from_file("_geom.out/diffBragg_detector.expt")[0].detector
    Expt.detector = new_det
    return Expt


def run_diffbragg(data_load, devId=0, num_macro=5):
    """
    :param data_load: instance of diffbragg_example.data_load.DataLoad
    :param devId: CUDA gpu device Id
    :param num_macro: number of refinement macro cycles
    :return:
    """

    logger = logging.getLogger("diffBragg.main")
    logger.setLevel(logging.DEBUG)

    # parameters object that controls hopper refinement

    xtal_refine_phil= os.path.dirname(__file__) + "/dbconfig/xtal_refine.phil"
    fhkl_refine_phil= os.path.dirname(__file__) + "/dbconfig/fhkl_refine.phil"
    config = open(xtal_refine_phil, "r").read()
    config2 = open(fhkl_refine_phil, "r").read()
    params = phil_scope.fetch(sources=[parse(config)]).extract()
    params.fix.Ndef = False
    params_fhkl = phil_scope.fetch(sources=[parse(config2)]).extract()
    for prm in (params, params_fhkl):
        prm.simulator.structure_factors.mtz_name = data_load.args.mtzFile
        prm.simulator.structure_factors.mtz_column = data_load.args.mtzCol
        prm.roi.hotpixel_mask = data_load.args.maskFile

    Expt = deepcopy(data_load.Expt)
    Finds = data_load.F.indices()
    Famps = data_load.F.data()
    FMap = {h: amp for h, amp in zip(Finds, Famps)}
    for cycle in range(num_macro):

        # REFINEMENT OF XTAL
        ref_out = hopper_utils.refine(Expt, data_load.Refs, params, return_modeler=True, free_mem=True, gpu_device=devId)
        Expt, _, Modeler, SIM, x = ref_out
        mdl_parm = hopper_utils.get_param_from_x(x, Modeler, as_dict=True)

        for prm in [params, params_fhkl]:
            prm.init.Nabc = mdl_parm['Na'], mdl_parm['Nb'], mdl_parm['Nc']
            prm.init.Ndef = mdl_parm['Nd'], mdl_parm['Ne'], mdl_parm['Nf']
            prm.init.G = mdl_parm['scale']

        # REFINEMENT OF FHKL
        ref_out_fhkl = hopper_utils.refine(Expt, data_load.Refs, params_fhkl, return_modeler=True, free_mem=True, gpu_device=devId)
        _, _, Modeler_fhkl, SIM_fhkl, x_hkl = ref_out_fhkl
        Fidx_to_asu = {i: hkl for hkl, i in SIM_fhkl.asu_map_int.items()}
        refined = np.where(SIM_fhkl.Fhkl_scales != 1)[0]
        new_amps = {}
        for i in refined:
            asu = Fidx_to_asu[i]
            if asu in FMap:
                scale = SIM_fhkl.Fhkl_scales[i]
                new_amp = np.sqrt(scale) * FMap[asu]
                new_amps[asu] = new_amp

        for i_hkl, hkl in enumerate(Finds):
            if hkl in new_amps:
                Famps[i_hkl] = new_amps[hkl]

        Fopt = data_load.F.customized_copy(data=Famps)
        Fopt.as_mtz_dataset(column_root_label="F").mtz_object().write("_temp.mtz")
        for prm in [params, params_fhkl]:
            prm.simulator.structure_factors.mtz_name = "_temp.mtz"
            prm.simulator.structure_factors.mtz_column = "F(+),SIGF(+),F(-),SIGF(-)"
        FMap = {h: amp for h, amp in zip(Fopt.indices(), Fopt.data())}
        params.filter_during_refinement.enable = False
        params_fhkl.filter_during_refinement.enable = False

        model_df = hopper_io.save_to_pandas(x_hkl, Modeler_fhkl, SIM_fhkl, data_load.args.exptName, params_fhkl, Expt, 0,
                                            data_load.args.reflName, None, 0, write_expt=False, write_pandas=False,
                                            exp_idx=data_load.args.exptIdx)

        # REFINEMENT OF DETECTOR
        Expt = detector_refinement(model_df, Expt, data_load.Refs, params_fhkl)

    # with te optimized model from diffBragg, we can predict a model value at every pixel
    energies = [utils.ENERGY_CONV / Expt.beam.get_wavelength()]
    fluxes = [SIM_fhkl.D.flux]
    mdl_parm = hopper_utils.get_param_from_x(x_hkl, Modeler_fhkl, as_dict=True)
    Bragg = diffBragg_forward(
        Expt.crystal, Expt.detector, Expt.beam, Fopt,
        energies, fluxes,
        oversample=SIM_fhkl.D.oversample,
        Ncells_abc=(mdl_parm['Na'], mdl_parm['Nb'], mdl_parm['Nc']),
        Ncells_def=(mdl_parm['Nd'], mdl_parm['Ne'], mdl_parm['Nf']),
        beamsize_mm=SIM_fhkl.D.beamsize_mm, device_Id=devId,
        spot_scale_override=mdl_parm['scale'],
        cuda=True,
        num_phi_steps=SIM_fhkl.D.phisteps, delta_phi=SIM_fhkl.D.phistep_deg,
        spindle_axis=SIM_fhkl.D.spindle_axis, no_Nabc_scale=True
    )
    return Bragg

