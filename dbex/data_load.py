

from simtbx.diffBragg import utils
from dials.array_family import flex
from dxtbx.model import ExperimentList


class DataLoad:
    """
    A utility class for loading and preparing crystallographic data necessary
    for differential Bragg analysis (or similar processing).

    This class reads data from MTZ, DIALS Experiment List, and DIALS
    Reflection Table files, and performs initial data preparation, such as
    generating Bijvoet mates and extracting raw image data for a specific
    experiment.
    """

    def __init__(self, args):
        """
        Initializes the DataLoad object by parsing input files and extracting
        data for a specified experiment.

        :param args: An object or namespace containing configuration arguments.
                     Required attributes include:
                     - ``mtzFile``: Path to the MTZ file.
                     - ``mtzCol``: Column name for structure factors in the MTZ file.
                     - ``exptName``: Path to the DIALS Experiment List file.
                     - ``exptIdx``: Index of the experiment to load.
                     - ``reflName``: Path to the DIALS Reflection Table file.
        :type args: object
        """
        self.args = args

        # --- Structure Factor Data ---
        F = utils.open_mtz(args.mtzFile, args.mtzCol)
        self.F = F.generate_bijvoet_mates()
        """
        A :py:class:`cctbx.miller.array` containing the structure factor
        amplitudes ($|F|$) including generated **Bijvoet mates**.
        """

        # --- Experiment List and Reflection Table ---
        # Select the specific experiment
        self.Expt = ExperimentList.from_file(args.exptName)[args.exptIdx]
        """
        A single :py:class:`dxtbx.model.Experiment` object for the experiment
        specified by ``args.exptIdx``.
        """

        # Select reflections belonging to the specific experiment
        Refs = flex.reflection_table.from_file(args.reflName)
        self.Refs = Refs.select(Refs['id'] == args.exptIdx)
        """
        A filtered :py:class:`flex.reflection_table` containing only reflections
        associated with the current experiment ID.
        """

        # --- Pixel Data and Background Estimates ---
        # Get the raw image data from the experiment
        self.data = utils.image_data_from_expt(self.Expt)
        """
        A :py:class:`flex.double` array of the raw image pixel values.
        """

        # Get the pixel data and background estimates near the reflections
        self.bbox, self.pids, self.tilt_coefs, self.bg_is_good, self.background_image = \
            utils.get_roi_background_and_selection_flags(
                self.Refs, self.data, use_robust_estimation=False, shoebox_sz=12,
                reject_roi_with_hotpix=False,
                pad_for_background_estimation=3, hotpix_mask=self.data < 0, weighted_fit=False)
        #self.bbox
        #"""
        #A list/array of the **bounding boxes** (regions of interest) for each
        #reflection on the detector.
        #"""
        #self.pids
        #"""
        #An array of **panel IDs** indicating which detector panel each reflection
        #is located on.
        #"""
        #self.tilt_coefs
        #"""
        #The **coefficients** (e.g., from a plane fit) used for the background
        #estimation model near the reflections.
        #"""
        #self.bg_is_good
        #"""
        #A **boolean array** flagging whether the background estimation was
        #considered successful/reliable for each reflection.
        #"""
        #self.background_image
        #"""
        #A representation of the **estimated background** pixels for the regions
        #of interest.
        #"""




#from simtbx.diffBragg import utils
#from dials.array_family import flex
#from dxtbx.model import ExperimentList
#
##
#class DataLoad:
#
#    def __init__(self, args):
#        self.args = args
#        F = utils.open_mtz(args.mtzFile, args.mtzCol)
#        self.F = F.generate_bijvoet_mates()
#        self.Finds = self.F.indices()
#        self.Famp = self.F.data()
#        self.Fmap = {h: amp for h, amp in zip(self.Finds, self.Famp)}
#
#        # experiment list and reflection table
#        self.Expt = ExperimentList.from_file(args.exptName)[args.exptIdx]
#
#        Refs = flex.reflection_table.from_file(args.reflName)
#        self.Refs = Refs.select(Refs['id'] == args.exptIdx)
#
#        # get the pixel data and background estimates near the reflections
#        self.data = utils.image_data_from_expt(self.Expt)
#        self.bbox, self.pids, self.tilt_coefs, self.bg_is_good, self.background_image = \
#            utils.get_roi_background_and_selection_flags(
#                Refs, self.data, use_robust_estimation=False, shoebox_sz=12,
#                reject_roi_with_hotpix=False,
#                pad_for_background_estimation=3, hotpix_mask=self.data < 0, weighted_fit=False)
