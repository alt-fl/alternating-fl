import os
import sys


# needs to run this script in the root folder of Alt-FL
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dlg import DLGAttacker

from exp_wrapper import Wrapper
from exp_args import ExperimentArgument

from logger import logger

logger.info("============Experiment start============")

args = ExperimentArgument()
wrapper = Wrapper(args)
model = wrapper.get_model()
auth_dataset, _, _ = wrapper.get_data_split()

attacker = DLGAttacker(model, auth_dataset, args.num_classes)

attacker.reconstruct(6969, use_idlg=True)
