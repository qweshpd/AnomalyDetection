# -*- coding: utf-8 -*-

import logging


logger = logging.getLogger("EnumAnalysis")
logger.setLevel(logging.ERROR)
formatter = logging.Formatter("[%(asctime)-15s] - %(levelname)-8s - %(message)s")

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)

fh = logging.FileHandler("EnumAnalysis.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(sh)



__all__ = ["EnumAnalysis"]