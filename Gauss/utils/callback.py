# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, Citic-Lab. All rights reserved.
# Authors: citic-lab
from Gauss.utils.Logger import logger


def multiprocess_callback(subprocess_result):
    logger.info("flags: " + str(subprocess_result))
    return subprocess_result
