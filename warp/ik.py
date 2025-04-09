from dataclasses import dataclass, field
import math
import os
import logging
import time

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

log = logging.getLogger(__name__) 