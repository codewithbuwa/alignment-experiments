"""
Author: Jordan Kevin Buwa Mbouobda
Purpose: Run grouped single-Gaussian sensitivity experiments.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.dpo_kto_1d.data_sensitivity import main


if __name__ == "__main__":
    main()
