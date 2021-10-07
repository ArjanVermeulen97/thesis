# -*- coding: utf-8 -*-
"""
Spyder Editor

Demonstration of what the physical and coding layer s
"""
from survey_opt import do_survey
import matplotlib.pyplot as plt

res = do_survey([0.8867, 0, 0.75])
plt.plot(res)