# -*- coding: utf-8 -*-
"""
Spyder Editor

Demonstration of what the physical and coding layer s
"""
from survey_opt import do_survey
import matplotlib.pyplot as plt

res = do_survey([1.3       , 0.5       , 0.62831853])
plt.plot(res)