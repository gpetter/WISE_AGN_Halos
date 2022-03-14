import lifelines
from lifelines.utils import median_survival_times
import numpy as np


def km_median(values, censored, censorship='upper', return_errors=False):
	kmf = lifelines.KaplanMeierFitter(alpha=0.32)
	if censorship == 'upper':
		kmf.fit_left_censoring(values, np.logical_not(censored))
		if return_errors:
			interval = median_survival_times(kmf.confidence_interval_)
			surv_median = kmf.median_survival_time_
			lowererr = np.float32(interval['KM_estimate_lower_0.68'] - surv_median)
			uppererr = np.float32(surv_median - interval['KM_estimate_upper_0.68'])
			return surv_median, lowererr[0], uppererr[0]
		else:
			return kmf.median_survival_time_
	elif censorship == 'lower':
		kmf.fit(values, censored)
		if return_errors:
			interval = median_survival_times(kmf.confidence_interval_)
			surv_median = kmf.median_survival_time_
			lowererr = surv_median - interval['KM_estimate_lower_0.68']
			uppererr = interval['KM_estimate_upper_0.68'] - surv_median
			return kmf.median_survival_time_, lowererr, uppererr
		else:
			return kmf.median_survival_time_
	else:
		print('error')
		return
