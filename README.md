# Confound Isolating

## Confound_prediction is a Python module to control confound effect in the prediction or classification model.

&nbsp;&nbsp; Any successful prediction model may be driven by a confounding effect that is correlated with the effect of interest. It is important to control that detected associations are not driven by unwanted effects. It is common issue in in neuroscience, epidemiology, economy, agriculture, etc. 


We  introduce a non-parametric approach, named *“confound-isolating cross-validation”*, to control for a confounding effect in a predictive
model. It is based on crafting a test set on which the effect of interest is independent from the confounding effect. 


What expect from Confound_prediction?

Developed framework is based on anti mutual information sampling, a
novel sampling approach to create a test set in which the effect
of interest is independent from the confounding effect.




Check list:
Variables:
X
y
Confound
z

Sampling size
number of samples to be removed

Return
x-test
x_train

Another methods.
We also provide 2 state of the art in neuroscience deconfounding techniques:
1.
2.



References
