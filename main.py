import numpy as np

from causalml.inference.meta import XGBTRegressor
from synthetic_data import make_uplift_classification

df, feature_names = make_uplift_classification()

x_control = df[df.treatment_group_key == 'control'][feature_names].values
y_control = df[df.treatment_group_key == 'control'].conversion.values
t_control = np.zeros(y_control.shape)

print('control data')
for name, arr in [('x', x_control), ('y', y_control), ('t', t_control)]:
    print(name, arr.shape)
    print("examples:")
    print(arr[:2])
    print('\n---\n')



x_treatment = df[df.treatment_group_key == 'treatment2'][feature_names].values
y_treatment = df[df.treatment_group_key == 'treatment2'].conversion.values
t_treatment = np.ones(y_control.shape)

print('treatment data')
for name, arr in [('x', x_treatment), ('y', y_treatment), ('t', t_treatment)]:
    print(name, arr.shape)
    print("examples:")
    print(arr[:2])
    print('\n---\n')


x = np.concatenate([x_control, x_treatment], axis=0)
y = np.concatenate([y_control, y_treatment], axis=0)
t = np.concatenate([t_control, t_treatment], axis=0)


print('combined data')
for name, arr in [('x', x), ('y', y), ('t', t)]:
    print(name, arr.shape)
    print("examples:")
    print(arr[:2])
    print('\n---\n')

xg = XGBTRegressor(random_state=42)
te, lb, ub = xg.estimate_ate(x, t, y)
print('Average Treatment Effect (XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te[0], lb[0], ub[0]))

treatment_conversion = df[df.treatment_group_key == 'treatment2'].conversion.mean()
control_conversion = df[df.treatment_group_key == 'control'].conversion.mean()

print(f'from data, treatment_conversion={treatment_conversion}, control_conversion={control_conversion}')
print(f'treatment effect = {treatment_conversion - control_conversion}')