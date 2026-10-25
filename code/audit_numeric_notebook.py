"""Run existing numerical notebook examples without plotting or changing outputs.

Run: MPLCONFIGDIR=/tmp/fwm-matplotlib code/.venv/bin/python code/audit_numeric_notebook.py
Executes the existing 500/1000/1000-point grids and DOP853 integrations.
Add --figures to regenerate the two manuscript plot PDFs.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

code_dir=Path(__file__).resolve().parent
os.chdir(code_dir)
os.environ.setdefault('MPLBACKEND','Agg')
sys.path.insert(0,str(code_dir))
nb=json.loads((code_dir/'Numeric plots for four-wave mixing.ipynb').read_text())
ns={'__name__':'__audit__'}
selected=[1,2,*range(4,23),24,*range(26,35),*range(40,44),*range(46,55),*range(61,71)]
for i in selected:
    src='\n'.join(line for line in ''.join(nb['cells'][i]['source']).splitlines()
                  if not line.lstrip().startswith('%') and not line.startswith('from IPython.')
                  and not line.startswith('InteractiveShell.'))
    print('Executing notebook cell',i,flush=True)
    exec(compile(src,f'Numeric plots for four-wave mixing.ipynb:cell-{i}','exec'),ns)
    if i==34:
        ana=np.concatenate([ns['case1_kron_sol_mpmath']['u_j_evals'],ns['case1_kron_sol_mpmath']['v_j_evals']])
        err=float(np.max(np.abs(ana-ns['case1_sol_scipy'])))
        assert ns['case1_solution'].success
        scale=max(1,float(np.max(np.abs(ns['case1_sol_scipy']))))
        print('CASE 1 field scale',scale,'scaled error',err/scale,flush=True)
        assert err/scale<1e-9,err/scale
        print('CASE 1: all eight complex envelopes, max absolute discrepancy',err,flush=True)
        if '--figures' in sys.argv:
            exec(compile(''.join(nb['cells'][37]['source']),'plot-cell-37','exec'),ns)
    if i==54:
        ana=np.array(ns['case2_As_ana']+ns['case2_Acs_ana'])
        num=ns['case2_A_solution'].y
        scale=float(np.max(np.abs(num)))
        assert scale>0
        err=float(np.max(np.abs(ana-num)))/scale
        beta_values=np.array([float(ns['beta'][j].subs(ns['case2_fwm_params'])) for j in range(1,5)])
        mismatch=float(-beta_values[0]-beta_values[1]+beta_values[2]+beta_values[3])
        assert abs(mismatch)>1e-6,mismatch
        assert np.max(ns['P_freq_inits'])<=10.0
        assert ns['case2_show_n_periods']==2
        print('CASE 2: exchange period [m]',ns['case2_z_end']/2,flush=True)
        print('CASE 2: Delta k [1/m]',mismatch,'length [m]',ns['case2_z_end'],flush=True)
        assert ns['case2_A_solution'].success
        assert err<1e-9,err
        print('CASE 2: physical fields, max discrepancy / max field amplitude',err,flush=True)
        if '--figures' in sys.argv:
            exec(compile(''.join(nb['cells'][56]['source']),'plot-cell-56','exec'),ns)
    if i==70:
        # Reconstruct the four independent physical fields from the cubic case.
        vals=ns['case3_kron_sol_mpmath']
        ana=np.array([vals['u_j_evals'][j]*ns['case3_u_v_norms_to_A'][j] for j in range(2)]
                     +[vals['v_j_evals'][j]*ns['case3_u_v_norms_to_A'][j+2] for j in range(2)])
        ana=ana[[0,3,2,1]]
        num=ns['case3_A_solution'].y
        scale=float(np.max(np.abs(num)))
        assert scale>0
        err=float(np.max(np.abs(ana-num)))/scale
        d4=ns['d'][4].subs([e.args for e in ns['case3_init_cons_params']['params']])
        assert d4==0,d4
        assert ns['case3_A_solution'].success
        assert err<1e-9,err
        print('CASE 3: cubic reduction d4=0, max discrepancy / max field amplitude',err,flush=True)
print('PASS: existing abstract, physical and cubic notebook examples')
