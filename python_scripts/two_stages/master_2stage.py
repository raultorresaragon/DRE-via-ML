# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# master_2stage.py — Run true OTR + DRE estimation for a single dataset (interactive)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from get_true_otr_two_stage import get_otr
from estimate_dre_two_stage import estimate_dre
from OTR_assess import assess_otr

filename = 's2_k2_logit_expo_1'
save     = True   # set to False to skip saving outputs

otr = get_otr(filename)
dre = estimate_dre(filename)

if save:
    import os
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')

    otr.to_csv(os.path.join(datasets_dir, f'{filename}_OTR_test.csv'),  index=False)
    dre.to_csv(os.path.join(datasets_dir, f'{filename}_DRE_test.csv'),  index=False)
    print(f"Saved OTR and DRE results for {filename}")



import glob  
import os
                                                                                                                                                            
script_dir   = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, '../_1trt_effect/2stages/datasets')                                                                               
                                                                                                                                                            
files = glob.glob(os.path.join(datasets_dir, 's2_k2_logit_*.csv'))                                                                                 
# keep only base datasets (exclude _OTR, _DRE, etc.)                                                                                                      
filenames = [os.path.basename(f).replace('.csv', '') 
            for f in files                                                                                                                               
            if not any(s in f for s in ['_OTR', '_DRE', '_assess', '_info'])]                                                                            
                                                                                                                                                            
for filename in sorted(filenames):                                                                                                                      
    otr = get_otr(filename)                                                                                                                               
    dre = estimate_dre(filename)      
    if save:
        otr.to_csv(os.path.join(datasets_dir, f'{filename}_OTR_test.csv'),  index=False)
        dre.to_csv(os.path.join(datasets_dir, f'{filename}_DRE_test.csv'),  index=False)
        res = assess_otr(filename)
        print(f"Saved OTR and DRE results for {filename}")
