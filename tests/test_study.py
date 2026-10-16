"""CPU checks for paired statistics and all 15 unique assignments."""
import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('confidence_study',Path(__file__).resolve().parents[1]/'scripts/study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
def test_paired_statistics():
 result=study.interval(np.zeros((5,12)))
 assert result['mean_pp']==0 and result['interval_97_5_pp']==[0,0]
 result=study.interval(np.ones((5,12))*.1)
 assert np.isclose(result['mean_pp'],10)
def test_assignments():
 jobs=[(list(study.STYLES)[(t%3+t//3)%3],study.SEEDS[t//3]) for t in range(15)]
 assert len(set(jobs))==15
 assert all(sum(s==style for s,_ in jobs)==5 for style in study.STYLES)
