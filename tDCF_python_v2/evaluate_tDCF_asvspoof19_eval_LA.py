# example of use of the evaluate_tDCF_asvspoof19 function using
# the ASVspoof 2019 official baseline contermeasure systems.
# (score files stored in folder "scores")
import sys

from evaluate_tDCF_asvspoof19 import evaluate_tDCF_asvspoof19

# set here the track and baseline system to use
track = 'LA';   
args = sys.argv


if 'dev'==args[1]:
    
    ASV_SCOREFILE = 'tDCF_python_v2/scores/ASVspoof2019_' + track + '_dev_asv_scores.txt'
    CM_SCOREFILE = args[2]
    evaluate_tDCF_asvspoof19(CM_SCOREFILE, ASV_SCOREFILE, False);
     

elif 'Eval'==args[1]:
    ASV_SCOREFILE = 'tDCF_python_v2/scores/ASVspoof2019_' + track + '_eval_asv_scores.txt'
    CM_SCOREFILE = args[2]
    evaluate_tDCF_asvspoof19(CM_SCOREFILE, ASV_SCOREFILE, False);  
