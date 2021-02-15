import sys
import numpy as np
import eval_metrics as em
import matplotlib.pyplot as plt

# Replace CM scores with your own scores or provide score file as the first argument.
cm_scores_file =  'Eval_scores_file.txt'
# Replace ASV scores with organizers' scores or provide score file as the second argument.
asv_score_file = '/tDCF_python/ASV_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt'

args = sys.argv
if len(args) > 1:
    if len(args) != 3:
        print('USAGE: python evaluate_tDCF_asvspoof19.py <cm_scoresFILE> <ASV_SCOREFILE>')
        exit()
    else:
        cm_scores_file = args[1]
        asv_score_file = args[2]

# Fix tandem detection cost function (t-DCF) parameters
Pspoof = 0.05
cost_model = {
    'Pspoof': Pspoof,  # Prior probability of a spoofing attack
    'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
    'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
    'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
    'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
    'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
    'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
}

# Load organizers' ASV scores
asv_data = np.genfromtxt(asv_score_file, dtype=str)
asv_sources = asv_data[:, 0]
asv_keys = asv_data[:, 1]
asv_scores = asv_data[:, 2].astype(np.float)

# Load CM scores
cm_data = np.genfromtxt(cm_scores_file, dtype=str)
cm_utt_id = cm_data[:, 0]
cm_sources = cm_data[:, 1]
cm_keys = cm_data[:, 2]
cm_scores = cm_data[:, 3].astype(np.float)

# Extract target, nontarget, and spoof scores from the ASV scores
tar_asv = asv_scores[asv_keys == 'target']
non_asv = asv_scores[asv_keys == 'nontarget']
spoof_asv = asv_scores[asv_keys == 'spoof']

# Extract bona fide (real human) and spoof scores from the CM scores
bona_cm = cm_scores[cm_keys == 'human']
spoof_cm = cm_scores[cm_keys == 'spoof']

# EERs of the standalone systems and fix ASV operating point to EER threshold
eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

spoof_cm_A07    = cm_scores[cm_sources == 'A07']
spoof_cm_A08    = cm_scores[cm_sources == 'A08']
spoof_cm_A09    = cm_scores[cm_sources == 'A09']
spoof_cm_A10    = cm_scores[cm_sources == 'A10']
spoof_cm_A11    = cm_scores[cm_sources == 'A11']
spoof_cm_A12    = cm_scores[cm_sources == 'A12']
spoof_cm_A13    = cm_scores[cm_sources == 'A13']
spoof_cm_A14    = cm_scores[cm_sources == 'A14']
spoof_cm_A15    = cm_scores[cm_sources == 'A15']
spoof_cm_A16    = cm_scores[cm_sources == 'A16']
spoof_cm_A17    = cm_scores[cm_sources == 'A17']
spoof_cm_A18    = cm_scores[cm_sources == 'A18']
spoof_cm_A19    = cm_scores[cm_sources == 'A19']
    
eer_cm_A07 = em.compute_eer(bona_cm, spoof_cm_A07)[0]
eer_cm_A08 = em.compute_eer(bona_cm, spoof_cm_A08)[0]
eer_cm_A09 = em.compute_eer(bona_cm, spoof_cm_A09)[0]
eer_cm_A10 = em.compute_eer(bona_cm, spoof_cm_A10)[0]
eer_cm_A11 = em.compute_eer(bona_cm, spoof_cm_A11)[0]
eer_cm_A12 = em.compute_eer(bona_cm, spoof_cm_A12)[0]
eer_cm_A13 = em.compute_eer(bona_cm, spoof_cm_A13)[0]
eer_cm_A14 = em.compute_eer(bona_cm, spoof_cm_A14)[0]
eer_cm_A15 = em.compute_eer(bona_cm, spoof_cm_A15)[0]
eer_cm_A16 = em.compute_eer(bona_cm, spoof_cm_A16)[0]
eer_cm_A17 = em.compute_eer(bona_cm, spoof_cm_A17)[0]
eer_cm_A18 = em.compute_eer(bona_cm, spoof_cm_A18)[0]
eer_cm_A19 = em.compute_eer(bona_cm, spoof_cm_A19)[0]

[Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)


# Compute t-DCF
tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)

# Minimum t-DCF
min_tDCF_index = np.argmin(tDCF_curve)
min_tDCF = tDCF_curve[min_tDCF_index]


print('ASV SYSTEM')
print('   EER            = {:8.5f} % (Equal error rate (target vs. nontarget discrimination)'.format(eer_asv * 100))
print('   Pfa            = {:8.5f} % (False acceptance rate of nontargets)'.format(Pfa_asv * 100))
print('   Pmiss          = {:8.5f} % (False rejection rate of targets)'.format(Pmiss_asv * 100))
print('   1-Pmiss,spoof  = {:8.5f} % (Spoof false acceptance rate)'.format((1 - Pmiss_spoof_asv) * 100))

print('\nCM SYSTEM')
print('   EER            = {:8.9f} % (Equal error rate for countermeasure)'.format(eer_cm * 100))

print('\nTANDEM')
print('   min-tDCF       = {:8.9f}'.format(min_tDCF))

print('BREAKDOWN CM SYSTEM')
print('   EER A07          = {:8.9f} % (Equal error rate for A07)'.format(eer_cm_A07 * 100))
print('   EER A08          = {:8.9f} % (Equal error rate for A08)'.format(eer_cm_A08 * 100))
print('   EER A09          = {:8.9f} % (Equal error rate for A09)'.format(eer_cm_A09 * 100))
print('   EER A10          = {:8.9f} % (Equal error rate for A10)'.format(eer_cm_A10 * 100))
print('   EER A11          = {:8.9f} % (Equal error rate for A11)'.format(eer_cm_A11 * 100))
print('   EER A12          = {:8.9f} % (Equal error rate for A12)'.format(eer_cm_A12 * 100))
print('   EER A13          = {:8.9f} % (Equal error rate for A13)'.format(eer_cm_A13 * 100))
print('   EER A14          = {:8.9f} % (Equal error rate for A14)'.format(eer_cm_A14 * 100))
print('   EER A15          = {:8.9f} % (Equal error rate for A15)'.format(eer_cm_A15 * 100))
print('   EER A16          = {:8.9f} % (Equal error rate for A16)'.format(eer_cm_A16 * 100))
print('   EER A17          = {:8.9f} % (Equal error rate for A17)'.format(eer_cm_A17 * 100))
print('   EER A18          = {:8.9f} % (Equal error rate for A18)'.format(eer_cm_A18 * 100))
print('   EER A19          = {:8.9f} % (Equal error rate for A19)'.format(eer_cm_A19 * 100))


# Visualize ASV scores and CM scores
plt.figure()
ax = plt.subplot(121)
plt.hist(tar_asv, histtype='step', density=True, bins=50, label='Target')
plt.hist(non_asv, histtype='step', density=True, bins=50, label='Nontarget')
plt.hist(spoof_asv, histtype='step', density=True, bins=50, label='Spoof')
plt.plot(asv_threshold, 0, 'o', markersize=10, mfc='none', mew=2, clip_on=False, label='EER threshold')
plt.legend()
plt.xlabel('ASV score')
plt.ylabel('Density')
plt.title('ASV score histogram')

ax = plt.subplot(122)
plt.hist(bona_cm, histtype='step', density=True, bins=50, label='Bona fide')
plt.hist(spoof_cm, histtype='step', density=True, bins=50, label='Spoof')
plt.legend()
plt.xlabel('CM score')
#plt.ylabel('Density')
plt.title('CM score histogram')


# Plot t-DCF as function of the CM threshold.
plt.figure()
plt.plot(CM_thresholds, tDCF_curve)
plt.plot(CM_thresholds[min_tDCF_index], min_tDCF, 'o', markersize=10, mfc='none', mew=2)
plt.xlabel('CM threshold index (operating point)')
plt.ylabel('Norm t-DCF')
plt.title('Normalized tandem t-DCF')
plt.plot([np.min(CM_thresholds), np.max(CM_thresholds)], [1, 1], '--', color='black')
plt.legend(('t-DCF', 'min t-DCF ({:.9f})'.format(min_tDCF), 'Arbitrarily bad CM (Norm t-DCF=1)'))
plt.xlim([np.min(CM_thresholds), np.max(CM_thresholds)])
plt.ylim([0, 1.5])

plt.show()
