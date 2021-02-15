function evaluate_tDCF_asvspoof19_v1(cm_key, cm_attacks, cm_score, trial)

% Set t-DCF parameters
cost_model.Pspoof       = 0.05;  % Prior probability of a spoofing attack
cost_model.Ptar         = (1 - cost_model.Pspoof) * 0.99; % Prior probability of target speaker
cost_model.Pnon         = (1 - cost_model.Pspoof) * 0.01; % Prior probability of nontarget speaker
cost_model.Cmiss_asv    = 1;     % Cost of ASV system falsely rejecting target speaker
cost_model.Cfa_asv      = 10;    % Cost of ASV system falsely accepting nontarget speaker
cost_model.Cmiss_cm     = 1;     % Cost of CM system falsely rejecting target speaker
cost_model.Cfa_cm       = 10;    % Cost of CM system falsely accepting spoof

% Load organizer's ASV scores
if strcmp(trial,'dev')
    [~, asv_key, asv_score] = textread('/medias/speech/projects/tak/tools/CQCC_analisys_toBECopied/ASV_spoof_2019/Subband_analysis/tDCF_v1/tDCF_v1/scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt', '%s %s %f');
elseif strcmp(trial,'eval')
    [~, asv_key, asv_score] = textread('/medias/speech/projects/tak/tools/CQCC_analisys_toBECopied/ASV_spoof_2019/Subband_analysis/tDCF_v1/tDCF_v1/scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt', '%s %s %f');
end

% Load CM scores (replace these with the scores of your detectors).
% [cm_utt_id, cm_attacks, cm_key, cm_score] = textread(CM_SCOREFILE, '%s %s %s %f');

% Extract target, nontarget and spoof scores from the ASV scores
tar_asv     = asv_score(strcmp(asv_key, 'target'))';
non_asv     = asv_score(strcmp(asv_key, 'nontarget'))';
spoof_asv   = asv_score(strcmp(asv_key, 'spoof'))';

% Extract bona fide (real human) and spoof scores from the CM scores
bona_cm     = cm_score(strcmp(cm_key, 'human'));
spoof_cm    = cm_score(strcmp(cm_key, 'spoof'));

% Fix ASV operating point to EER threshold
[eer_asv, asv_threshold] = compute_eer(tar_asv, non_asv);

% % Visualize ASV and CM scores
% figure(1);
% subplot(121);
% [h, xi] = ksdensity(tar_asv);   plot(xi, h, 'gr'); hold on;
% [h, xi] = ksdensity(non_asv);   plot(xi, h, 'k');
% [h, xi] = ksdensity(spoof_asv); plot(xi, h, 're');
% h = plot(asv_threshold, 0, 'mo'); set(h, 'MarkerSize', 10);
% legend('Target', 'Nontarget', 'Spoof', 'EER threshold');
% xlabel('ASV score'); ylabel('PDF');
% title('ASV scores', 'FontSize', 12, 'FontWeight', 'bold');
% 
% subplot(122);
% [h, xi] = ksdensity(bona_cm);   plot(xi, h, 'gr'); hold on;
% [h, xi] = ksdensity(spoof_cm);  plot(xi, h, 're');
% legend('Bona fide', 'Spoof');
% xlabel('CM score'); ylabel('PDF');
% title('CM scores', 'FontSize', 12, 'FontWeight', 'bold');

% Obtain the detection error rates of the ASV system
[Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold);

% Equal error rate of the countermeasure
[eer_cm, ~] = compute_eer(bona_cm, spoof_cm);

if strcmp(trial,'dev')
    
    spoof_cm_A01    = cm_score(strcmp(cm_attacks, 'A01'));
    spoof_cm_A02    = cm_score(strcmp(cm_attacks, 'A02'));
    spoof_cm_A03    = cm_score(strcmp(cm_attacks, 'A03'));
    spoof_cm_A04    = cm_score(strcmp(cm_attacks, 'A04'));
    spoof_cm_A05    = cm_score(strcmp(cm_attacks, 'A05'));
    spoof_cm_A06    = cm_score(strcmp(cm_attacks, 'A06'));
    
    eer_cm_A01 = compute_eer(bona_cm, spoof_cm_A01);
    eer_cm_A02 = compute_eer(bona_cm, spoof_cm_A02);
    eer_cm_A03 = compute_eer(bona_cm, spoof_cm_A03);
    eer_cm_A04 = compute_eer(bona_cm, spoof_cm_A04);
    eer_cm_A05 = compute_eer(bona_cm, spoof_cm_A05);
    eer_cm_A06 = compute_eer(bona_cm, spoof_cm_A06);
    
elseif strcmp(trial,'eval')
    
    spoof_cm_A07    = cm_score(strcmp(cm_attacks, 'A07'));
    spoof_cm_A08    = cm_score(strcmp(cm_attacks, 'A08'));
    spoof_cm_A09    = cm_score(strcmp(cm_attacks, 'A09'));
    spoof_cm_A10    = cm_score(strcmp(cm_attacks, 'A10'));
    spoof_cm_A11    = cm_score(strcmp(cm_attacks, 'A11'));
    spoof_cm_A12    = cm_score(strcmp(cm_attacks, 'A12'));
    spoof_cm_A13    = cm_score(strcmp(cm_attacks, 'A13'));
    spoof_cm_A14    = cm_score(strcmp(cm_attacks, 'A14'));
    spoof_cm_A15    = cm_score(strcmp(cm_attacks, 'A15'));
    spoof_cm_A16    = cm_score(strcmp(cm_attacks, 'A16'));
    spoof_cm_A17    = cm_score(strcmp(cm_attacks, 'A17'));
    spoof_cm_A18    = cm_score(strcmp(cm_attacks, 'A18'));
    spoof_cm_A19    = cm_score(strcmp(cm_attacks, 'A19'));
    
    eer_cm_A07 = compute_eer(bona_cm, spoof_cm_A07);
    eer_cm_A08 = compute_eer(bona_cm, spoof_cm_A08);
    eer_cm_A09 = compute_eer(bona_cm, spoof_cm_A09);
    eer_cm_A10 = compute_eer(bona_cm, spoof_cm_A10);
    eer_cm_A11 = compute_eer(bona_cm, spoof_cm_A11);
    eer_cm_A12 = compute_eer(bona_cm, spoof_cm_A12);
    eer_cm_A13 = compute_eer(bona_cm, spoof_cm_A13);
    eer_cm_A14 = compute_eer(bona_cm, spoof_cm_A14);
    eer_cm_A15 = compute_eer(bona_cm, spoof_cm_A15);
    eer_cm_A16 = compute_eer(bona_cm, spoof_cm_A16);
    eer_cm_A17 = compute_eer(bona_cm, spoof_cm_A17);
    eer_cm_A18 = compute_eer(bona_cm, spoof_cm_A18);
    eer_cm_A19 = compute_eer(bona_cm, spoof_cm_A19);
end

% Compute t-DCF
[tDCF_curve, CM_thresholds] = compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, true);

% Minimum normalized t-DCF and the corresponding threshold
[min_tDCF, argmin_tDCF] = min(tDCF_curve);
min_tDCF_threshold = CM_thresholds(argmin_tDCF);

fprintf('ASV SYSTEM\n');
fprintf('     EER               \t= %5.5f %%\t\t Equal error rate (target vs. nontarget discrimination)\n',   100 * eer_asv);
fprintf('     Pfa               \t= %5.5f %%\t\t False acceptance rate of nontargets\n',   100 * Pfa_asv);
fprintf('     Pmiss             \t= %5.5f %%\t\t Miss (false rejection) rate of targets\n',   100 * Pmiss_asv);
fprintf('     1-Pmiss,spoof     \t= %5.5f %%\t Spoof false acceptance rate ("NOT miss spoof trial")\n\n', 100 * (1 - Pmiss_spoof_asv));

fprintf('CM SYSTEM\n');
fprintf('     EER pooled        \t= %f %%\t Equal error rate from CM scores pooled across all attacks. \n\n', 100 * eer_cm);

fprintf('TANDEM\n');
fprintf('     min-tDCF          \t= %f\n\n',   min_tDCF);

if strcmp(trial,'dev')
    fprintf('BREAKDOWN CM SYSTEM\n');
    fprintf('     EER A01           \t= %f %%\t Equal error rate from CM scores A01 attack. \n', 100 * eer_cm_A01);
    fprintf('     EER A02           \t= %f %%\t Equal error rate from CM scores A02 attack. \n', 100 * eer_cm_A02);
    fprintf('     EER A03           \t= %f %%\t Equal error rate from CM scores A03 attack. \n', 100 * eer_cm_A03);
    fprintf('     EER A04           \t= %f %%\t Equal error rate from CM scores A04 attack. \n', 100 * eer_cm_A04);
    fprintf('     EER A05           \t= %f %%\t Equal error rate from CM scores A05 attack. \n', 100 * eer_cm_A05);
    fprintf('     EER A06           \t= %f %%\t Equal error rate from CM scores A06 attack. \n', 100 * eer_cm_A06);
elseif strcmp(trial,'eval')
    fprintf('BREAKDOWN CM SYSTEM\n');
    fprintf('     EER A07           \t= %f %%\t Equal error rate from CM scores A07 attack. \n', 100 * eer_cm_A07);
    fprintf('     EER A08           \t= %f %%\t Equal error rate from CM scores A08 attack. \n', 100 * eer_cm_A08);
    fprintf('     EER A09           \t= %f %%\t Equal error rate from CM scores A09 attack. \n', 100 * eer_cm_A09);
    fprintf('     EER A10           \t= %f %%\t Equal error rate from CM scores A10 attack. \n', 100 * eer_cm_A10);
    fprintf('     EER A11           \t= %f %%\t Equal error rate from CM scores A11 attack. \n', 100 * eer_cm_A11);
    fprintf('     EER A12           \t= %f %%\t Equal error rate from CM scores A12 attack. \n', 100 * eer_cm_A12);
    fprintf('     EER A13           \t= %f %%\t Equal error rate from CM scores A13 attack. \n', 100 * eer_cm_A13);
    fprintf('     EER A14           \t= %f %%\t Equal error rate from CM scores A14 attack. \n', 100 * eer_cm_A14);
    fprintf('     EER A15           \t= %f %%\t Equal error rate from CM scores A15 attack. \n', 100 * eer_cm_A15);
    fprintf('     EER A16           \t= %f %%\t Equal error rate from CM scores A16 attack. \n', 100 * eer_cm_A16);
    fprintf('     EER A17           \t= %f %%\t Equal error rate from CM scores A17 attack. \n', 100 * eer_cm_A17);
    fprintf('     EER A18           \t= %f %%\t Equal error rate from CM scores A18 attack. \n', 100 * eer_cm_A18);
    fprintf('     EER A19           \t= %f %%\t Equal error rate from CM scores A19 attack. \n', 100 * eer_cm_A19);
end

% % Plot t-DCF as function of the CM threshold.
% figure(2);
% plot(CM_thresholds, tDCF_curve); hold on;
% h = plot(min_tDCF_threshold, min_tDCF, 'mo'); set(h, 'MarkerSize', 10);
% xlabel('CM threshold (operating point)');
% ylabel('Norm t-DCF');
% title('Normalized t-DCF', 'FontSize', 12, 'FontWeight', 'bold');
% h = line([min(CM_thresholds) max(CM_thresholds)], [1 1]); set(h, 'LineStyle', '--', 'Color', 'k');
% legend('t-DCF', sprintf('min t-DCF (%5.5f)', min_tDCF), 'Arbitrarily bad CM (Norm t-DCF=1)');
% axis([min(CM_thresholds) max(CM_thresholds) 0 1.5]);
% 
% % Traditional DET plot using Bosaris toolkit
% plot_type = Det_Plot.make_plot_window_from_string('old');
% plot_obj = Det_Plot(plot_type, 'Detection error trade-off plot of CM');
% plot_obj.set_system(bona_cm, spoof_cm, sprintf('CM system (EER=%2.2f %%)', 100* eer_cm));
% plot_obj.plot_steppy_det({'b','LineWidth',2},' ');
% plot_obj.display_legend();

end
