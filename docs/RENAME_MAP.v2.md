# RENAME_MAP.v2.md — GitHub Final Submission Script Rename Register
> Generated: 2026-09-27 (Spec: reorg_crm_submit_20260927)
> Baseline: 81 scripts under `scripts/*.{py,R,sh}` (pre-release baseline tree)

| # | old_path (relative to `scripts/`) | new_path (relative to `scripts/`) | Action | Rationale |
|---|---|---|---|---|
| 1 | 100_cohort_composition.py | 100_cohort_composition.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 2 | 101_geometry_cross_seed.py | 101_geometry_cross_seed.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 3 | 102_known_edge_per_seed.py | 102_known_edge_per_seed.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 4 | 103_reproduction_status.py | 103_reproduction_status.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 5 | 10_download_string_ppi.py | 10_download_string_ppi.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 6 | 11_download_tcga_xena.R | 11_download_tcga_xena.R | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 7 | 20_build_graph_esm.py | 20_build_graph_esm.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 8 | 21_build_graph_xena_large.py | 21_build_graph_xena_large.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 9 | 22_build_test_graph_ci.py | 22_build_test_graph_ci.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 10 | 30_embed_esm2.py | 30_embed_esm2.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 11 | 32_precompute_c3l_targets.py | 32_precompute_c3l_targets.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 12 | 33_build_patient_gene_edges.py | 33_build_patient_gene_edges.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 13 | 34_build_patient_esm_graph.py | 34_build_patient_esm_graph.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 14 | 35_build_patient_paad_graph.py | 35_build_patient_paad_graph.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 15 | 36_build_graph_variants.py | 36_build_graph_variants.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 16 | 39_build_random_edge_controls.py | 39_build_random_edge_controls.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 17 | 40_pretrain.py | 40_pretrain.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 18 | 41_pretrain_single_cancer.py | 41_pretrain_single_cancer.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 19 | 50_eval_checkpoint.py | 50_eval_checkpoint.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 20 | 51_eval_gene_embeddings.py | 51_eval_gene_embeddings.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 21 | 52_eval_single_cancer.py | 52_eval_single_cancer.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 22 | 53_eval_esm_baseline.py | 53_eval_esm_baseline.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 23 | 54_eval_esm650.py | 54_eval_esm650.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 24 | 55_eval_supervised_ppi.py | 55_eval_supervised_ppi.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 25 | 56_center_check_diag.py | 56_center_check_diag.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 26 | 60_metabric_validation.py | 60_metabric_validation.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 27 | 61_patient_eval.py | 61_patient_eval.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 28 | 62_patient_pretrain_eval.py | 62_patient_pretrain_eval.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 29 | 63_paad_transfer_eval.py | 63_paad_transfer_eval.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 30 | 64_drug_response_gdsc.py | 64_drug_response_gdsc.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 31 | 65_drug_response_stats.py | 65_drug_response_stats.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 32 | 66_tissue_specific_ppi.py | 66_tissue_specific_ppi.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 33 | 70_go_enrichment.py | 70_go_enrichment.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 34 | 71_pathway_membership.py | 71_pathway_membership.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 35 | 72_c3l_probe.py | 72_c3l_probe.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 36 | 73_feature_scan_probe.py | 73_feature_scan_probe.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 37 | 74_c3l_permutation_test.py | 74_c3l_permutation_test.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 38 | 75_cluster_diagnostics.py | 75_cluster_diagnostics.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 39 | 75_patient_setfingerprint.py | 120_patient_setfingerprint.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 40 | 76_kegg_membership_table.py | 76_kegg_membership_table.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 41 | 77_build_holdout_graph.py | 77_build_holdout_graph.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 42 | 78_eval_holdout.py | 78_eval_holdout.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 43 | 79_build_control_graphs.py | 79_build_control_graphs.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 44 | 80_collect_losses.py | 80_collect_losses.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 45 | 81_assemble_results.py | 81_assemble_results.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 46 | 82_gen_final_tables.py | 82_gen_final_tables.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 47 | 84_extract_fig1_layout.py | 84_extract_fig1_layout.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 48 | 85_make_figures.py | 85_make_figures.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 49 | 86_check_figure_specs.py | 86_check_figure_specs.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 50 | 87_build_fig1_composite.py | 87_build_fig1_composite.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 51 | 88_audit_manuscript.py | 88_audit_manuscript.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 52 | 89_audit_numbers.py | 89_audit_numbers.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 53 | 89_collect_holdout_results.py | 121_collect_holdout_results.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 54 | 90_audit_references.py | 90_audit_references.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 55 | 90_bench_device.py | 122_bench_device.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 56 | 90_report_control_graphs.py | 123_report_control_graphs.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 57 | 91_eval_probe_regimes.py | 124_eval_probe_regimes.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 58 | 91_namr_denoiser_diagnostic.py | 125_namr_denoiser_diagnostic.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 59 | 92_c3l_prototype_diagnostic.py | 126_c3l_prototype_diagnostic.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 60 | 92_eval_ckpt_std.py | 127_eval_ckpt_std.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 61 | 93_csp_geometry_shortcut.py | 93_csp_geometry_shortcut.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 62 | 94_c3l_diagnostic.py | 94_c3l_diagnostic.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 63 | 95_tissue_string_sensitivity.py | 95_tissue_string_sensitivity.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 64 | 96_drug_baselines.py | 128_drug_baselines.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 65 | 96_graph_smoothing_baseline.py | 129_graph_smoothing_baseline.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 66 | 97_kegg_per_seed.py | 130_kegg_per_seed.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 67 | 97_patient_baselines_2cancer.py | 131_patient_baselines_2cancer.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 68 | 98_audit_checkpoints.py | 132_audit_checkpoints.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 69 | 98_fix_patient_graph.py | 133_fix_patient_graph.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 70 | 99_c3l_elimination_ablation.py | 134_c3l_elimination_ablation.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 71 | 99_kegg_paired_diffs.py | 135_kegg_paired_diffs.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 72 | 99_parse_c3l_results.py | 136_parse_c3l_results.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 73 | 99_patient_baselines_full.py | 137_patient_baselines_full.py | RENAME | Duplicate numeric prefix; promoted to 3-digit phase-safe range (120–137). |
| 74 | config.py | config.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 75 | figure_data.py | figure_data.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 76 | figure_style.py | figure_style.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 77 | gfm_losses.py | gfm_losses.py | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 78 | run_ablation_evals.sh | run_ablation_evals.sh | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 79 | run_holdout_suite.sh | run_holdout_suite.sh | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 80 | run_namr_sweep_local.sh | run_namr_sweep_local.sh | KEEP | No naming conflict; retains original 2-digit numeric slot. |
| 81 | run_namr_sweep_parallel.sh | run_namr_sweep_parallel.sh | KEEP | No naming conflict; retains original 2-digit numeric slot. |

**Baseline rows total**: 81 (1:1 coverage, every `scripts/` baseline file appears exactly once; KEEP/RENAME/DELETE are mutually exclusive).

Legend: `KEEP` = no change required. `RENAME` = duplicate prefix resolution. `DELETE` = excluded from submission scope.
