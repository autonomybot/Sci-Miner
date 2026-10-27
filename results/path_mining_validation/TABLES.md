# Path mining and validation: result tables

All numbers are produced by `run_all.py`; nothing here is typed by hand.

## Table 1. Science-graph size and cleaning

| metric | e3sm.csv | neo4j export |
|---|---|---|
| raw_rows | 3.065e+04 | 6.287e+04 |
| rows_with_missing_fields | 1.402e+04 | 367 |
| unique_valid_rows | 3.058e+04 | 6.286e+04 |
| files | 1,319 | 1,390 |
| equations_with_source_file | 6,011 | 2.13e+04 |
| scientific_symbols_kept | 1,470 | 6,173 |
| symbols_dropped_(markup/numbers/descriptions) | 3 | 808 |
| equations_in_search_graph | 4,060 | 1,326 |
| edges_in_search_graph | 8,837 | 1.381e+04 |
| connected_components | 322 | 65 |
| largest_component_nodes | 4,567 | 6,949 |
| largest_component_share | 0.8259 | 0.9267 |
| median_symbol_degree | 2 | 1 |
| max_symbol_degree | 502 | 136 |

## Table 2. Agreement of every e3sm.csv edge with the Fortran sources

`chance` re-pairs each edge with a random Fortran file.

| relationship | check | edges | passed | rate (%) | chance (%) |
|---|---|---|---|---|---|
| has | file exists in Fortran folder | 1400 | 1400 | 100.0 |  |
| contains | symbol is an identifier of its file (all) | 14288 | 14288 | 100.0 |  |
| contains | symbol is an identifier of its file (identifier-like symbols) | 14254 | 14254 | 100.0 | 4.5 |
| contains | same, ignoring underscores/sub-script marks (lenient) | 14254 | 14254 | 100.0 |  |
| encodes | source file exists | 6050 | 6050 | 100.0 |  |
| encodes | >= 50% of equation identifiers found in its file | 6050 | 6050 | 100.0 | 4.4 |
| hasVariable | symbol occurs in the equation text (all) | 8536 | 7237 | 84.8 |  |
| hasVariable | symbol is an identifier of the equation's file (identifier-like, file found) | 8533 | 8533 | 100.0 | 10.5 |
| hasVariable | same, lenient identifier match | 8533 | 8533 | 100.0 |  |
| hasConstant | symbol occurs in the equation text (all) | 309 | 279 | 90.3 |  |
| hasConstant | symbol is an identifier of the equation's file (identifier-like, file found) | 309 | 309 | 100.0 | 10.7 |
| hasConstant | same, lenient identifier match | 309 | 309 | 100.0 |  |

## Table 3. Line-level provenance of every equation

| status | equations | share (%) | chance (%) |
|---|---|---|---|
| located | 3705 | 61.2 | 0.7 |
| partial | 2345 | 38.8 | 2.6 |
| unlocated | 0 | 0.0 | 96.7 |
| no_identifier | 0 | 0.0 | 0.0 |
| file_missing | 0 | 0.0 | 0.0 |

Among located equations: 51.3% have a matching assignment target, 42.0% share at least one numeric literal, median statement window = 1.

## Table 4. Variable provenance

| status | symbols | share (%) |
|---|---|---|
| declared | 12200 | 85.6 |
| used_only | 2042 | 14.3 |
| absent | 12 | 0.1 |

Variables of located equations that occur inside the matched statement:

| kind | size | in matched line (%) |
|---|---|---|
| const | 309 | 100.0 |
| var | 8585 | 99.9 |

## Table 5. Symbolic consistency (random equation sample)

| sampled_equations | parse_to_lhs=rhs | rate |
|---|---|---|
| 1500.000 | 398.000 | 0.265 |

## Table 6. Fifteen equation-richest files

| file | equations | located | partial | traced_share |
|---|---|---|---|---|
| m_SparseMatrixToMaps.F90 | 105 | 71 | 34 | 1.00 |
| d2a3dikj.F90 | 66 | 49 | 17 | 1.00 |
| ionf_mod.F90 | 64 | 52 | 12 | 1.00 |
| EMI_CanopyStateType_DataMod.F90 | 58 | 45 | 13 | 1.00 |
| dice_flux_atmice_mod.F90 | 58 | 49 | 9 | 1.00 |
| mo_msis_ubc.F90 | 53 | 47 | 6 | 1.00 |
| redistributetest.F90 | 51 | 34 | 17 | 1.00 |
| EcosysMicDynParamMod.F90 | 49 | 42 | 7 | 1.00 |
| dead_mct_mod.F90 | 49 | 23 | 26 | 1.00 |
| nf_vario.F90 | 48 | 32 | 16 | 1.00 |
| H5VLff.F90 | 47 | 24 | 23 | 1.00 |
| dead_mod.F90 | 46 | 30 | 16 | 1.00 |
| rof_cpl_indices.F90 | 42 | 42 | 0 | 1.00 |
| mo_rte_util_array.F90 | 40 | 26 | 14 | 1.00 |
| ice_fall.F90 | 39 | 31 | 8 | 1.00 |

## Table 7. Path mining on random variable pairs (e3sm.csv)

| metric | value |
|---|---|
| queries | 150.000 |
| connected | 138.000 |
| connected_share | 0.920 |
| median_equations_best | 4.000 |
| median_hub_degree_best | 56.000 |
| median_hub_degree_naive | 59.000 |
| mean_V_best | 0.763 |
| mean_V_naive | 0.729 |
| mean_score_best | 0.914 |
| mean_score_naive | 0.892 |
| verified_share_best | 0.283 |
| partial_share_best | 0.717 |
| rejected_share_best | 0.000 |
| verified_share_naive | 0.159 |
| ranking_changed_top_path | 0.355 |
| candidate_verified_share | 0.154 |

## Table 8. Paper case studies

`endpoint missing` = the variable has no equation link in that graph; `no path` = both endpoints exist but are not connected.

| graph | case | source | target | status | n_equations | S_src | S_var | S_sym | V | score | status_class |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e3sm.csv | Lake chain (Fortran) | rhow | savedtke1 | path found | 4 | 1.000 | 1.000 | 1.000 | 0.850 | 0.667 | partially_verified |
| e3sm.csv | Lake chain: density to Richardson | rhow | ri | path found | 2 | 1.000 | 1.000 | 1.000 | 0.850 | 0.804 | partially_verified |
| e3sm.csv | Lake chain: temperature to buoyancy | t_lake | n2 | path found | 2 | 1.000 | 1.000 | 1.000 | 0.850 | 0.807 | partially_verified |
| e3sm.csv | Fig.5A lake pathway | rho_w | savedtke1 | endpoint missing |  |  |  |  |  |  |  |
| e3sm.csv | Fig.4 aerosol pathway | NH3 | dT/dy | endpoint missing |  |  |  |  |  |  |  |
| e3sm.csv | Fig.5B methane pathway | T_soisno | conc_ch4_sat | endpoint missing |  |  |  |  |  |  |  |
| e3sm.csv | Density to Richardson number | rho_w | Ri | endpoint missing |  |  |  |  |  |  |  |
| e3sm.csv | Mixing energy to buoyancy | tke | N2 | path found | 3 | 1.000 | 1.000 | 0.000 | 0.667 | 0.916 | partially_verified |
| e3sm.csv | Soil temperature to methane | T_Soisno | CH4 | endpoint missing |  |  |  |  |  |  |  |
| e3sm.csv | von Karman to Richardson number | kappa | Ri | path found | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.288 | verified |
| neo4j export | Lake chain (Fortran) | rhow | savedtke1 | path found | 5 | 0.400 | 0.667 | 0.000 | 0.367 | 0.465 | rejected |
| neo4j export | Lake chain: density to Richardson | rhow | ri | path found | 3 | 0.333 | 0.500 | 0.000 | 0.308 | 0.504 | rejected |
| neo4j export | Lake chain: temperature to buoyancy | t_lake | n2 | endpoint missing |  |  |  |  |  |  |  |
| neo4j export | Fig.5A lake pathway | rho_w | savedtke1 | no path |  |  |  |  |  |  |  |
| neo4j export | Fig.4 aerosol pathway | NH3 | dT/dy | path found | 4 | 0.750 | 0.400 | 0.500 | 0.550 | 0.772 | partially_verified |
| neo4j export | Fig.5B methane pathway | T_soisno | conc_ch4_sat | endpoint missing |  |  |  |  |  |  |  |
| neo4j export | Density to Richardson number | rho_w | Ri | no path |  |  |  |  |  |  |  |
| neo4j export | Mixing energy to buoyancy | tke | N2 | path found | 3 | 0.667 | 0.000 | 0.333 | 0.333 | 0.523 | rejected |
| neo4j export | Soil temperature to methane | T_Soisno | CH4 | endpoint missing |  |  |  |  |  |  |  |
| neo4j export | von Karman to Richardson number | kappa | Ri | path found | 4 | 0.250 | 0.800 | 0.000 | 0.350 | 0.637 | rejected |

## Table 9. Numerical replay of the lake chain (Fortran formulas vs E3SM output)

| step | n | MAE | RMSE | R2 | replayable | note |
|---|---|---|---|---|---|---|
| rho_w = 1000(1-1.9549e-5\|T-T_dmax\|^1.68)(1-f_ice)+917 f_ice | 26283 | 0.00055 | 0.00137 | 0.999996 | True | layers 1-3 |
| N2 = (g/rho_w) d(rho_w)/dz | 17522 | 1.02e-07 | 1.34e-07 | 1.000000 | True | layers 1-2 |
| Ri = [-1+sqrt(1+40 N2 (k z)^2 / (ws^2 exp(-2 ks z)))]/20 | 17522 | 1.15e+03 | 2.09e+03 | -0.431588 | False | layers 1-2; LAKE_WS range 274.4-299.1, LAKE_KS up to 0.0082 are not a velocity/attenuation scale |
| kme = km + ke + fang | 17522 | 76.5 | 220 | -201237793717.185089 | False | ice-free layers 1-2 |
| savedtke1 = kme(1) cwat | 8761 | 686 | 2.25e+03 | -86430.314440 | False | TKE1 field spans 275.4-298.3; kme(1)*cwat spans 0.9-34750 |

## Table 10. Current compact surrogate relations on the held-out 30 % of the series

Lake (current one-variable equation):

| relation | model | n | MAE | RMSE | NRMSE | R2 | r |
|---|---|---|---|---|---|---|---|
| rho_w(level 1) vs savedtke1 | current compact equation (test split) | 2629 | 0.428 | 0.591 | 0.306 | -0.174 | -0.022 |
| rho_w(level 1) vs savedtke1 | current compact equation (full series) | 8761 | 0.724 | 1.060 | 0.380 | -0.345 | 0.066 |
| rho_w(level 1) vs savedtke1 | MLP baseline (64-64) | 2629 | 0.426 | 0.638 | 0.330 | -0.368 | 0.427 |
| rho_w(level 1) vs savedtke1 | constant (train mean) | 2629 | 0.644 | 0.702 | 0.363 | -0.656 |  |

Methane (current one-variable equation):

| relation | model | n | MAE | RMSE | NRMSE | R2 | r |
|---|---|---|---|---|---|---|---|
| T_soisno(level 1) vs conc_ch4_sat | current compact equation (test split) | 2633 | 0.925 | 1.112 | 0.071 | 0.946 | 0.985 |
| T_soisno(level 1) vs conc_ch4_sat | current compact equation (full series) | 8776 | 2.522 | 3.297 | 0.143 | 0.813 | 0.902 |
| T_soisno(level 1) vs conc_ch4_sat | same compact form, re-fitted on train | 2633 | 20.937 | 22.459 | 1.435 | -21.027 | -0.993 |
| T_soisno(level 1) vs conc_ch4_sat | MLP baseline (64-64) | 2633 | 7.076 | 8.023 | 0.513 | -1.810 | 0.982 |
| T_soisno(level 1) vs conc_ch4_sat | constant (train mean) | 2633 | 4.213 | 4.786 | 0.306 | -0.000 | 0.000 |

## Table 11. Plausibility of the lake-scalar columns in the supplied CSV

| column | min | median | max | corr_with_TLAKE_level1 |
|---|---|---|---|---|
| TKE1[lndgrid=1] | 275.4 | 282.4 | 298.3 | 0.987 |
| LAKE_WS[lndgrid=1] | 274.4 | 283 | 299.1 |  |
| LAKE_KS[lndgrid=1] | 0 | 8.979e-05 | 0.008163 |  |
| TLAKE[levlak=1,lndgrid=1] | 275.9 | 281.8 | 296.2 | 1.000 |
