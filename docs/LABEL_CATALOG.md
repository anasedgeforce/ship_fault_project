# Category names and UI label contract

Use `GET /v1/labels` (with the configured API key, when required) for the active model catalog. The same catalog is included in `/v1/models` as `label_catalog`. A reference snapshot is available in `docs/label_catalog.json`.

Keep machine IDs in requests, predictions, MongoDB records, label files, grouping and accuracy comparisons. Use display names only for presentation. Labels were not changed in the training data or model artifacts.

## Standard terminology

| Field | Display label | Meaning |
| --- | --- | --- |
| Family | Fault family | Main predicted condition category; includes Normal. |
| Detailed condition / child | Fault subtype | Specific condition within the family. |
| Family score | Family model score | Model score for the predicted family. |
| Subtype score | Subtype model score | Model score for the predicted subtype. |
| Decision | Review status | Whether review or input correction is required. |
| Recording duration | Recording duration | Duration derived from sensor timestamps. |

Model scores are not measured accuracy or calibrated fault probabilities. Accuracy requires verified labels. Manual CSV results show one aggregate family and subtype score for the full recording.

## Families and subtypes

| Machine ID | Fault family | Fault subtype |
| --- | --- | --- |
| `normal__1200_rpm` | Normal | 1,200 RPM |
| `normal__1300_rpm` | Normal | 1,300 RPM |
| `normal__1400_rpm` | Normal | 1,400 RPM |
| `normal__1500_rpm` | Normal | 1,500 RPM |
| `normal__1600_rpm` | Normal | 1,600 RPM |
| `normal__1700_rpm` | Normal | 1,700 RPM |
| `normal__1500_rpm__v1` | Normal | 1,500 RPM (Variant 1) |
| `normal__loading_unspecified` | Normal | Loading Unspecified |
| `normal__with_loader` | Normal | With Loader |
| `inner_race__high_severity__ub_double_mass` | Inner Race Fault | High Severity — Double-Mass Unbalance |
| `inner_race__high_severity__ub_single_mass` | Inner Race Fault | High Severity — Single-Mass Unbalance |
| `inner_race__high_severity__no_ub_specified` | Inner Race Fault | High Severity — Unbalance Not Specified |
| `inner_race__low_severity__ub_double_mass` | Inner Race Fault | Low Severity — Double-Mass Unbalance |
| `inner_race__low_severity__ub_single_mass` | Inner Race Fault | Low Severity — Single-Mass Unbalance |
| `inner_race__low_severity__no_ub_specified` | Inner Race Fault | Low Severity — Unbalance Not Specified |
| `outer_race__high_severity__ub_double_mass` | Outer Race Fault | High Severity — Double-Mass Unbalance |
| `outer_race__high_severity__ub_single_mass` | Outer Race Fault | High Severity — Single-Mass Unbalance |
| `outer_race__high_severity__no_ub_specified` | Outer Race Fault | High Severity — Unbalance Not Specified |
| `outer_race__low_severity__ub_double_mass` | Outer Race Fault | Low Severity — Double-Mass Unbalance |
| `outer_race__low_severity__ub_single_mass` | Outer Race Fault | Low Severity — Single-Mass Unbalance |
| `outer_race__low_severity__no_ub_specified` | Outer Race Fault | Low Severity — Unbalance Not Specified |
| `cocked_bearing__with_loader` | Cocked Bearing | With Loader |
| `cocked_bearing__without_loader` | Cocked Bearing | Without Loader |
| `cocked_bearing__without_loader__ub_double_mass` | Cocked Bearing | Without Loader — Double-Mass Unbalance |
| `foundation_looseness__level_1` | Foundation Looseness | Level 1 |
| `foundation_looseness__level_2` | Foundation Looseness | Level 2 |
| `foundation_looseness__level_1__ub_single_mass` | Foundation Looseness | Level 1 — Single-Mass Unbalance |
| `foundation_looseness__level_2__ub_single_mass` | Foundation Looseness | Level 2 — Single-Mass Unbalance |
| `foundation_looseness__loose_bottom_bolts` | Foundation Looseness | Loose Bottom Bolts |
| `dynamic_unbalance__phase_180__without_loader` | Dynamic Unbalance | 180° — Without Loader |
| `dynamic_unbalance__phase_90__without_loader` | Dynamic Unbalance | 90° — Without Loader |
| `dynamic_unbalance__double_mass__without_loader` | Dynamic Unbalance | Double Mass — Without Loader |
| `dynamic_unbalance__single_mass__without_loader` | Dynamic Unbalance | Single Mass — Without Loader |

There are six families and 33 subtypes. The Normal 1,500 RPM variant remains distinct. Unbalance Not Specified does not mean unbalance is absent. Loose Bottom Bolts remains Foundation Looseness even where the original source folder is named Normal. Loader status is displayed only when encoded in that subtype.

## Review status

| API code | Display name |
| --- | --- |
| `classified` | Prediction Available |
| `needs_review` | Review Required |
| `invalid_input` | Invalid Input |
| `failed` | Analysis Failed |

Prediction Available is not a confirmed diagnosis. Review Required does not mean prediction failed: candidate labels and scores can still be returned. Input verification and prediction accuracy remain separate.

## Review messages

| API code | Display message |
| --- | --- |
| `continuous_duration_unvalidated` | Continuous operation has not been validated for this recording duration. |
| `physical_axis_mapping_unverified` | Physical sensor-axis mapping has not been verified. |
| `equipment_domain_unvalidated` | Model suitability for this equipment has not been validated. |
| `family_detail_disagree` | The family and independent subtype predictions disagree. |
| `low_model_score` | One or more model scores are below the review threshold. |
| `novelty_reference_unavailable` | The reference for unfamiliar signals is unavailable. |
| `outside_training_feature_reference` | The signal differs from the training reference. |
| `flat_channel` | A channel is flat or nearly flat. |
| `possible_clipping` | A channel may be clipped at its measurement limit. |
| `identical_channels` | Two channels contain identical readings; check acquisition and wiring. |
| `mixed_fault_predictions_within_recording` | The recording contains inconsistent fault predictions. |

## Integration

1. Fetch `/v1/labels` when loading model metadata.
2. Resolve a prediction ID with `catalog.labels[id]`.
3. Resolve review status with `catalog.decisions[code]` and reasons with `catalog.review_reasons[code]`.
4. Use each subtype's `family_id` for relationships, never split its display name.
5. Submit stable IDs, not display strings, in known-label files.
6. Preserve review flags and missing results; do not translate them into confirmed faults.

The existing page uses this catalog for result cards, tables, groups, live predictions and saved-recording selectors. Future UI integration can use the same contract without changing model code.
