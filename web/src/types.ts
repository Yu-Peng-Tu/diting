// 工作台 API 响应类型。数字一律来自后端真实 artifacts JSON，本文件只描述形状。

export interface Scope {
  evaluated_on?: string
  data_scope?: string
  data_scope_short?: string
  level?: string
  split_version?: string
  dataset?: string
  test_set_status?: string
  source_tag?: string
  label_policy?: string
  threshold?: number
  [key: string]: unknown
}

export interface Envelope {
  scope: Scope
  sources: Record<string, string>
  test_rows_loaded: number
  test_set_status: string
  caveats: string[]
  [key: string]: any
}

export interface BinaryMetrics {
  n_samples: number
  n_positive: number
  n_negative: number
  tp: number
  fp: number
  tn: number
  fn: number
  recall: number
  precision: number
  fpr: number
  f1: number
  average_precision: number
  roc_auc?: number
  threshold?: number
  false_alarms_per_10k_normal: number
  confusion_matrix: { labels: string[]; matrix: number[][] }
  note?: string
  [key: string]: any
}

export interface Summary extends Envelope {
  frozen_threshold: number
  same_distribution: {
    binary_at_calibrated_threshold: BinaryMetrics
    binary_at_default_threshold: BinaryMetrics
    multiclass: {
      macro_precision: number
      macro_recall: number
      macro_f1: number
      per_class: Record<string, { precision: number; recall: number; f1: number; support: number }>
      confusion_matrix: { labels: string[]; matrix: number[][] }
      gear_rpm_confusion: Record<string, any>
    }
    per_attack_type: Record<string, any>
    vs_l0_rows: Array<Record<string, any>>
    vs_l0_deltas: Array<Record<string, any>>
  }
  cross_domain: {
    zeroshot_at_frozen_0_54: BinaryMetrics
    zeroshot_at_0_5: BinaryMetrics
    official_label_sensitivity: Record<string, any>
    adaptation_rows: Array<{ policy: string; fold: string; threshold: number; threshold_source: string; model_weights_modified: boolean; metrics: BinaryMetrics; [key: string]: any }>
    same_eval_fold_delta_vs_frozen: Record<string, any>
    fold_sizes: Record<string, number>
    per_attack_type: Record<string, any>
    feature_drift_by_group: Record<string, { n_features: number; mean_abs_delta: number; max_abs_delta: number; sum_abs_delta: number }>
    feature_drift_top: Array<{ feature: string; group: string; hcrl_val_mean: number; otids_mean: number; abs_mean_delta: number; [key: string]: any }>
    dominant_drift_group: string
    five_class_mapping: {
      per_true_type: Record<string, any>
      macro_f1_reported: boolean
      macro_f1_reason: string
    }
  }
  sealing: Record<string, any>
}

export interface PrCurve extends Envelope {
  average_precision: number
  n_points: number
  n_thresholds: number
  source: string
  threshold_note: string
  points: Array<{ index: number; precision: number; recall: number; threshold: number | null }>
}

export interface ThresholdPoint {
  threshold: number
  recall: number
  fpr: number
  youden_j: number
  tp: number
  fp: number
  tn: number
  fn: number
}

export interface ThresholdCurve extends Envelope {
  version: string
  criterion: string
  criterion_note: string
  calibrated_on: string
  default_threshold: number
  selected_threshold: number
  selected_youden_j: number
  n_grid_points: number
  grid_range: number[]
  curve: ThresholdPoint[]
  recompute_note: string
}

export interface Ablation extends Envelope {
  same_distribution: Record<string, any>
  cross_domain: Record<string, any>
  direction_note: string
}

export interface ErrorsData extends Envelope {
  threshold_used: number
  n_val_windows: number
  n_false_positives: number
  n_false_negatives: number
  label_boundary_analysis: {
    window_label_theta: number
    near_boundary_band: number[]
    error_attack_ratios: number[]
    n_errors: number
    n_errors_near_boundary: number
    fp_attack_ratios: number[]
    fn_attack_ratios: number[]
    finding: string
  }
  governance_recommendations: string[]
  records: Array<Record<string, any>>
  note: string
  [key: string]: any
}

export interface SampleItem {
  sample_id: string
  window_id: string
  segment_id: string
  source: string
  split: string
  dominant_class: string
  label: number
  attack_ratio: number
}

export interface Samples extends Envelope {
  samples: SampleItem[]
  n_samples: number
  sealing: Record<string, any>
  usage: string
}

export interface Health extends Envelope {
  status: string
  model_version: string
  frozen_threshold: number
  n_demo_samples: number
  sealing: Record<string, any>
}

export interface DetectResult extends Envelope {
  attack_probability: number
  normal_probability: number
  threshold: number
  threshold_is_frozen: boolean
  decision: 'attack' | 'normal'
  multiclass: { predicted_class: string; probabilities: Record<string, number>; note: string }
  top_features: Array<{ feature: string; group: string; value: number; contribution_to_attack_proba: number }>
  contribution_method: string
  model_version: string
  split_version: string
  n_features: number
  input: Record<string, any>
}

export interface Dataset {
  summary: Summary
  pr: PrCurve
  thr: ThresholdCurve
  ablation: Ablation
  errors: ErrorsData
  samples: Samples
  health: Health
}
