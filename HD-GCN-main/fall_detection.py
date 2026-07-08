import json
import os

import numpy as np


# =========================================================
# 0. Runtime labels
# =========================================================
GROUP_LABELS = ["normal", "fall-like", "fall"]
GROUP_NAME_TO_ID = {name: index for index, name in enumerate(GROUP_LABELS)}
CLASS_NAMES = ["Normal", "Fall-like", "Fall"]


# =========================================================
# 1. NTU120 label names
# =========================================================
DEFAULT_LABEL_NAMES = [
    "A001 drink water", "A002 eat meal", "A003 brush teeth", "A004 brush hair",
    "A005 drop", "A006 pick up", "A007 throw", "A008 sit down",
    "A009 stand up", "A010 clapping", "A011 reading", "A012 writing",
    "A013 tear up paper", "A014 put on jacket", "A015 take off jacket", "A016 put on a shoe",
    "A017 take off a shoe", "A018 put on glasses", "A019 take off glasses", "A020 put on a hat/cap",
    "A021 take off a hat/cap", "A022 cheer up", "A023 hand waving", "A024 kicking something",
    "A025 reach into pocket", "A026 hopping", "A027 jump up", "A028 phone call",
    "A029 play with phone/tablet", "A030 type on a keyboard", "A031 point to something", "A032 taking a selfie",
    "A033 check time (from watch)", "A034 rub two hands", "A035 nod head/bow", "A036 shake head",
    "A037 wipe face", "A038 salute", "A039 put palms together", "A040 cross hands in front",
    "A041 sneeze/cough", "A042 staggering", "A043 falling down", "A044 headache",
    "A045 chest pain", "A046 back pain", "A047 neck pain", "A048 nausea/vomiting",
    "A049 fan self", "A050 punch/slap", "A051 kicking", "A052 pushing",
    "A053 pat on back", "A054 point finger", "A055 hugging", "A056 giving object",
    "A057 touch pocket", "A058 shaking hands", "A059 walking towards", "A060 walking apart",
    "A061 put on headphone", "A062 take off headphone", "A063 shoot at basket", "A064 bounce ball",
    "A065 tennis bat swing", "A066 juggle table tennis ball", "A067 hush", "A068 flick hair",
    "A069 thumb up", "A070 thumb down", "A071 make OK sign", "A072 make victory sign",
    "A073 staple book", "A074 counting money", "A075 cutting nails", "A076 cutting paper",
    "A077 snap fingers", "A078 open bottle", "A079 sniff/smell", "A080 squat down",
    "A081 toss a coin", "A082 fold paper", "A083 ball up paper", "A084 play magic cube",
    "A085 apply cream on face", "A086 apply cream on hand", "A087 put on bag", "A088 take off bag",
    "A089 put object into bag", "A090 take object out of bag", "A091 open a box", "A092 move heavy objects",
    "A093 shake fist", "A094 throw up cap/hat", "A095 capitulate", "A096 cross arms",
    "A097 arm circles", "A098 arm swings", "A099 run on the spot", "A100 butt kicks",
    "A101 cross toe touch", "A102 side kick", "A103 yawn", "A104 stretch oneself",
    "A105 blow nose", "A106 hit with object", "A107 wield knife", "A108 knock over",
    "A109 grab stuff", "A110 shoot with gun", "A111 step on foot", "A112 high-five",
    "A113 cheers and drink", "A114 carry object", "A115 take a photo", "A116 follow",
    "A117 whisper", "A118 exchange things", "A119 support somebody", "A120 rock-paper-scissors",
]

# =========================================================
# 2. Custom 60-class fall-detection config
# =========================================================
DEFAULT_RETAINED_SOURCE_IDS = list(range(49)) + [60, 61, 79, 86, 87, 88, 89, 90, 91, 100, 113]
DEFAULT_POSITIVE_SOURCE_ID = 42
DEFAULT_FALL_LIKE_SOURCE_IDS = [4, 5, 7, 8, 15, 16, 41, 79, 88, 89, 90, 91, 100]
DEFAULT_MONITORED_SOURCE_IDS = [42, 41, 79, 7, 8, 15, 16, 88, 89, 90, 91, 100]
DEFAULT_SQUAT_SOURCE_ID = 79

# =========================================================
# 3. Training rules
# =========================================================
DEFAULT_CLASS_WEIGHT_RULE = {
    "enabled": True,
    "positive": 6.0,
    "fall_like": 2.5,
    "other": 1.0,
}
DEFAULT_OVERSAMPLE_RULE = {
    "enabled": True,
    "positive": 4.0,
    "fall_like": 2.0,
    "other": 1.0,
    "source_multipliers": {},
}
DEFAULT_HARD_NEGATIVE_RULE = {
    "enabled": True,
    "extra_multiplier": 2.0,
    "fall_bonus_multiplier": 2.0,
    "fine_tune_epochs": 10,
    "lr_scale": 0.1,
}


# =========================================================
# 4. Config helpers
# =========================================================
def _merge_dict(defaults, override):
    merged = dict(defaults)
    if override:
        merged.update(override)
    return merged


def _unique_int_list(values):
    result = []
    for value in values or []:
        int_value = int(value)
        if int_value not in result:
            result.append(int_value)
    return result


def build_config(source=None, num_classes=None):
    source = source or {}
    retained_source_ids = _unique_int_list(source.get("retained_source_ids") or DEFAULT_RETAINED_SOURCE_IDS)
    if num_classes is not None and int(num_classes) != len(retained_source_ids):
        raise ValueError(
            "num_class ({}) must match retained_source_ids length ({})".format(
                num_classes, len(retained_source_ids)
            )
        )

    positive_source_id = int(
        source.get("positive_source_id", source.get("positive_class_id", DEFAULT_POSITIVE_SOURCE_ID))
    )
    fall_like_source_ids = _unique_int_list(
        source.get("fall_like_source_ids", source.get("fall_like_seed_ids") or DEFAULT_FALL_LIKE_SOURCE_IDS)
    )
    monitored_source_ids = _unique_int_list(
        source.get("monitored_source_ids") or DEFAULT_MONITORED_SOURCE_IDS
    )

    source_to_compact = {
        int(source_id): index for index, source_id in enumerate(retained_source_ids)
    }
    compact_to_source = list(retained_source_ids)

    if positive_source_id not in source_to_compact:
        raise ValueError("positive_source_id {} is not in retained_source_ids".format(positive_source_id))

    missing_fall_like = [source_id for source_id in fall_like_source_ids if source_id not in source_to_compact]
    if missing_fall_like:
        raise ValueError("fall_like_source_ids not retained: {}".format(missing_fall_like))

    group_by_source_id = {}
    for source_id in retained_source_ids:
        if source_id == positive_source_id:
            group_by_source_id[source_id] = "fall"
        elif source_id in fall_like_source_ids:
            group_by_source_id[source_id] = "fall-like"
        else:
            group_by_source_id[source_id] = "normal"

    group_id_by_compact_id = np.asarray(
        [GROUP_NAME_TO_ID[group_by_source_id[source_id]] for source_id in retained_source_ids],
        dtype=np.int64,
    )
    group_name_by_compact_id = [group_by_source_id[source_id] for source_id in retained_source_ids]

    return {
        "enabled": True,
        "class_names": list(CLASS_NAMES),
        "num_classes": len(retained_source_ids),
        "retained_source_ids": list(retained_source_ids),
        "source_to_compact": source_to_compact,
        "compact_to_source": compact_to_source,
        "positive_source_id": positive_source_id,
        "positive_class_id": source_to_compact[positive_source_id],
        "fall_like_source_ids": list(fall_like_source_ids),
        "fall_like_seed_ids": [source_to_compact[source_id] for source_id in fall_like_source_ids],
        "group_by_source_id": group_by_source_id,
        "group_id_by_compact_id": group_id_by_compact_id,
        "group_name_by_compact_id": group_name_by_compact_id,
        "monitored_source_ids": [source_id for source_id in monitored_source_ids if source_id in retained_source_ids],
        "class_weight_rule": _merge_dict(DEFAULT_CLASS_WEIGHT_RULE, source.get("class_weight_rule")),
        "oversample_rule": _merge_dict(DEFAULT_OVERSAMPLE_RULE, source.get("oversample_rule")),
        "hard_negative_rule": _merge_dict(DEFAULT_HARD_NEGATIVE_RULE, source.get("hard_negative_rule")),
        "squat_source_id": int(source.get("squat_source_id", DEFAULT_SQUAT_SOURCE_ID)),
    }


# =========================================================
# 5. Label-name helpers
# =========================================================
def load_label_names(path, num_classes=None):
    labels = []
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file_obj:
            labels = [line.strip() for line in file_obj if line.strip()]
    if not labels:
        labels = list(DEFAULT_LABEL_NAMES)
    if num_classes is not None and len(labels) < num_classes:
        labels.extend([f"class_{index}" for index in range(len(labels), num_classes)])
    return labels


def build_compact_label_names(all_label_names, config):
    return [
        all_label_names[source_id] if source_id < len(all_label_names) else "class_{}".format(source_id)
        for source_id in config["retained_source_ids"]
    ]


# =========================================================
# 6. Training weight and sampling helpers
# =========================================================
def build_class_weights(num_classes, config):
    weights = np.full(num_classes, config["class_weight_rule"]["other"], dtype=np.float32)
    if not config["enabled"] or not config["class_weight_rule"]["enabled"]:
        return weights

    weights[config["positive_class_id"]] = config["class_weight_rule"]["positive"]
    weights[np.asarray(config["fall_like_seed_ids"], dtype=np.int64)] = config["class_weight_rule"]["fall_like"]
    return weights


def build_sample_weights(labels, config, hard_negative_info=None):
    labels = np.asarray(labels, dtype=np.int64)
    weights = np.full(len(labels), config["oversample_rule"]["other"], dtype=np.float32)
    if not config["enabled"] or not config["oversample_rule"]["enabled"]:
        return weights

    weights[labels == config["positive_class_id"]] = config["oversample_rule"]["positive"]
    weights[np.isin(labels, np.asarray(config["fall_like_seed_ids"], dtype=np.int64))] = (
        config["oversample_rule"]["fall_like"]
    )

    source_multipliers = config["oversample_rule"].get("source_multipliers") or {}
    for source_id, multiplier in source_multipliers.items():
        source_id = int(source_id)
        if source_id not in config["source_to_compact"]:
            continue
        compact_id = int(config["source_to_compact"][source_id])
        weights[labels == compact_id] *= float(multiplier)

    if not hard_negative_info:
        return weights

    all_indices = np.asarray(hard_negative_info.get("all_indices", []), dtype=np.int64)
    fall_indices = np.asarray(hard_negative_info.get("fall_indices", []), dtype=np.int64)
    if len(all_indices):
        weights[all_indices] *= float(config["hard_negative_rule"]["extra_multiplier"])
    if len(fall_indices):
        weights[fall_indices] *= float(config["hard_negative_rule"]["fall_bonus_multiplier"])

    return weights


# =========================================================
# 7. ID conversion helpers
# =========================================================
def compact_ids_to_group_ids(compact_ids, config):
    compact_ids = np.asarray(compact_ids, dtype=np.int64)
    return config["group_id_by_compact_id"][compact_ids]


def compact_ids_to_source_ids(compact_ids, config):
    compact_ids = np.asarray(compact_ids, dtype=np.int64)
    compact_to_source = np.asarray(config["compact_to_source"], dtype=np.int64)
    return compact_to_source[compact_ids]


# =========================================================
# 8. Score helpers
# =========================================================
def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits[None, :]
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def classify_probabilities(probabilities, config, fall_like_ids=None, thresholds=None):
    del fall_like_ids, thresholds
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1:
        raise ValueError("classify_probabilities expects a 1D probability vector")

    top1_label_id = int(np.argmax(probabilities))
    top1_source_id = int(config["compact_to_source"][top1_label_id])
    top3_label_ids = [int(label_id) for label_id in np.argsort(probabilities)[-3:][::-1]]
    top3_source_ids = [int(config["compact_to_source"][label_id]) for label_id in top3_label_ids]
    fall_score = float(probabilities[config["positive_class_id"]])
    group_label = config["group_name_by_compact_id"][top1_label_id]
    return {
        "top1_label_id": top1_label_id,
        "top1_source_id": top1_source_id,
        "top3_label_ids": top3_label_ids,
        "top3_source_ids": top3_source_ids,
        "fall_score": round(fall_score, 6),
        "group_label": group_label,
        "internal_state": group_label,
        "external_alarm": group_label,
    }


# =========================================================
# 9. Hard-negative mining
# =========================================================
def find_hard_negative_info(scores, labels, config):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = softmax(scores)
    predictions = np.argmax(probabilities, axis=1)

    true_group_ids = compact_ids_to_group_ids(labels, config)
    pred_group_ids = compact_ids_to_group_ids(predictions, config)

    normal_id = GROUP_NAME_TO_ID["normal"]
    fall_like_id = GROUP_NAME_TO_ID["fall-like"]
    fall_id = GROUP_NAME_TO_ID["fall"]

    fall_mask = (true_group_ids == normal_id) & (pred_group_ids == fall_id)
    fall_like_mask = (true_group_ids == normal_id) & (pred_group_ids == fall_like_id)
    all_mask = fall_mask | fall_like_mask

    return {
        "all_indices": np.where(all_mask)[0].astype(np.int64),
        "fall_indices": np.where(fall_mask)[0].astype(np.int64),
        "fall_like_indices": np.where(fall_like_mask)[0].astype(np.int64),
    }


# =========================================================
# 10. Report helpers
# =========================================================
def _safe_rate(mask):
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask))


def _build_confusion(true_ids, pred_ids, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_id, pred_id in zip(true_ids, pred_ids):
        matrix[int(true_id), int(pred_id)] += 1
    return matrix


def _label_name(source_id, label_names):
    if source_id < len(label_names):
        return label_names[source_id]
    return "class_{}".format(source_id)


# =========================================================
# 11. Evaluation reports
# =========================================================
def generate_report(scores, labels, config, label_names=None):
    labels = np.asarray(labels, dtype=np.int64)
    label_names = label_names or list(DEFAULT_LABEL_NAMES)
    probabilities = softmax(scores)
    predictions = np.argmax(probabilities, axis=1)

    positive_class_id = config["positive_class_id"]
    fall_scores = probabilities[:, positive_class_id]
    true_group_ids = compact_ids_to_group_ids(labels, config)
    pred_group_ids = compact_ids_to_group_ids(predictions, config)
    true_source_ids = compact_ids_to_source_ids(labels, config)
    pred_source_ids = compact_ids_to_source_ids(predictions, config)

    fall_id = GROUP_NAME_TO_ID["fall"]
    fall_like_id = GROUP_NAME_TO_ID["fall-like"]
    normal_id = GROUP_NAME_TO_ID["normal"]

    pred_fall_mask = pred_group_ids == fall_id
    pred_fall_like_mask = pred_group_ids == fall_like_id
    true_fall_mask = true_group_ids == fall_id
    true_fall_like_mask = true_group_ids == fall_like_id
    true_normal_mask = true_group_ids == normal_id

    fall_precision = (
        float(np.sum(pred_fall_mask & true_fall_mask)) / float(np.sum(pred_fall_mask))
        if np.any(pred_fall_mask) else 0.0
    )
    fall_recall = _safe_rate(pred_fall_mask[true_fall_mask]) if np.any(true_fall_mask) else 0.0
    normal_to_fall_rate = _safe_rate(pred_fall_mask[true_normal_mask]) if np.any(true_normal_mask) else 0.0
    normal_to_fall_like_rate = _safe_rate(pred_fall_like_mask[true_normal_mask]) if np.any(true_normal_mask) else 0.0
    fall_like_to_fall_rate = _safe_rate(pred_fall_mask[true_fall_like_mask]) if np.any(true_fall_like_mask) else 0.0
    compact_top1_accuracy = _safe_rate(predictions == labels) if len(labels) else 0.0

    group_confusion = _build_confusion(true_group_ids, pred_group_ids, len(GROUP_LABELS))
    compact_confusion = _build_confusion(labels, predictions, config["num_classes"])

    source_class_metrics = []
    source_metrics_map = {}
    for source_id in config["retained_source_ids"]:
        class_mask = true_source_ids == source_id
        metric = {
            "source_id": int(source_id),
            "compact_id": int(config["source_to_compact"][source_id]),
            "label_name": _label_name(source_id, label_names),
            "group": config["group_by_source_id"][source_id],
            "count": int(np.sum(class_mask)),
            "predicted_fall_rate": round(_safe_rate(pred_fall_mask[class_mask]) if np.any(class_mask) else 0.0, 6),
            "predicted_fall_like_rate": round(_safe_rate(pred_fall_like_mask[class_mask]) if np.any(class_mask) else 0.0, 6),
            "predicted_normal_rate": round(_safe_rate(pred_group_ids[class_mask] == normal_id) if np.any(class_mask) else 0.0, 6),
            "mean_fall_score": round(float(np.mean(fall_scores[class_mask])) if np.any(class_mask) else 0.0, 6),
        }
        source_class_metrics.append(metric)
        source_metrics_map[str(source_id)] = metric

    squat_metric = source_metrics_map.get(str(config["squat_source_id"]), {})
    squat_to_fall_rate = float(squat_metric.get("predicted_fall_rate", 0.0))
    squat_count = int(squat_metric.get("count", 0))

    return {
        "mode": "retained_eval",
        "retained_source_ids": [int(source_id) for source_id in config["retained_source_ids"]],
        "fall_like_source_ids": [int(source_id) for source_id in config["fall_like_source_ids"]],
        "positive_source_id": int(config["positive_source_id"]),
        "positive_label_name": _label_name(config["positive_source_id"], label_names),
        "group_labels": list(GROUP_LABELS),
        "compact_mapping": {
            "compact_to_source": [int(source_id) for source_id in config["compact_to_source"]],
            "source_to_compact": {
                str(source_id): int(compact_id)
                for source_id, compact_id in config["source_to_compact"].items()
            },
        },
        "group_confusion": {
            "labels": list(GROUP_LABELS),
            "matrix": group_confusion.tolist(),
        },
        "compact_confusion": {
            "matrix": compact_confusion.tolist(),
        },
        "metrics": {
            "compact_top1_accuracy": round(float(compact_top1_accuracy), 6),
            "fall_precision": round(float(fall_precision), 6),
            "fall_recall": round(float(fall_recall), 6),
            "normal_to_fall_rate": round(float(normal_to_fall_rate), 6),
            "normal_to_fall_like_rate": round(float(normal_to_fall_like_rate), 6),
            "fall_like_to_fall_rate": round(float(fall_like_to_fall_rate), 6),
            "predicted_fall_count": int(np.sum(pred_fall_mask)),
            "predicted_fall_like_count": int(np.sum(pred_fall_like_mask)),
            "true_fall_count": int(np.sum(true_fall_mask)),
            "true_fall_like_count": int(np.sum(true_fall_like_mask)),
            "true_normal_count": int(np.sum(true_normal_mask)),
            "squat_source_id": int(config["squat_source_id"]),
            "squat_label_name": _label_name(config["squat_source_id"], label_names),
            "squat_sample_count": squat_count,
            "squat_to_fall_rate": round(float(squat_to_fall_rate), 6),
        },
        "monitored_source_metrics": {
            str(source_id): source_metrics_map[str(source_id)]
            for source_id in config["monitored_source_ids"]
            if str(source_id) in source_metrics_map
        },
        "source_class_metrics": source_class_metrics,
        "prediction_examples": {
            "top1_source_ids_head": [int(source_id) for source_id in pred_source_ids[:20]],
        },
    }


def generate_shadow_ood_report(scores, source_labels, config, label_names=None):
    source_labels = np.asarray(source_labels, dtype=np.int64)
    label_names = label_names or list(DEFAULT_LABEL_NAMES)
    probabilities = softmax(scores)
    predictions = np.argmax(probabilities, axis=1)
    pred_group_ids = compact_ids_to_group_ids(predictions, config)
    fall_scores = probabilities[:, config["positive_class_id"]]

    fall_id = GROUP_NAME_TO_ID["fall"]
    fall_like_id = GROUP_NAME_TO_ID["fall-like"]
    normal_id = GROUP_NAME_TO_ID["normal"]

    source_class_metrics = []
    for source_id in sorted({int(value) for value in source_labels.tolist()}):
        class_mask = source_labels == source_id
        source_class_metrics.append({
            "source_id": int(source_id),
            "label_name": _label_name(source_id, label_names),
            "count": int(np.sum(class_mask)),
            "predicted_fall_rate": round(_safe_rate(pred_group_ids[class_mask] == fall_id), 6),
            "predicted_fall_like_rate": round(_safe_rate(pred_group_ids[class_mask] == fall_like_id), 6),
            "predicted_normal_rate": round(_safe_rate(pred_group_ids[class_mask] == normal_id), 6),
            "mean_fall_score": round(float(np.mean(fall_scores[class_mask])) if np.any(class_mask) else 0.0, 6),
        })

    return {
        "mode": "shadow_ood",
        "metrics": {
            "predicted_fall_rate": round(_safe_rate(pred_group_ids == fall_id), 6),
            "predicted_fall_like_rate": round(_safe_rate(pred_group_ids == fall_like_id), 6),
            "predicted_normal_rate": round(_safe_rate(pred_group_ids == normal_id), 6),
            "mean_fall_score": round(float(np.mean(fall_scores)) if len(fall_scores) else 0.0, 6),
            "sample_count": int(len(source_labels)),
        },
        "source_class_metrics": source_class_metrics,
    }


# =========================================================
# 12. Model-selection and JSON helpers
# =========================================================
def report_sort_key(report, shadow_report=None):
    metrics = report["metrics"]
    shadow_metrics = (shadow_report or {}).get("metrics", {})
    return (
        -float(shadow_metrics.get("predicted_fall_rate", 0.0)),
        -float(shadow_metrics.get("predicted_fall_like_rate", 0.0)),
        -float(metrics.get("normal_to_fall_rate", 0.0)),
        -float(metrics.get("fall_like_to_fall_rate", 0.0)),
        float(metrics.get("fall_precision", 0.0)),
        float(metrics.get("fall_recall", 0.0)),
        float(metrics.get("compact_top1_accuracy", 0.0)),
    )


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)
