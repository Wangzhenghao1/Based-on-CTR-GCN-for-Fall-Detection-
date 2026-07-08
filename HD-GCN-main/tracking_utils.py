import numpy as np


TORSO_JOINTS = (5, 6, 11, 12)


def _bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def _bbox_scale(box):
    width = max(1.0, float(box[2] - box[0]))
    height = max(1.0, float(box[3] - box[1]))
    return float(np.sqrt(width * width + height * height))


def _torso_center(person, keypoint_score_thr=0.2):
    keypoints = person["keypoints"]
    scores = person.get("keypoint_scores")
    valid_points = []
    for index in TORSO_JOINTS:
        if scores is not None and scores[index] < keypoint_score_thr:
            continue
        if keypoints[index][0] > 1 and keypoints[index][1] > 1:
            valid_points.append(keypoints[index])
    if valid_points:
        return np.mean(valid_points, axis=0).astype(np.float32)
    bbox = person["bbox"]
    return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float32)


def _build_detection(person, keypoint_score_thr):
    bbox = person["bbox"].astype(np.float32)
    return {
        "person": person,
        "bbox": bbox,
        "center": _torso_center(person, keypoint_score_thr),
        "scale": _bbox_scale(bbox),
        "keypoints": person["keypoints"].astype(np.float32),
        "scores": person["keypoint_scores"].astype(np.float32),
    }


def _pose_shape_distance(track, det, keypoint_score_thr):
    valid = (track["scores"] >= keypoint_score_thr) & (det["scores"] >= keypoint_score_thr)
    if np.count_nonzero(valid) < 4:
        return None
    track_pose = (track["keypoints"][valid] - track["center"]) / max(1.0, track["scale"])
    det_pose = (det["keypoints"][valid] - det["center"]) / max(1.0, det["scale"])
    return float(np.mean(np.linalg.norm(track_pose - det_pose, axis=1)))


def _predicted_center(track, reid_mode):
    max_steps = 8 if reid_mode else 3
    steps = min(max(1, int(track["missed"]) + 1), max_steps)
    return track["center"] + track["velocity"] * float(steps)


def _candidate_cost(track, det, dist_thr, max_missed, keypoint_score_thr, reid_mode):
    center_dist = float(np.linalg.norm(det["center"] - _predicted_center(track, reid_mode)))
    iou = _bbox_iou(track["bbox"], det["bbox"])
    pose_dist = _pose_shape_distance(track, det, keypoint_score_thr)
    size_gate = 0.75 * max(track["scale"], det["scale"])
    if reid_mode:
        center_gate = max(float(dist_thr) * 2.5, size_gate * 1.8)
        iou_gate = 0.01
        pose_gate = 0.48
        max_cost = 1.85
    else:
        center_gate = max(float(dist_thr), size_gate)
        iou_gate = 0.03
        pose_gate = 0.40
        max_cost = 1.45
    pose_ok = pose_dist is not None and pose_dist <= pose_gate
    if center_dist > center_gate and iou < iou_gate and not pose_ok:
        return None
    motion_cost = center_dist / max(center_gate, 1.0)
    iou_cost = 1.0 - iou
    pose_cost = 0.8 if pose_dist is None else min(pose_dist / pose_gate, 2.0)
    missed_penalty = min(float(track["missed"]) / max(1.0, float(max_missed)), 3.0) * 0.08
    cost = 0.52 * motion_cost + 0.30 * iou_cost + 0.18 * pose_cost + missed_penalty
    if cost > max_cost:
        return None
    return cost


def _greedy_match(track_ids, det_indices, tracks_state, detections, dist_thr, max_missed, keypoint_score_thr, reid_mode):
    candidates = []
    for track_id in track_ids:
        for det_index in det_indices:
            cost = _candidate_cost(
                tracks_state[track_id], detections[det_index], dist_thr, max_missed, keypoint_score_thr, reid_mode
            )
            if cost is not None:
                candidates.append((cost, track_id, det_index))
    candidates.sort(key=lambda item: item[0])
    matched_tracks = set()
    matched_dets = set()
    matches = []
    for cost, track_id, det_index in candidates:
        if track_id in matched_tracks or det_index in matched_dets:
            continue
        matched_tracks.add(track_id)
        matched_dets.add(det_index)
        matches.append((track_id, det_index, cost))
    return matches


def _append_observation(output_tracks, track_id, frame_idx, person):
    output_tracks.setdefault(track_id, [])
    output_tracks[track_id].append({"frame": frame_idx, "data": person})


def _update_track(track, det, frame_idx):
    frame_gap = max(1, frame_idx - int(track["last_frame"]))
    new_velocity = (det["center"] - track["center"]) / float(frame_gap)
    if track["missed"] > 0:
        track["velocity"] = 0.55 * new_velocity + 0.45 * track["velocity"]
    else:
        track["velocity"] = 0.75 * new_velocity + 0.25 * track["velocity"]
    track["center"] = det["center"]
    track["bbox"] = det["bbox"]
    track["scale"] = det["scale"]
    track["keypoints"] = det["keypoints"]
    track["scores"] = det["scores"]
    track["missed"] = 0
    track["hits"] += 1
    track["total_hits"] += 1
    track["last_frame"] = frame_idx


def _create_track(det, frame_idx, min_hits):
    return {
        "center": det["center"],
        "velocity": np.array([0.0, 0.0], dtype=np.float32),
        "bbox": det["bbox"],
        "scale": det["scale"],
        "keypoints": det["keypoints"],
        "scores": det["scores"],
        "missed": 0,
        "hits": 1,
        "total_hits": 1,
        "last_frame": frame_idx,
        "confirmed": min_hits <= 1,
    }


def simple_tracking(
    pose_results_list,
    dist_thr=100,
    max_missed=20,
    min_hits=3,
    score_thr=0.4,
    reid_max_missed=None,
    keypoint_score_thr=0.2,
    log_prefix="tracking",
):
    if reid_max_missed is None:
        reid_max_missed = max(max_missed * 3, max_missed + 30)
    next_id = 0
    tracks_state = {}
    output_tracks = {}
    reconnected = 0
    expired = 0
    print("Running occlusion-aware tracking...")
    for frame_idx, persons in enumerate(pose_results_list):
        valid_persons = [person for person in (persons or []) if person["bbox"][4] >= score_thr]
        detections = [_build_detection(person, keypoint_score_thr) for person in valid_persons]
        unmatched_dets = set(range(len(detections)))
        matched_track_ids = set()
        active_ids = [track_id for track_id, track in tracks_state.items() if track["missed"] <= max_missed]
        active_matches = _greedy_match(
            active_ids, sorted(unmatched_dets), tracks_state, detections, dist_thr, max_missed, keypoint_score_thr, False
        )
        for track_id, det_index, _cost in active_matches:
            det = detections[det_index]
            _update_track(tracks_state[track_id], det, frame_idx)
            if tracks_state[track_id]["hits"] >= min_hits:
                tracks_state[track_id]["confirmed"] = True
            if tracks_state[track_id]["confirmed"]:
                det["person"]["track_id"] = track_id
                _append_observation(output_tracks, track_id, frame_idx, det["person"])
            matched_track_ids.add(track_id)
            unmatched_dets.discard(det_index)
        lost_ids = [
            track_id for track_id, track in tracks_state.items()
            if track_id not in matched_track_ids
            and track["confirmed"]
            and max_missed < track["missed"] <= reid_max_missed
        ]
        reid_matches = _greedy_match(
            lost_ids, sorted(unmatched_dets), tracks_state, detections, dist_thr, max_missed, keypoint_score_thr, True
        )
        for track_id, det_index, _cost in reid_matches:
            det = detections[det_index]
            _update_track(tracks_state[track_id], det, frame_idx)
            tracks_state[track_id]["confirmed"] = True
            det["person"]["track_id"] = track_id
            _append_observation(output_tracks, track_id, frame_idx, det["person"])
            matched_track_ids.add(track_id)
            unmatched_dets.discard(det_index)
            reconnected += 1
        for track_id in list(tracks_state.keys()):
            if track_id in matched_track_ids:
                continue
            track = tracks_state[track_id]
            track["missed"] += 1
            track["hits"] = 0
            track["center"] = track["center"] + track["velocity"] * float(min(track["missed"], 3)) * 0.25
            if track["missed"] > reid_max_missed:
                del tracks_state[track_id]
                expired += 1
        for det_index in sorted(unmatched_dets):
            det = detections[det_index]
            track_id = next_id
            next_id += 1
            tracks_state[track_id] = _create_track(det, frame_idx, min_hits)
            if tracks_state[track_id]["confirmed"]:
                det["person"]["track_id"] = track_id
                _append_observation(output_tracks, track_id, frame_idx, det["person"])
    clean_pose_results = []
    for persons in pose_results_list:
        clean_frame_persons = []
        for person in persons or []:
            if "track_id" in person:
                clean_frame_persons.append(person)
        clean_pose_results.append(clean_frame_persons)
    print("Tracking done. Valid IDs: {}, reconnected: {}, expired: {}, alive: {}".format(
        len(output_tracks), reconnected, expired, len(tracks_state)
    ))
    return clean_pose_results, output_tracks
