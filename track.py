from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import torch

from lib.utils.opts import opts
from lib.utils.utils import _normalize_optim_config, cfg_get

from lib.models.stNet import get_det_net, load_model
from lib.dataset.coco import COCO

from lib.external.nms import soft_nms

from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process

from lib.utils.sort import *
import torch.nn.functional as F

import cv2

from progress.bar import Bar
from log_helper import log_print
import motmetrics as mm
import pickle
import yaml
import pandas as pd
import glob

CONFIDENCE_thres = 0.3
COLORS = [(255, 0, 0)]

FONT = cv2.FONT_HERSHEY_SIMPLEX

def cv2_demo(frame, detections):
    det = []
    for i in range(detections.shape[0]):
        if detections[i, 4] >= CONFIDENCE_thres:
            pt = detections[i, :]
            cv2.rectangle(frame,(int(pt[0])-4, int(pt[1])-4),(int(pt[2])+4, int(pt[3])+4),COLORS[0], 2)
            cv2.putText(frame, str(pt[4]), (int(pt[0]), int(pt[1])), FONT, 1, (0, 255, 0), 1)
            det.append([int(pt[0]), int(pt[1]),int(pt[2]), int(pt[3]),detections[i, 4]])
    return frame, det

def process(model, image, return_time):
    with torch.no_grad():
        output = model(image)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg']
        forward_time = time.time()
        dets = ctdet_decode(hm, wh, reg=reg)
    if return_time:
        return output, dets, forward_time
    else:
        return output, model.feature, dets

def post_process(dets, meta, num_classes=1, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        dets[0][j][:, :4] /= scale
    return dets[0]

def pre_process(image, scale=1):
    height, width = image.shape[2:4]
    new_height = int(height * scale)
    new_width = int(width * scale)

    inp_height, inp_width = height, width
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    meta = {'c': c, 's': s,
            'out_height': inp_height ,
            'out_width': inp_width}
    return meta

def merge_outputs(detections, num_classes ,max_per_image):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

        soft_nms(results[j], Nt=0.5, method=2)

    scores = np.hstack(
      [results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results

def extract_features_for_class_1(ret, feature_map):
    h_map, w_map = feature_map.shape[2:]
    f1 = []

    if 1 in ret:
        boxes = ret[1]
        for box in boxes:
            x_min, y_min, x_max, y_max, conf = box

            if x_max <= x_min or y_max <= y_min:
                f1.append(torch.zeros(feature_map.size(1)))
                continue

            x_min = int(max(0, x_min * w_map / feature_map.size(-1)))
            x_max = int(min(w_map, x_max * w_map / feature_map.size(-1)))
            y_min = int(max(0, y_min * h_map / feature_map.size(-2)))
            y_max = int(min(h_map, y_max * h_map / feature_map.size(-2)))

            if x_max <= x_min or y_max <= y_min:
                f1.append(torch.zeros(feature_map.size(1)))
                continue

            cropped_feature = feature_map[:, :, y_min:y_max, x_min:x_max]

            if cropped_feature.numel() == 0:
                feature_vector = torch.zeros(feature_map.size(1))
            else:
                pooled_feature = F.adaptive_avg_pool2d(cropped_feature, (1, 1))
                feature_vector = pooled_feature.view(-1)

            f1.append(feature_vector)

    ret['f1'] = f1
    return ret

def test(opt, split, modelPath, show_flag, results_name):

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    print(opt.model_name)

    dataset = COCO(opt, split)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    model = get_det_net({'hm': dataset.num_classes, 'wh': 2, 'reg': 2}, opt.model_name)  # 建立模型
    model = load_model(model, modelPath)
    model = model.cuda()
    model.eval()

    results = {}
    res = {}
    return_time = False
    scale = 1
    num_classes = dataset.num_classes
    max_per_image = opt.K

    num_iters = len(data_loader)
    bar = Bar('processing', max=num_iters)

    for ind, (file_path, img_id, pre_processed_images) in enumerate(data_loader):
        if(ind>len(data_loader)-1):
            break

        bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters,total=bar.elapsed_td, eta=bar.eta_td
        )

        #read images
        detection = []
        meta = pre_process(pre_processed_images['input'], scale)
        image = pre_processed_images['input'].cuda()
        img = pre_processed_images['imgOri'].squeeze().numpy()

        #detection
        output, feature, dets = process(model, image, return_time)
        #POST PROCESS
        dets = post_process(dets, meta, num_classes)
        detection.append(dets)
        ret = merge_outputs(detection, num_classes, max_per_image)

        results[img_id.numpy().astype(np.int32)[0]] = ret
        ret = extract_features_for_class_1(ret, feature)
        res[file_path[0]] = ret
        bar.next()
    bar.finish()

    videos_cells_by_t = build_videos_cells_by_t(res = res)
    optim_config = _normalize_optim_config(opt.track_cfg)

    save_dir = 'tracking_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for video_name, cells_by_t in videos_cells_by_t.items():
        # 1) 构建图
        G = create_graph(cells_by_t, optim_config)
        
        # 2) ILP
        solution_edges_initial = solve_ilp(G, optim_config)
        if solution_edges_initial is None:
            print("No ILP solution for", video_name)
            continue
        
        # 3) parse初步轨迹
        pred_tracks_initial, frame_tracks = parse_prediction(solution_edges_initial)
        print(f"Initial ILP found {len(pred_tracks_initial)} tracks for {video_name}.")

        # 4) 插入虚拟节点，衔接碎片
        pred_tracks_final = insert_virtual_nodes(pred_tracks_initial, cells_by_t, optim_config)

        # 5) 去掉虚拟节点 & 过滤最短轨迹
        pred_tracks_post = post_process_tracks(pred_tracks_final, optim_config)

        frame_tracks = rebuild_frame_tracks_from_pred_tracks(pred_tracks_post)

        # 保存跟踪结果到 txt 文件
        txt_path = os.path.join(save_dir, f'{video_name}.txt')
        with open(txt_path, 'w') as f:
            for frame_id in sorted(frame_tracks.keys()):
                tracks_in_frame = frame_tracks[frame_id]
                for track in tracks_in_frame:
                    tid, x, y, w, h, isv = track[:6]
                    conf = None
                    if len(track) >= 7:
                        maybe_7th = track[6]
                        if isinstance(maybe_7th, (float,int)):
                            conf = maybe_7th
                    
                    isv_str = "1" if isv else "0"
                    
                    if conf is not None:
                        f.write(f"{frame_id},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{isv_str},{conf:.2f}\n")
                    else:
                        f.write(f"{frame_id},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{isv_str}\n")


def remove_tail_virtual_nodes(predictions):
    df = predictions.copy()
    df['is_virt'] = df['is_virt'].astype(int)
    
    df.sort_values(by=['id','frame'], ascending=[True,False], inplace=True)
    
    group_ids = df['id'].unique()
    
    keep_index = []
    
    for tid in group_ids:
        sub = df[df['id']==tid]
        sub_indices = sub.index.tolist()
        sub_frames  = sub['frame'].tolist()
        sub_virt    = sub['is_virt'].tolist()
        
        # 逆序遍历
        remove_mode = True
        for i, row_idx in enumerate(sub_indices):
            if not remove_mode:
                keep_index.append(row_idx)
                continue
            
            if sub_virt[i] == 1:
                pass
            else:
                remove_mode = False
                keep_index.append(row_idx)
        
    filtered_df = df.loc[keep_index].copy()
    
    filtered_df.sort_values(by=['id','frame'], ascending=[True,True], inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)
    
    return filtered_df


def post_process_tracks(pred_tracks, optim_config):
    min_track_length = optim_config.get('min_track_length', 2)
    final_tracks = {}
    
    for tid, frames_dict in pred_tracks.items():
        real_count = 0
        for fr, box in frames_dict.items():
            if len(box) >= 5:

                isv = box[4]

                if isinstance(isv, bool):
                    if not isv:
                        real_count += 1
                else:
                    real_count += 1
            else:
                real_count += 1
        
        if real_count >= min_track_length:
            final_tracks[tid] = frames_dict
    
    return final_tracks


class CVKalman:
    def __init__(self, init_pos, init_vel=(0., 0.), dt=1.0,
                 sigma_p=50.0, sigma_v=50.0, sigma_a=1.0, sigma_r=3.0):
        self.dt = dt
        self.x = np.array([init_pos[0], init_pos[1],
                           init_vel[0], init_vel[1]], dtype=float)

        self.P = np.diag([sigma_p, sigma_p, sigma_v, sigma_v])

        self.sigma_a = sigma_a

        self.R = np.diag([sigma_r ** 2, sigma_r ** 2])

    def _F(self, dt):
        return np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)

    def _Q(self, dt):
        q = self.sigma_a ** 2
        return q * np.array(
            [[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
             [0, dt ** 4 / 4, 0, dt ** 3 / 2],
             [dt ** 3 / 2, 0, dt ** 2, 0],
             [0, dt ** 3 / 2, 0, dt ** 2]]
        )

    def predict(self, dt=None):
        if dt is None:
            dt = self.dt
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        return self.x.copy()

    def update(self, z):
        """
        z : (x,y)
        """
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=float)
        y = np.array(z) - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    @property
    def pos(self):
        return self.x[:2]

    @property
    def vel(self):
        return self.x[2:]


def replay_track_with_kf(frames_map, dt=1.0):
    frames_sorted = sorted(frames_map.keys())
    if not frames_sorted:
        return None, {}

    f0 = frames_sorted[0]
    x0, y0, w0, h0, isv0 = frames_map[f0]
    cvkf = CVKalman(init_pos=(x0 + w0 / 2, y0 + h0 / 2))

    frames_map_out = {}

    for f in frames_sorted:
        x, y, w, h, isv = frames_map[f]
        cx, cy = x + w / 2, y + h / 2
        cvkf.predict(dt)
        cvkf.update((cx, cy))
        vx, vy = cvkf.vel
        cx_est, cy_est = cvkf.pos
        x_est, y_est = cx_est - w / 2, cy_est - h / 2
        frames_map_out[f] = (x_est, y_est, w, h, isv, vx, vy)

    return cvkf, frames_map_out


def insert_virtual_nodes(pred_tracks_initial,
                         cells_by_t=None,
                         optim_config=None):
    if optim_config is None:
        optim_config = {}

    # 参数
    Nt_max        = optim_config.get('fragment_length_max', 15)   # N_τ
    max_gap       = optim_config.get('max_gap', 10)               # 允许插值帧数
    Td            = optim_config.get('Td', 25.0)                  # 归一化位置阈值
    Tv            = optim_config.get('Tv', 6.0)                   # 归一化速度阈值
    wp            = optim_config.get('wp', 0.7)                   # 位置权重
    wv            = optim_config.get('wv', 0.3)                   # 速度权重
    delta_FS      = optim_config.get('delta_FS', 0.4)             # 拼接阈值

    pred_tracks, kf_dict = {}, {}
    for tid, frames in pred_tracks_initial.items():
        kf, frames_out = replay_track_with_kf(frames)
        pred_tracks[tid] = frames_out
        kf_dict[tid] = kf

    frag_ids = []
    for tid, fr_dict in pred_tracks.items():
        frames_sorted = sorted(fr_dict.keys())
        if len(frames_sorted) < Nt_max:
            frag_ids.append(tid)

    used = set()
    merged = True
    while merged:
        merged = False
        for tid_A in frag_ids:
            if tid_A in used:
                continue

            frames_A = sorted(pred_tracks[tid_A].keys())
            tA_end   = frames_A[-1]
            kf_A     = kf_dict[tid_A]
            pA_end   = np.array(kf_A.pos)
            vA_end   = np.array(kf_A.vel)

            best_tidB, best_score, best_gap = None, -np.inf, None

            for tid_B in frag_ids:
                if tid_B == tid_A or tid_B in used:
                    continue
                frames_B = sorted(pred_tracks[tid_B].keys())
                tB_start = frames_B[0]
                dt_gap   = tB_start - tA_end
                if dt_gap <= 0 or dt_gap > max_gap:
                    continue

                # 2.1 预测到 tB_start
                kf_tmp = CVKalman(pA_end, vA_end)
                kf_tmp.predict(dt_gap)
                pA_pred = kf_tmp.pos
                vA_pred = kf_tmp.vel

                # 2.2 计算 Δd / Δv
                xB, yB, wB, hB, *_ = pred_tracks[tid_B][tB_start]
                pB_start = np.array([xB + wB / 2, yB + hB / 2])

                # 速度：用 tid_B 的第一帧 KF 速度；若无，设 0
                vxB = pred_tracks[tid_B][tB_start][5] if len(pred_tracks[tid_B][tB_start]) > 5 else 0.
                vyB = pred_tracks[tid_B][tB_start][6] if len(pred_tracks[tid_B][tB_start]) > 6 else 0.
                vB_start = np.array([vxB, vyB])

                delta_d = np.linalg.norm(pA_pred - pB_start)
                delta_v = np.linalg.norm(vA_pred - vB_start)

                # 2.3 评分
                S = wp * (1 - delta_d / Td) + wv * (1 - delta_v / Tv)
                if S > best_score:
                    best_score, best_tidB, best_gap = S, tid_B, dt_gap

            # ---------- 3. 拼接 ----------
            if best_tidB is not None and best_score > delta_FS:
                merged = True
                tid_B = best_tidB
                gap   = best_gap

                kf_tmp = CVKalman(pA_end, vA_end)
                vir_frames = []
                for k in range(1, gap):
                    t = tA_end + k
                    kf_tmp.predict()
                    cx, cy = kf_tmp.pos
                    vx, vy = kf_tmp.vel
                    # 用上一帧的 w,h 作为近似
                    _, _, wA, hA, *_ = pred_tracks[tid_A][tA_end]
                    x, y = cx - wA / 2, cy - hA / 2
                    vir_frames.append((t, (x, y, wA, hA, True, vx, vy)))

                # 写入虚拟节点
                for t, box in vir_frames:
                    pred_tracks[tid_A][t] = box

                # 把 B 整段并入 A
                for tB, boxB in pred_tracks[tid_B].items():
                    pred_tracks[tid_A][tB] = boxB

                # 更新末端 KF
                kf_dict[tid_A] = kf_dict[tid_B]
                used.add(tid_B)

    return {tid: frames for tid, frames in pred_tracks.items() if tid not in used}


def rebuild_frame_tracks_from_pred_tracks(pred_tracks):
    frame_tracks = {}
    for tid, frames_dict in pred_tracks.items():
        for fr, box in frames_dict.items():
            if len(box) < 5:
                x, y, w, h = box[:4]
                is_v = False
                new_track_tuple = (tid, x, y, w, h, is_v)
            else:
                x, y, w, h, is_v = box[:5]
                vx, vy = 0.0, 0.0
                if len(box) >= 7:
                    vx, vy = box[5], box[6]
                new_track_tuple = (tid, x, y, w, h, is_v)
            
            if fr not in frame_tracks:
                frame_tracks[fr] = []
            frame_tracks[fr].append(new_track_tuple)
    
    return frame_tracks

def create_graph(cells_by_t, optim_config):
    import networkx as nx
    import numpy as np
    from scipy.spatial import KDTree
    from sklearn.metrics.pairwise import cosine_similarity

    G = nx.DiGraph()
    
    # 从优化配置中获取参数
    max_children = optim_config.get('max_children', 3)
    distance_threshold = optim_config.get('distance_threshold', 10)
    confidence_threshold = optim_config.get('confidence_threshold', 0.3)
    confidence_function = optim_config.get('confidence_function', 'quadratic')
    weight_position = optim_config.get('weight_position', 1.0)
    weight_velocity = optim_config.get('weight_velocity', 1.0)
    weight_appearance = optim_config.get('weight_appearance', 1.0)
    gaussian_sigma = optim_config.get('gaussian_sigma', 1.0)
    epsilon = 1

    # 添加节点
    for t in cells_by_t:
        for cell in cells_by_t[t]:
            if len(cell) == 5:
                cell_id, position, velocity, confidence, appearance = cell
            elif len(cell) == 4:
                cell_id, position, velocity, confidence = cell
                appearance = None
            else:
                cell_id, position, velocity = cell[:3]
                confidence = 1.0
                appearance = None
            if confidence < confidence_threshold:
                continue
            G.add_node((t, cell_id), pos=position, velocity=velocity, confidence=confidence, appearance=appearance)
    
    # 添加边(相邻帧)
    frames = sorted(cells_by_t.keys())
    from scipy.spatial import KDTree
    for idx, t in enumerate(frames):
        t_next = t + 1
        if t_next in cells_by_t:
            current_frame = [cell for cell in cells_by_t[t] if cell[3] >= confidence_threshold]
            next_frame = [cell for cell in cells_by_t[t_next] if cell[3] >= confidence_threshold]

            next_positions = [cell[1][:2] for cell in next_frame]
            next_ids = [cell[0] for cell in next_frame]
            next_velocities = [cell[2] for cell in next_frame]
            next_confidences = [cell[3] for cell in next_frame]
            next_appearances = [cell[4] if len(cell) > 4 else None for cell in next_frame]

            if next_positions:
                kd_tree = KDTree(next_positions)

                for current_cell in current_frame:
                    current_id = current_cell[0]
                    current_pos = current_cell[1][:2]
                    current_vel = current_cell[2]
                    current_conf = current_cell[3]
                    current_app = current_cell[4] if len(current_cell) > 4 else None

                    distances, indices = kd_tree.query(current_pos, k=max_children)
                    if not isinstance(distances, np.ndarray):
                        distances = [distances]
                        indices = [indices]

                    for distance, index in zip(distances, indices):
                        if distance < distance_threshold:
                            next_id = next_ids[index]
                            next_pos = next_positions[index]
                            next_vel_val = next_velocities[index]
                            next_conf_val = next_confidences[index]
                            next_app_val = next_appearances[index]

                            delta_p = np.linalg.norm(np.array(next_pos) - np.array(current_pos))
                            if gaussian_sigma == 0:
                                sim_p = 1 / (delta_p + epsilon)
                            else:
                                sim_p = np.exp(- (delta_p ** 2) / (2 * gaussian_sigma ** 2))

                            delta_v = np.linalg.norm(next_vel_val - current_vel)
                            if gaussian_sigma == 0:
                                sim_v = 1 / (delta_v + epsilon)
                            else:
                                sim_v = np.exp(- (delta_v ** 2) / (2 * gaussian_sigma ** 2))

                            if weight_appearance > 0 and current_app is not None and next_app_val is not None:
                                sim_app = cosine_similarity([current_app.cpu().numpy()], [next_app_val.cpu().numpy()])[0][0]
                            else:
                                sim_app = 0.0

                            if confidence_function == 'linear':
                                w_conf = current_conf * next_conf_val
                            elif confidence_function == 'quadratic':
                                w_conf = (current_conf * next_conf_val) ** 2
                            else:
                                w_conf = 1.0

                            weight = w_conf * (weight_position * sim_p + weight_velocity * sim_v + weight_appearance * sim_app)
                            G.add_edge((t, current_id), (t_next, next_id), weight=weight)
    return G

def solve_ilp(graph, optim_config):
    import pulp
    import numpy as np

    start_cost = optim_config.get('start_cost', 0.0)
    end_cost = optim_config.get('end_cost', 0.0)
    min_track_length = optim_config.get('min_track_length', None)
    max_track_length = optim_config.get('max_track_length', None)

    density_threshold = optim_config.get('density_threshold', 0.5)
    min_distance_threshold = optim_config.get('min_distance_threshold', 3.0)

    problem = pulp.LpProblem("TrackForming", pulp.LpMaximize)

    edges = {}
    positions = {}
    nodes = list(graph.nodes())

    for u, v in graph.edges():
        var_name = f"edge_{u}_{v}"
        edges[(u, v)] = pulp.LpVariable(var_name, cat=pulp.LpBinary)
        positions[u] = graph.nodes[u]['pos']
        positions[v] = graph.nodes[v]['pos']

    if start_cost > 0 or end_cost > 0:
        start_node = 'Start'
        end_node = 'End'
        graph.add_node(start_node)
        graph.add_node(end_node)
        for node in nodes:
            var_start = f"edge_{start_node}_{node}"
            var_end = f"edge_{node}_{end_node}"
            edges[(start_node, node)] = pulp.LpVariable(var_start, cat=pulp.LpBinary)
            edges[(node, end_node)] = pulp.LpVariable(var_end, cat=pulp.LpBinary)
            graph.add_edge(start_node, node, weight=start_cost)
            graph.add_edge(node, end_node, weight=end_cost)

    problem += pulp.lpSum(edges[e] * graph.edges[e]['weight'] for e in edges)

    # 入出度限制
    for node in nodes:
        incoming_edges = [edges[(u, node)] for u in graph.predecessors(node) if (u, node) in edges]
        outgoing_edges = [edges[(node, v)] for v in graph.successors(node) if (node, v) in edges]
        problem += pulp.lpSum(incoming_edges) <= 1, f"MaxIn_{node}"
        problem += pulp.lpSum(outgoing_edges) <= 1, f"MaxOut_{node}"

    if min_track_length is not None or max_track_length is not None:
        track_vars = {}
        track_id = 0

        def add_track_length_constraints(current_node, current_length):
            outgoing_edges = [edges[(current_node, v)] for v in graph.successors(current_node) if (current_node, v) in edges]
            problem += track_var >= current_length, f"TrackLength_{current_node}"
            for edge_var, successor_node in zip(outgoing_edges, graph.successors(current_node)):
                problem += edge_var <= track_var, f"EdgeTrack_{current_node}_{successor_node}"
                add_track_length_constraints(successor_node, current_length + 1)

        for node in nodes:
            incoming_edges = [edges[(u, node)] for u in graph.predecessors(node) if (u, node) in edges]
            if len(incoming_edges) == 0:
                track_var = pulp.LpVariable(f"Track_{track_id}", lowBound=0, cat=pulp.LpInteger)
                track_id += 1
                add_track_length_constraints(node, 1)
                if min_track_length is not None:
                    problem += track_var >= min_track_length, f"MinTrackLength_{node}"
                if max_track_length is not None:
                    problem += track_var <= max_track_length, f"MaxTrackLength_{node}"

    problem.solve()

    if pulp.LpStatus[problem.status] == "Optimal":
        # solution_edges: list of (u,v,pos_u,pos_v)
        selected_edges = [
            (u, v, positions[u], positions[v])
            for u, v in edges
            if edges[(u, v)].varValue > 0.5 and u in positions and v in positions
            if u != 'Start' and v != 'End'
        ]

        trajectories_with_edges = build_trajectories_with_edges(selected_edges)

        filtered_edges = []
        for traj_nodes, traj_positions, traj_edges in trajectories_with_edges:
            density = compute_track_density(traj_positions)
            total_displacement = np.linalg.norm(traj_positions[-1] - traj_positions[0])
            if density <= density_threshold and total_displacement >= min_distance_threshold:
                # 保留该轨迹的所有edges
                filtered_edges.extend(traj_edges)

        return filtered_edges
    else:
        return None

def parse_prediction(solution_edges):
    """ Parses the solution edges to create tracks with consistent IDs."""
    # 建立节点的前驱和后继映射
    node_successors = {}
    node_predecessors = {}
    nodes = set()
    node_centers = {}
    for u, v, pos_u, pos_v in solution_edges:
        node_successors[u] = v
        node_predecessors[v] = u
        nodes.add(u)
        nodes.add(v)
        node_centers[u] = pos_u
        node_centers[v] = pos_v

    # 查找起始节点（没有前驱的节点）
    start_nodes = [node for node in nodes if node not in node_predecessors]

    pred_tracks = {}
    frame_tracks = {}
    track_id_counter = 0
    visited_nodes = set()

    # 遍历每个起始节点，构建轨迹
    for start_node in start_nodes:
        node = start_node
        track_id = track_id_counter
        track_id_counter += 1
        while True:
            if node in visited_nodes:
                break
            visited_nodes.add(node)
            frame_id, cell_id = node
            center = node_centers.get(node)
            if center is None:
                break
            # 转换为左上角坐标
            tl_x = center[0] - center[2] / 2
            tl_y = center[1] - center[3] / 2
            # 更新 pred_tracks
            if track_id not in pred_tracks:
                pred_tracks[track_id] = {}
            pred_tracks[track_id][frame_id] = (tl_x, tl_y, center[2], center[3], False)
            # 更新 frame_tracks
            if frame_id not in frame_tracks:
                frame_tracks[frame_id] = []
            frame_tracks[frame_id].append((track_id, tl_x, tl_y, center[2], center[3], False))
            # 移动到下一个节点
            if node in node_successors:
                node = node_successors[node]
            else:
                break
    return pred_tracks, frame_tracks


def build_trajectories_with_edges(selected_edges):
    from collections import defaultdict
    import numpy as np

    graph_dict = defaultdict(list)
    nodes_set = set()

    # 构建图结构
    for u, v, pos_u, pos_v in selected_edges:
        graph_dict[u].append((v, pos_u, pos_v))
        nodes_set.add(u)
        nodes_set.add(v)

    # 入度统计
    in_degree = {n:0 for n in nodes_set}
    for u in graph_dict:
        for (vv, pos_u, pos_v) in graph_dict[u]:
            in_degree[vv] += 1
    start_nodes = [n for n in nodes_set if in_degree[n] == 0]

    trajectories = []
    for start in start_nodes:
        # 重建轨迹:从start出发
        # 找到start点pos: 使用graph_dict[start]的第一条边的pos_u作为起始点
        traj_nodes = [start]
        traj_positions = []
        traj_edges = []

        if start in graph_dict and len(graph_dict[start]) > 0:
            # 使用第一条边的 pos_u 作为起始点位置
            first_edge = graph_dict[start][0]
            pos_start = first_edge[1]
            traj_positions.append(pos_start)
        else:
            # 如果start没有后继边，找到start的pos
            pos_start = None
            for (uu, vv, pu, pv) in selected_edges:
                if uu == start:
                    pos_start = pu
                    break
                if vv == start:
                    pos_start = pv
                    break
            if pos_start is None:
                # 无法找到start点pos,跳过
                continue
            traj_positions.append(pos_start)

        current = start
        while current in graph_dict and len(graph_dict[current]) == 1:
            next_node, pos_u, pos_v = graph_dict[current][0]
            traj_nodes.append(next_node)
            traj_positions.append(pos_v)
            traj_edges.append((current, next_node, pos_u, pos_v))
            current = next_node

        trajectories.append((traj_nodes, np.array(traj_positions), traj_edges))

    return trajectories


def compute_track_density(traj_positions):
    """
    计算轨迹密度
    """
    import numpy as np
    L = len(traj_positions)
    if L < 2:
        return 0.0

    p_t = traj_positions[0]
    total_N = 0
    denominator = 0
    # k从1到L-1
    for k in range(1, L):
        p_tk = traj_positions[k]
        R_tk = np.linalg.norm(p_tk - p_t)
        # 在 {p_k, p_{k+1}, ..., p_{L-1}} 中统计位于半径 R_tk 内的点数 N_{t+k}
        sub_points = traj_positions[k:]  # k到末尾的点
        count_abnormal = np.sum(np.linalg.norm(sub_points - p_t, axis=1) <= R_tk)
        total_N += count_abnormal
        denominator += (L - k)  # (L-k)为剩余点数量

    if denominator == 0:
        return 0.0
    density = total_N / denominator
    return density

def build_videos_cells_by_t(res, area_percentage=10):
    import os
    import numpy as np

    videos_cells_by_t = {}
    tid_counters = {}
    for image_path, detections_dict in res.items():
        path_parts = image_path.split('/')
        try:
            test_index = path_parts.index('test')
            video_name = path_parts[test_index + 1]
        except ValueError:
            continue

        if video_name not in videos_cells_by_t:
            videos_cells_by_t[video_name] = {}
            tid_counters[video_name] = 0

        cells_by_t = videos_cells_by_t[video_name]
        tid_counter = tid_counters[video_name]

        image_name = os.path.basename(image_path)
        frame_number_str = os.path.splitext(image_name)[0]
        frame_number = int(frame_number_str)

        boxes = detections_dict[1]
        features = detections_dict['f1']
        for i, (box, feature) in enumerate(zip(boxes, features)):
                x_min, y_min, x_max, y_max, confidence = box
                appearence = feature
                if confidence < 0.3:
                    continue

                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0
                w = x_max - x_min
                h = y_max - y_min

                flow_dir = os.path.join(*path_parts[:test_index + 2], 'flow')
                flow_path = os.path.join(flow_dir, frame_number_str + '.npy')

                if os.path.exists(flow_path):
                    flow_map = np.load(flow_path)
                    flow_channels, flow_h, flow_w = flow_map.shape

                    area_ratio = area_percentage / 100.0
                    region_w = w * np.sqrt(area_ratio)
                    region_h = h * np.sqrt(area_ratio)

                    region_x_min = max(int(x_center - region_w / 2), int(x_min))
                    region_y_min = max(int(y_center - region_h / 2), int(y_min))
                    region_x_max = min(int(x_center + region_w / 2), int(x_max))
                    region_y_max = min(int(y_center + region_h / 2), int(y_max))

                    region_x_min = np.clip(region_x_min, 0, flow_w - 1)
                    region_y_min = np.clip(region_y_min, 0, flow_h - 1)
                    region_x_max = np.clip(region_x_max, 0, flow_w - 1)
                    region_y_max = np.clip(region_y_max, 0, flow_h - 1)

                    # 提取选定区域的光流
                    flow_region = flow_map[:, region_y_min:region_y_max + 1, region_x_min:region_x_max + 1]
                    # 计算区域内的平均光流
                    flow_x = np.mean(flow_region[0])
                    flow_y = np.mean(flow_region[1])
                else:
                    # 若光流文件不存在（如最后一帧），速度设为 0
                    flow_x = 0.0
                    flow_y = 0.0

                # 分配唯一的目标ID
                tid = tid_counter
                tid_counter += 1

                # 构建位置信息和速度信息
                position = np.array([x_center, y_center, w, h])
                velocity = np.array([flow_x, flow_y])

                if frame_number not in cells_by_t:
                    cells_by_t[frame_number] = []
                cells_by_t[frame_number].append((tid, position, velocity, confidence, appearence))

        tid_counters[video_name] = tid_counter

    return videos_cells_by_t

def convert_bbox_to_mot(df):
    """ Convert bounding box to MOT challenge format (left, top, width, height) to (x1, y1, x2, y2) """
    df['x1'] = df['left']
    df['y1'] = df['top']
    df['x2'] = df['left'] + df['width']
    df['y2'] = df['top'] + df['height']
    return df[['frame', 'id', 'x1', 'y1', 'x2', 'y2']]

def evaluate(truths, predictions):
    """ Evaluate tracking performance using MOT metrics """
    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(set(truths['frame'].unique()).union(set(predictions['frame'].unique()))):
        gt = truths[truths['frame'] == frame]
        pr = predictions[predictions['frame'] == frame]
        gt_boxes = gt[['x1', 'y1', 'x2', 'y2']].values
        pr_boxes = pr[['x1', 'y1', 'x2', 'y2']].values
        distances = mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=0.5)
        acc.update(
            gt['id'].values,
            pr['id'].values,
            distances
        )
    return acc


if __name__ == '__main__':
    opt = opts().parse()

    split = 'test'
    show_flag = opt.save_track_results
    if (not os.path.exists(opt.save_results_dir)):
        os.mkdir(opt.save_results_dir)

    if opt.load_model != '':
        modelPath = opt.load_model
    else:
        modelPath = './checkpoints/DSFNet.pth'
    print(modelPath)

    results_name = opt.model_name+'_'+modelPath.split('/')[-1].split('.')[0]
    test(opt, split, modelPath, show_flag, results_name)