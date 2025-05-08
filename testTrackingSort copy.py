from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import torch

from lib.utils.opts import opts

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

CONFIDENCE_thres = 0.3
COLORS = [(255, 0, 0)]
global_track_id_counter = 0

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
    h_map, w_map = feature_map.shape[2:]  # 特征图的高宽
    f1 = []  # 存储类别号为 1 的外观特征

    if 1 in ret:  # 检查类别号 1 是否存在
        boxes = ret[1]
        for box in boxes:
            x_min, y_min, x_max, y_max, conf = box

            # 检测框合法性检查
            if x_max <= x_min or y_max <= y_min:
                f1.append(torch.zeros(feature_map.size(1)))  # 填充零特征
                continue

            # 将检测框坐标映射到特征图坐标
            x_min = int(max(0, x_min * w_map / feature_map.size(-1)))
            x_max = int(min(w_map, x_max * w_map / feature_map.size(-1)))
            y_min = int(max(0, y_min * h_map / feature_map.size(-2)))
            y_max = int(min(h_map, y_max * h_map / feature_map.size(-2)))

            # 检查裁剪区域有效性
            if x_max <= x_min or y_max <= y_min:
                f1.append(torch.zeros(feature_map.size(1)))  # 填充零特征
                continue

            # 裁剪特征图
            cropped_feature = feature_map[:, :, y_min:y_max, x_min:x_max]

            # 特殊处理空裁剪区域
            if cropped_feature.numel() == 0:
                feature_vector = torch.zeros(feature_map.size(1))  # 填充零特征
            else:
                pooled_feature = F.adaptive_avg_pool2d(cropped_feature, (1, 1))
                feature_vector = pooled_feature.view(-1)

            # 添加特征向量
            f1.append(feature_vector)

    # 将类别号 1 的外观特征加入 ret
    ret['f1'] = f1
    return ret

def save_videos_cells_by_t(videos_cells_by_t, filepath):
    """保存 videos_cells_by_t 到本地文件"""
    with open(filepath, 'wb') as f:
        pickle.dump(videos_cells_by_t, f)

def load_videos_cells_by_t(filepath):
    """从本地文件加载 videos_cells_by_t"""
    with open(filepath, 'rb') as f:
        videos_cells_by_t = pickle.load(f)
    return videos_cells_by_t


def test(opt, split, modelPath, show_flag, results_name):

    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    # # Logger(opt)
    # print(opt.model_name)

    # dataset = COCO(opt, split)

    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # model = get_det_net({'hm': dataset.num_classes, 'wh': 2, 'reg': 2}, opt.model_name)  # 建立模型
    # model = load_model(model, modelPath)
    # device = torch.device("cuda:1")  # 指定使用的 GPU
    # model = model.cuda()
    # model.eval()

    # results = {}
    # res = {}
    # return_time = False
    # scale = 1
    # num_classes = dataset.num_classes
    # max_per_image = opt.K

    # file_folder_pre = ''
    # im_count = 0

    # saveTxt = opt.save_track_results
    # if saveTxt:
    #     #track_results_save_dir = os.path.join(opt.save_results_dir, 'trackingResults'+opt.model_name)
    #     track_results_save_dir = os.path.join(opt.save_results_dir, 'trackingResults'+'sort')
    #     if not os.path.exists(track_results_save_dir):
    #         os.mkdir(track_results_save_dir)

    # num_iters = len(data_loader)
    # bar = Bar('processing', max=num_iters)
    # videoName= '/test_'+str(CONFIDENCE_thres)+'.mp4'

    # fps=10
    # size=(1024,1024)
    # fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    # videoWriter=cv2.VideoWriter(videoName,fourcc,fps,size)
    # cnt = 0

    # for ind, (file_path, img_id, pre_processed_images) in enumerate(data_loader):
    #     # print(ind)
    #     if(ind>len(data_loader)-1):
    #         break

    #     bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
    #         ind, num_iters,total=bar.elapsed_td, eta=bar.eta_td
    #     )

    #     #set tracker
    #     file_folder_cur = pre_processed_images['file_name'][0].split('/')[-3]
    #     if file_folder_cur != file_folder_pre:
    #         if saveTxt and file_folder_pre!='':
    #              fid.close()
    #         file_folder_pre = file_folder_cur
    #         mot_tracker = Sort()
    #         if saveTxt:
    #             im_count = 0
    #             txt_path = os.path.join(track_results_save_dir, file_folder_cur+'.txt')
    #             fid = open(txt_path, 'w+')

    #     #read images
    #     detection = []
    #     meta = pre_process(pre_processed_images['input'], scale)
    #     image = pre_processed_images['input'].cuda()
    #     img = pre_processed_images['imgOri'].squeeze().numpy()

    #     #detection
    #     output, feature, dets = process(model, image, return_time)
    #     #POST PROCESS
    #     dets = post_process(dets, meta, num_classes)
    #     detection.append(dets)
    #     ret = merge_outputs(detection, num_classes, max_per_image)

    #     #update tracker
    #     dets_track = dets[1]
    #     dets_track_select = np.argwhere(dets_track[:,-1]>CONFIDENCE_thres)
    #     dets_track = dets_track[dets_track_select[:,0],:]
    #     track_bbs_ids = mot_tracker.update(dets_track)
        
    #     if(show_flag):
    #         frame, det = cv2_demo(img, track_bbs_ids)
            
    #         cv2.imwrite(f'results/frame_{ind}.jpg',img)
    #         # 写入视频
    #         videoWriter.write(frame)

    #         #cv2.waitKey(5)
    #         #hm1 = output['hm'].squeeze(0).squeeze(0).cpu().detach().numpy()
    #         #img2 = cv2.resize(hm1,(1280,720))
    #         #video2.write(img2)
    #         #cv2.imshow('hm', hm1)
    #         #cv2.waitKey(5)

    #     if saveTxt:
    #         im_count += 1
    #         track_bbs_ids = track_bbs_ids[::-1,:]
    #         track_bbs_ids[:,2:4] = track_bbs_ids[:,2:4]-track_bbs_ids[:,:2]
    #         for it in range(track_bbs_ids.shape[0]):
    #             #print(track_bbs_ids[it,-1])
    #             #print('\n')
    #             fid.write('%d,%d,%0.2f,%0.2f,%0.2f,%0.2f,1,-1,-1,-1\n'%(im_count,
    #                       track_bbs_ids[it,-1], track_bbs_ids[it,0],track_bbs_ids[it,1],
    #                             track_bbs_ids[it, 2], track_bbs_ids[it, 3]))
    #             #im_count,track_bbs_ids[it,-1], track_bbs_ids[it,0],track_bbs_ids[it,1],track_bbs_ids[it, 2], track_bbs_ids[it, 3]
    #     results[img_id.numpy().astype(np.int32)[0]] = ret
    #     ret = extract_features_for_class_1(ret, feature)
    #     res[file_path[0]] = ret
    #     bar.next()
    #     cnt += 1
    #     # if cnt%400==0:
    #     #     break
    # bar.finish()


    '''
    begin
    '''
    # 定义配置字典
    # optim_config = {
    #     # 边权重相关配置
    #     'confidence_function': 'linear',         # 置信度函数：'linear'、'quadratic'、'constant'
    #     'weight_position': 1.0,                  # 位置差异的权重 a
    #     'weight_velocity': 1.0,                  # 速度差异的权重 b
    #     'weight_appearance': 1.0,                # 外观差异的权重 c
    #     'epsilon': 1e-6,
    #     'gaussian_sigma': 1.0,                   # 高斯函数的标准差

    #     # 超参数配置
    #     'max_children': 3,                      # 每个节点最多的关联数量
    #     'distance_threshold': 10,               # 关联的距离阈值
    #     'confidence_threshold': 0.3,            # 置信度阈值

    #     # 约束条件配置
    #     'add_trajectory_start_end_costs': False,# 是否添加轨迹起始和终止成本
    #     'start_cost': 0.1,                      # 轨迹起始成本
    #     'end_cost': 0.1,                        # 轨迹终止成本
    #     'min_track_length': None,               # 轨迹的最小长度
    #     'max_track_length': None,               # 轨迹的最大长度

    #     # 遮挡和目标消失处理配置
    #     'allow_missing_detections': False,      # 是否允许轨迹中断（缺失检测）
    #     'max_missing_frames': 0,                # 允许的最大缺失帧数

    #     # 长时间跨度关联配置
    #     'time_window': 1,                       # 关联的时间窗口大小，默认为1表示仅相邻帧
    # }
    global global_track_id_counter
    import itertools
    static_config = {
        'confidence_function': 'quadratic',
        'weight_position': 10,
        'weight_velocity': 30,#[0, 1, 10, 30, 50, 100],
        'weight_appearance': 50,#, 10, 30, 50, 100],
        'gaussian_sigma': 2,
    }

    # 参数取值范围
    # best：
    # 'confidence_function': 'quadratic', 'weight_position': 10, 'weight_velocity': 30, 'weight_appearance': 50, 'epsilon': 1, 'sigma_position': 2, 'sigma_velocity': 2
    param_ranges = {
        'start_cost': [0.1],#, 0.5, 1, 5, 10],
        'end_cost': [0.1],#, 0.5, 1, 5, 10],
        'min_track_length': [1],
        'max_track_length': [None],#, 50, 80, 100, 150, 200, None],
        'max_missing_frames': [0],#, 1, 2, 3, 5, 8, 10],
        'density_threshold': [0.45],
        'min_distance_threshold': [2],
        'fragment_length_max' : [15],   # Nτ
        'max_gap'             : [10],   # 插值最大帧间隔
        'Td'                  : [25.],  # 位置归一化阈
        'Tv'                  : [6.],   # 速度归一化阈
        'wp'                  : [0.7],
        'wv'                  : [0.3],
        'delta_FS'            : [0.4]
    }

    optim_configs = [
        {
            **static_config,  # 添加固定配置
            'start_cost': comb[0],
            'end_cost': comb[1],
            'min_track_length': comb[2],
            'max_track_length': comb[3],
            'max_missing_frames': comb[4],
            'density_threshold': comb[5],
            'min_distance_threshold': comb[6],
            
        }
        for comb in itertools.product(
            param_ranges['start_cost'],
            param_ranges['end_cost'],
            param_ranges['min_track_length'],
            param_ranges['max_track_length'],
            param_ranges['max_missing_frames'],
            param_ranges['density_threshold'],
            param_ranges['min_distance_threshold'],
            
        )
    ]

    best_mota = 0.0
    best_idf1 = 0.0

    cache_filepath = 'videos_cells_by_t.pkl'

    videos_cells_by_t = load_videos_cells_by_t(cache_filepath)
    # save_videos_cells_by_t(videos_cells_by_t, cache_filepath)

    for idx, optim_config in enumerate(optim_configs):
        save_dir = 'tracking_results'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 评估累计
        all_summaries=[]
        for video_name, cells_by_t in videos_cells_by_t.items():
            # 1) 构建图
            log_print("map")
            G = create_graph(cells_by_t, optim_config)
            
            # 2) 一次ILP
            solution_edges_initial = solve_ilp(G, optim_config)
            if solution_edges_initial is None:
                print("No ILP solution for", video_name)
                continue
            
            # 3) parse初步轨迹
            pred_tracks_initial, frame_tracks = parse_prediction(solution_edges_initial)
            print(f"Initial ILP found {len(pred_tracks_initial)} tracks for {video_name}.")

            # 4) 后处理: 在数据结构层面插入虚拟节点，衔接碎片
            pred_tracks_final = insert_virtual_nodes(pred_tracks_initial, cells_by_t, optim_config)

            # 5) 去掉虚拟节点 & 过滤最短轨迹
            pred_tracks_post = post_process_tracks(pred_tracks_final, optim_config)

            # 6) 重建 frame_tracks，用于评估或保存
            frame_tracks = rebuild_frame_tracks_from_pred_tracks(pred_tracks_post)

                 
            # 保存跟踪结果到 txt 文件
            txt_path = os.path.join(save_dir, f'{video_name}.txt')
            with open(txt_path, 'w') as f:
                for frame_id in sorted(frame_tracks.keys()):
                    tracks_in_frame = frame_tracks[frame_id]
                    for track in tracks_in_frame:
                        # track 至少6 => (tid, x,y,w,h,is_v)
                        tid, x, y, w, h, isv = track[:6]
                        
                        # 若还有第7个 => 可能是 conf or vx?
                        # 若您是 conf => conf = track[6]
                        # else: conf = None
                        conf = None
                        if len(track) >= 7:
                            # 根据需求判断 track[6] 是否 float => conf
                            # or if isinstance(track[6], bool) => skip
                            maybe_7th = track[6]
                            if isinstance(maybe_7th, (float,int)):
                                conf = maybe_7th
                            # else it might be vx => ignore or parse differently
                        
                        # isv_str
                        isv_str = "1" if isv else "0"
                        
                        if conf is not None:
                            f.write(f"{frame_id},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{isv_str},{conf:.2f}\n")
                        else:
                            f.write(f"{frame_id},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{isv_str}\n")
        log_print("fin")

        # 跟踪评估
        truth_dir = '/home/yinz/workspace/evaluation/gt/*.txt'
        result_dir = '/home/yinz/workspace/DSFNet/tracking_results/*.txt'
        all_summaries = []

        # Store the summaries in a DataFrame
        summary_list = []

        # Process each ground truth and prediction file pair
        for truth_path, result_path in zip(sorted(glob.glob(truth_dir)), sorted(glob.glob(result_dir))):
            truths = pd.read_csv(
                truth_path,
                header=None,
                names=['frame', 'id', 'left', 'top', 'width', 'height', 'conf', 'x', 'y', 'z','a','b','c'],
                index_col=False
            )
            predictions = pd.read_csv(
                result_path,
                header=None,
                names=['frame', 'id', 'left', 'top', 'width', 'height', 'is_virt'],
                index_col=False
            )

            predictions = remove_tail_virtual_nodes(predictions)

            # Convert bbox coordinates
            truths = convert_bbox_to_mot(truths)
            predictions = convert_bbox_to_mot(predictions)

            # Evaluate tracking
            acc = evaluate(truths, predictions)
            mh = mm.metrics.create()
            summary = mh.compute(acc, metrics=[
            'mota', 'idf1', 'mostly_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches'
            ], name='metrics')
            all_summaries.append(summary)

            # Extract file name for simplified output
            result_name = result_path.split('/')[-1].split('.')[0]
            summary_row = [result_name] + summary.iloc[0].tolist()
            summary_list.append(summary_row)

        # Create a DataFrame for all summaries
        columns = ['name', 'mota', 'idf1', 'mostly_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches']
        summary_df = pd.DataFrame(summary_list, columns=columns)

        # Print the summary table
        log_print(f'\n{summary_df.to_string(index=False)}')

        # Aggregate all summaries and calculate average
        overall_summary = pd.concat(all_summaries).mean(axis=0)
        print_optim_configs_method3(optim_config)
        log_print(f'\n{overall_summary}')

        if overall_summary['mota']>best_mota:
            best_mota = overall_summary['mota']
            log_print(update_best_mota = best_mota)
        if overall_summary['idf1']>best_idf1:
            best_idf1 = overall_summary['idf1']
            log_print(update_best_idf1 = best_idf1)

    # dataset.run_eval(results, opt.save_results_dir, results_name)

import pandas as pd

def print_optim_configs_method3(optim_config):
    items = list(optim_config.items())
    lines = []
    for j in range(0, len(items), 3):
        chunk = items[j:j+3]
        formatted_chunk = ", ".join(f"{k}: {v}" for k, v in chunk)
        lines.append(f"  {formatted_chunk}")
    config_str = "\n".join(lines)
    log_print(f'\n{config_str}')

def remove_tail_virtual_nodes(predictions):
    """
    对同一 track_id 的帧，从最后一帧往前查找:
      - 若是虚拟节点 is_virt=1 => 删除
      - 若是真实节点 is_virt=0 => 停止往前
    这样仅去掉轨迹末尾的“无效”虚拟节点, 保留中间/头部虚拟节点.

    参数:
      predictions: pd.DataFrame，含列 ['frame','id','is_virt', 以及 left/top/width/height等]
                   假设 'frame' 和 'id' int, 'is_virt' => 0/1(或bool).
    返回:
      filtered_df: 去除尾部虚拟节点后的 DataFrame
    """
    # 1) 先按照 track_id, frame排序
    #    我们要逆序 => frame descending
    df = predictions.copy()
    # 若 is_virt 可能是str => 先转int
    df['is_virt'] = df['is_virt'].astype(int)
    
    df.sort_values(by=['id','frame'], ascending=[True,False], inplace=True)
    
    # 2) 分组
    group_ids = df['id'].unique()
    
    # 结果收集
    keep_index = []
    
    for tid in group_ids:
        sub = df[df['id']==tid]  # 该 track
        # sub已按 frame 降序排列
        sub_indices = sub.index.tolist()  # 行号
        sub_frames  = sub['frame'].tolist()
        sub_virt    = sub['is_virt'].tolist()
        
        # 逆序遍历
        remove_mode = True
        # "remove_mode"表示我们还在移除阶段
        # 一旦遇到 is_virt=0 => break => remove_mode=False => 以后都不删
        for i, row_idx in enumerate(sub_indices):
            if not remove_mode:
                # 不再删除
                keep_index.append(row_idx)
                continue
            
            # 若还在 remove_mode => check is_virt
            if sub_virt[i] == 1:
                # => 这是虚拟节点,删除 => 不加到keep
                # pass
                pass
            else:
                # is_virt=0 => 真实节点 => stop remove
                remove_mode = False
                keep_index.append(row_idx)
        
        # 之后sub的剩余行我们不处理，因为都逆序遍历完了
        # 但我们写在for i... => 全部遍历
    
    # keep_index里是保留行
    filtered_df = df.loc[keep_index].copy()
    
    # 3) 排序回正序(如有需要)
    filtered_df.sort_values(by=['id','frame'], ascending=[True,True], inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)
    
    return filtered_df


def post_process_tracks(pred_tracks, optim_config):
    """
    1) 过滤真实节点数 < min_track_length 的轨迹
    2) 保留虚拟节点，以供可视化或评估
    假设 pred_tracks[tid][frame] 可能是:
      (x, y, w, h, is_virtual) 或 (x, y, w, h, is_virtual, vx, vy)
    """
    min_track_length = optim_config.get('min_track_length', 2)
    final_tracks = {}
    
    for tid, frames_dict in pred_tracks.items():
        real_count = 0
        for fr, box in frames_dict.items():
            # 根据长度判断:
            if len(box) >= 5:
                # box[:5] => (x, y, w, h, isv)
                # isv = box[4]
                # 如果是 bool => True/False
                isv = box[4]
                # 若它是 bool => 说明 box[4] 为 is_virtual
                # 真实节点 => if not isv
                if isinstance(isv, bool):
                    if not isv:
                        real_count += 1
                else:
                    # box[4] 不是 bool => 说明这条数据没有 is_virtual
                    # => 视为真实节点
                    real_count += 1
            else:
                # 若 len(box)<5 => 可能是 (x,y,w,h) => 视为真实节点
                real_count += 1
        
        if real_count >= min_track_length:
            final_tracks[tid] = frames_dict
    
    return final_tracks

import os
import glob
import numpy as np
import pandas as pd
import itertools
import networkx as nx
import pulp
import motmetrics as mm
from scipy.spatial import KDTree
from sklearn.metrics.pairwise import cosine_similarity

class KalmanFilter:
    def __init__(self):
        # 状态向量 [x, y, vx, vy]
        self.state = np.zeros(4, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32)
        # 状态转移矩阵 (匀速模型)
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        # 观测矩阵
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        # 过程/观测噪声协方差
        self.Q = np.eye(4, dtype=np.float32)*0.01
        self.R = np.eye(2, dtype=np.float32)*0.1

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.H @ self.state  # 返回 [x, y]

    def update(self, measurement):
        # measurement: [x, y]
        y = measurement - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state += K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

def forward_filter(measurements, kf):
    """
    measurements: list of (x,y)
    返回:
      fwd_states: [N,4]
      fwd_covs  : [N,4,4]
    """
    fwd_states = []
    fwd_covs   = []
    x = kf.state.copy()
    P = kf.P.copy()
    
    for z in measurements:
        # predict
        x = kf.F @ x
        P = kf.F @ P @ kf.F.T + kf.Q
        # update
        y = z - (kf.H @ x)
        S = kf.H @ P @ kf.H.T + kf.R
        K = P @ kf.H.T @ np.linalg.inv(S)
        x = x + K @ y
        I = np.eye(4, dtype=np.float32)
        P = (I - K @ kf.H) @ P
        
        fwd_states.append(x.copy())
        fwd_covs.append(P.copy())
    return fwd_states, fwd_covs

import numpy as np

class CVKalman:
    """
    四维常速模型  x=[x, y, vx, vy]ᵀ
    只用于预测位置/速度，不关心框宽高，适合卫星小目标。
    """
    def __init__(self, init_pos, init_vel=(0., 0.), dt=1.0,
                 sigma_p=50.0, sigma_v=50.0, sigma_a=1.0, sigma_r=3.0):
        """
        init_pos : (x, y)
        init_vel : (vx, vy)
        """
        self.dt = dt
        # 状态向量
        self.x = np.array([init_pos[0], init_pos[1],
                           init_vel[0], init_vel[1]], dtype=float)

        # 协方差
        self.P = np.diag([sigma_p, sigma_p, sigma_v, sigma_v])

        # 过程噪声系数
        self.sigma_a = sigma_a

        # 观测噪声 (只量测位置)
        self.R = np.diag([sigma_r ** 2, sigma_r ** 2])

    # ---------- 内部矩阵 ----------
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

    # ---------- 对外接口 ----------
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

    # 便捷取值
    @property
    def pos(self):
        return self.x[:2]

    @property
    def vel(self):
        return self.x[2:]


def replay_track_with_kf(frames_map, dt=1.0):
    """
    让 KF 以论文要求对已有轨迹做一次 predict→update 回放，
    输出包含速度 vx,vy 的轨迹以及回放完毕后的 KF。
    frames_map : {frame: (x,y,w,h,is_virtual)}
    """
    frames_sorted = sorted(frames_map.keys())
    if not frames_sorted:
        return None, {}

    # 第 1 帧初始化 KF（无速度假设）
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
        # 把 KF 的位置投回左上角 + 宽高
        cx_est, cy_est = cvkf.pos
        x_est, y_est = cx_est - w / 2, cy_est - h / 2
        frames_map_out[f] = (x_est, y_est, w, h, isv, vx, vy)

    return cvkf, frames_map_out


def insert_virtual_nodes(pred_tracks_initial,
                         cells_by_t=None,
                         optim_config=None):
    """
    论文 5.4.2 实现：
      1) 先用 KF 回放获得末端位置 & 速度
      2) 对短段集合 F 做一次 '一跳预测' 匹配
      3) 满足评分阈值后插入虚拟节点并级联
    """
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

    # ---------- 1. KF 回放 ----------
    pred_tracks, kf_dict = {}, {}
    for tid, frames in pred_tracks_initial.items():
        kf, frames_out = replay_track_with_kf(frames)
        pred_tracks[tid] = frames_out
        kf_dict[tid] = kf

    # ---------- 2. 生成候选碎片集合 F ----------
    frag_ids = []
    for tid, fr_dict in pred_tracks.items():
        frames_sorted = sorted(fr_dict.keys())
        # 长度 < Nt_max 且未到末帧（若有 cells_by_t，可用最后帧索引判断）
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
                kf_tmp = CVKalman(pA_end, vA_end)   # 拷贝
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

        # 循环直到再也拼不动

    # ---------- 4. 输出 ----------
    return {tid: frames for tid, frames in pred_tracks.items() if tid not in used}



def rebuild_frame_tracks_from_pred_tracks(pred_tracks):
    """
    pred_tracks: {tid: {frame: (x,y,w,h,is_v[, vx,vy,...])}}
    => 返回 frame_tracks: {frame: [(tid, x,y,w,h,is_v[, vx,vy,...])]}

    如果后续写文件只需 tid, x,y,w,h,isv,可将 vx,vy 等保留在 track tuple 里看需求。
    """
    frame_tracks = {}
    for tid, frames_dict in pred_tracks.items():
        for fr, box in frames_dict.items():
            # box 至少5个维度: (x,y,w,h,is_v), 可能≥7 (vx,vy)
            if len(box) < 5:
                # 如果不足5 => 视为 (x,y,w,h)? => 补 is_v=False
                x, y, w, h = box[:4]
                is_v = False
                # vx, vy = 0,0  (可选)
                new_track_tuple = (tid, x, y, w, h, is_v)
            else:
                # 取前5个
                x, y, w, h, is_v = box[:5]
                # 如果还存在 vx, vy => box[5], box[6]
                vx, vy = 0.0, 0.0
                if len(box) >= 7:
                    vx, vy = box[5], box[6]
                # 可以选择保留
                # new_track_tuple = (tid, x, y, w, h, is_v, vx, vy)
                # 如果写文件只要6个 => (tid,x,y,w,h,is_v)
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
    epsilon = 1
    gaussian_sigma = optim_config.get('gaussian_sigma', 1.0)

    # 只考虑相邻帧
    time_window = 1

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

        # 构建轨迹(含edges)以进行密度检查
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
    """
    返回一个列表，每个元素为 (traj_nodes, traj_positions, traj_edges)
    traj_nodes: 轨迹节点序列 (u,v,...)
    traj_positions: 与traj_nodes对应的position序列的np.array(L, 2)
    traj_edges: 构成该轨迹的边列表[(u,v,pos_u,pos_v), ...]
    """
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
    traj_positions: numpy数组，形状为 (L, 2)
    """
    import numpy as np
    L = len(traj_positions)
    if L < 2:
        # 如果轨迹太短，没有后续点，就不视为异常
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

# def build_videos_cells_by_t(res, area_percentage=10):
#     import os
#     import numpy as np

#     videos_cells_by_t = {}  # 存储每个视频的 cells_by_t
#     tid_counters = {}       # 存储每个视频的全局目标ID计数器
#     for image_path, detections_dict in res.items():
#         # 提取视频名称和帧编号
#         # 假设图像路径格式为 './data/viso/test/001/img1/000001.jpg'
#         path_parts = image_path.split('/')
#         try:
#             test_index = path_parts.index('test')
#             video_name = path_parts[test_index + 1]  # 获取视频名称，例如 '001'
#         except ValueError:
#             print(f"无法在路径中找到 'test'：{image_path}")
#             continue

#         # 初始化视频的 cells_by_t 和 tid_counter
#         if video_name not in videos_cells_by_t:
#             videos_cells_by_t[video_name] = {}
#             tid_counters[video_name] = 0  # 初始化目标ID计数器

#         cells_by_t = videos_cells_by_t[video_name]
#         tid_counter = tid_counters[video_name]

#         # 提取帧编号
#         image_name = os.path.basename(image_path)          # 获取文件名，例如 '000001.jpg'
#         frame_number_str = os.path.splitext(image_name)[0] # 提取帧编号字符串，例如 '000001'
#         frame_number = int(frame_number_str)               # 转换为整数

#         # 处理每个检测框
#         boxes = detections_dict[1]  # 获取检测框
#         features = detections_dict['f1']  # 获取外观特征
#         for i, (box, feature) in enumerate(zip(boxes, features)):
#             # detections_array 的形状为 [N, 5]，其中 N 为检测框数量
            
#                 x_min, y_min, x_max, y_max, confidence = box
#                 appearence = feature
#                 # **添加置信度过滤**
#                 if confidence < 0.3:
#                     continue  # 跳过低置信度的检测

#                 # 计算中心点坐标、宽度和高度
#                 x_center = (x_min + x_max) / 2.0
#                 y_center = (y_min + y_max) / 2.0
#                 w = x_max - x_min
#                 h = y_max - y_min

#                 # 构建光流文件路径
#                 # 光流路径格式为 './data/viso/test/001/flow/000001.npy'
#                 flow_dir = os.path.join(*path_parts[:test_index + 2], 'flow')  # 构建 flow 文件夹路径
#                 flow_path = os.path.join(flow_dir, frame_number_str + '.npy')  # 构建光流文件路径

#                 # 检查光流文件是否存在
#                 if os.path.exists(flow_path):
#                     flow_map = np.load(flow_path)  # 加载光流图，形状为 [2, H, W]
#                     flow_channels, flow_h, flow_w = flow_map.shape

#                     # 计算选取区域的大小
#                     area_ratio = area_percentage / 100.0  # 将百分比转换为小数
#                     # 计算选取区域的宽度和高度
#                     region_w = w * np.sqrt(area_ratio)
#                     region_h = h * np.sqrt(area_ratio)

#                     # 确定区域的边界，不超过目标框
#                     region_x_min = max(int(x_center - region_w / 2), int(x_min))
#                     region_y_min = max(int(y_center - region_h / 2), int(y_min))
#                     region_x_max = min(int(x_center + region_w / 2), int(x_max))
#                     region_y_max = min(int(y_center + region_h / 2), int(y_max))

#                     # 确保边界在图像范围内
#                     region_x_min = np.clip(region_x_min, 0, flow_w - 1)
#                     region_y_min = np.clip(region_y_min, 0, flow_h - 1)
#                     region_x_max = np.clip(region_x_max, 0, flow_w - 1)
#                     region_y_max = np.clip(region_y_max, 0, flow_h - 1)

#                     # 提取选定区域的光流
#                     flow_region = flow_map[:, region_y_min:region_y_max + 1, region_x_min:region_x_max + 1]
#                     # 计算区域内的平均光流
#                     flow_x = np.mean(flow_region[0])
#                     flow_y = np.mean(flow_region[1])
#                 else:
#                     # 若光流文件不存在（如最后一帧），速度设为 0
#                     flow_x = 0.0
#                     flow_y = 0.0

#                 # 分配唯一的目标ID
#                 tid = tid_counter
#                 tid_counter += 1

#                 # 构建位置信息和速度信息
#                 position = np.array([x_center, y_center, w, h])
#                 velocity = np.array([flow_x, flow_y])

#                 # 将数据添加到 cells_by_t，包括置信度
#                 if frame_number not in cells_by_t:
#                     cells_by_t[frame_number] = []
#                 cells_by_t[frame_number].append((tid, position, velocity, confidence, appearence))

#         # 更新该视频的目标ID计数器
#         tid_counters[video_name] = tid_counter

#     return videos_cells_by_t


import os
import numpy as np
from collections import defaultdict

import os
from collections import defaultdict

def parse_all_ground_truths(base_path="data/viso/test"):
    """
    读取 base_path 下每个视频 <vid>/gt/gt.txt
    返回:
      video_gt_dict = {
        '001': {
           1: [(obj_id, x_min, y_min, w, h, conf, vx, vy, ...), ... ],
           2: [...],
           ...
        },
        '016': {...},
        ...
      }
      first_appear_dict = {
        ('001', obj_id): frame_id,
        ...
      }
    """
    video_gt_dict = {}
    first_appear_dict = {}
    
    # 列出 test下所有子目录(每个视频)
    videos = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for vid in videos:
        gt_path = os.path.join(base_path, vid, "gt", "gt.txt")
        if not os.path.exists(gt_path):
            continue
        
        video_gt_dict[vid] = defaultdict(list)
        
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9:
                    # 不够字段 => 跳过或报错
                    continue
                frame_id = int(parts[0])    # 可能1-based
                obj_id   = int(parts[1])
                x_min    = float(parts[2])
                y_min    = float(parts[3])
                w        = float(parts[4])
                h        = float(parts[5])
                conf     = float(parts[6])
                vx       = float(parts[7])
                vy       = float(parts[8])
                # 若有更多字段，可继续
                tup = (obj_id, x_min, y_min, w, h, conf, vx, vy)
                video_gt_dict[vid][frame_id].append(tup)
                
                key = (vid, obj_id)
                if key not in first_appear_dict:
                    first_appear_dict[key] = frame_id  # 记录首次出现帧
    
    return video_gt_dict, first_appear_dict

def overlap_exceed_10pct(det_box, gt_box):
    """
    det_box: (x_min, y_min, w, h)
    gt_box : (x_min, y_min, w, h)
    若相交面积 >= 0.1 * det_area => True
    """
    (dxmin, dymin, dw, dh) = det_box
    (gxmin, gymin, gw, gh) = gt_box
    
    if dw<=0 or dh<=0:
        return False
    det_area = dw*dh
    
    # intersection
    d_xmax = dxmin + dw
    d_ymax = dymin + dh
    g_xmax = gxmin + gw
    g_ymax = gymin + gh
    
    inter_xmin = max(dxmin, gxmin)
    inter_ymin = max(dymin, gymin)
    inter_xmax = min(d_xmax, g_xmax)
    inter_ymax = min(d_ymax, g_ymax)
    
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w*inter_h
    
    if inter_area >= 0.1*det_area:
        return True
    return False


import numpy as np
import os

def build_videos_cells_by_t(
    res,
    video_gt_dict,
    first_appear_dict,
    area_percentage=10
):
    """
    res: {image_path -> {1: array_of_detection, 'f1': features}}
    video_gt_dict: 同 parse_all_ground_truths 返回
    first_appear_dict: { (video_name, obj_id): first_frame_id }
    area_percentage: 用于光流区域 => 10表示10%
    
    返回:
      videos_cells_by_t = {
        video_name: {
          frame_number: [
             (tid, position(4d), velocity(2d), conf, appearance),
             ...
          ]
        }
      }
    """
    videos_cells_by_t = {}
    tid_counters = {}
    
    for image_path, detections_dict in res.items():
        # 1) 提取 video_name & frame_number
        path_parts = image_path.split('/')
        try:
            test_index = path_parts.index('test')
            video_name = path_parts[test_index + 1]
        except ValueError:
            print(f"无法在路径中找到 'test': {image_path}")
            continue
        
        if video_name not in videos_cells_by_t:
            videos_cells_by_t[video_name] = {}
            tid_counters[video_name] = 0
        cells_by_t = videos_cells_by_t[video_name]
        tid_counter = tid_counters[video_name]
        
        image_name = os.path.basename(image_path)
        frame_number_str = os.path.splitext(image_name)[0]
        frame_number = int(frame_number_str)  # 这里假设 => 1-based
        
        # 2) 构建 detection_list
        detection_list = []
        
        boxes = detections_dict[1]       # shape [N,5] => (x_min,y_min,x_max,y_max,conf)
        features = detections_dict['f1'] # shape [N, ...]
        
        for i, (box, feat) in enumerate(zip(boxes, features)):
            x_min, y_min, x_max, y_max, confidence = box
            if confidence<0.3:
                continue
            x_center = 0.5*(x_min+x_max)
            y_center = 0.5*(y_min+y_max)
            w = x_max - x_min
            h = y_max - y_min
            
            # => 先 placeholder flow_x,flow_y=0
            # 后面计算光流(见下)
            flow_x, flow_y = 0.0, 0.0
            
            # we will fill flow_x,flow_y after area flow
            # build detection => (tid, position, velocity, confidence, feat)
            # tid:
            tid = tid_counter
            tid_counter += 1
            
            position = np.array([x_center,y_center,w,h], dtype=np.float32)
            velocity = np.array([flow_x,flow_y], dtype=np.float32)
            detection_list.append((tid, position, velocity, confidence, feat))
        
        # 3) 计算光流 => for each detection => 取 [region_x_min,region_x_max,...] 并 np.mean
        #    这里和您原先代码一样
        flow_dir = os.path.join(*path_parts[:test_index+2], 'flow')  # data/viso/test/XX/flow
        flow_path = os.path.join(flow_dir, frame_number_str + '.npy')
        if os.path.exists(flow_path):
            flow_map = np.load(flow_path)  # shape [2, H, W]
            flow_channels, flow_h, flow_w = flow_map.shape
            
            # 逐个更新 detection 的 flow
            new_detection_list = []
            for (tid, pos, vel, conf, feat) in detection_list:
                x_c, y_c, ww, hh = pos
                # area
                region_w = ww * np.sqrt(area_percentage/100.0)
                region_h = hh * np.sqrt(area_percentage/100.0)
                
                # region_x_min ...
                region_x_min = max(int(x_c - region_w/2), int(x_c - ww/2))
                region_y_min = max(int(y_c - region_h/2), int(y_c - hh/2))
                region_x_max = min(int(x_c + region_w/2), int(x_c + ww/2))
                region_y_max = min(int(y_c + region_h/2), int(y_c + hh/2))
                
                # clip to flow range
                region_x_min = np.clip(region_x_min, 0, flow_w-1)
                region_y_min = np.clip(region_y_min, 0, flow_h-1)
                region_x_max = np.clip(region_x_max, 0, flow_w-1)
                region_y_max = np.clip(region_y_max, 0, flow_h-1)
                
                if region_x_max>=region_x_min and region_y_max>=region_y_min:
                    flow_region = flow_map[:, region_y_min:region_y_max+1, region_x_min:region_x_max+1]
                    fx = np.mean(flow_region[0])
                    fy = np.mean(flow_region[1])
                else:
                    fx, fy = 0.0,0.0
                
                velocity = np.array([fx,fy], dtype=np.float32)
                new_detection_list.append( (tid, pos, velocity, conf, feat) )
            
            detection_list = new_detection_list
        else:
            # 不存在 => 保持velocity=0
            pass
        
        # 4) 若 frame_number==1 => 用 GT 替换 detection_list
        #    (请注意frame_number可能是1-based,如果您想0-based,请改 if frame_number==0)
        if frame_number==1:
            detection_list.clear()
            # 读取 gt => video_gt_dict[video_name][1]
            # 可能为空
            if frame_number in video_gt_dict.get(video_name, {}):
                gtlist = video_gt_dict[video_name][frame_number]
                for (obj_id, gxmin, gymin, gw, gh, gconf, gvx, gvy) in gtlist:
                    tid = tid_counter
                    tid_counter+=1
                    x_center = gxmin + 0.5*gw
                    y_center = gymin + 0.5*gh
                    position = np.array([x_center,y_center,gw,gh], dtype=np.float32)
                    velocity = np.array([gvx,gvy], dtype=np.float32)
                    
                    # appearance先写None
                    detection_list.append( (tid, position, velocity, gconf, None) )
        
        # 5) 处理 "首次出现" => 用 GT => 删除相交>10%
        #    先看 gt 里有没有 frame_number
        if frame_number in video_gt_dict.get(video_name, {}):
            gtlist = video_gt_dict[video_name][frame_number]
            filtered_list = detection_list[:]  # 先copy
            for (obj_id, gxmin, gymin, gw, gh, gconf, gvx, gvy) in gtlist:
                # 看 first_appear_dict
                if (video_name, obj_id) in first_appear_dict:
                    first_appear_frame = first_appear_dict[(video_name, obj_id)]
                    if first_appear_frame == frame_number:
                        # => 先删除相交>10%
                        new_filtered = []
                        gt_box = (gxmin, gymin, gw, gh)
                        for (dtid, dpos, dvel, dconf, dfeat) in filtered_list:
                            dxmin = dpos[0] - dpos[2]/2
                            dymin = dpos[1] - dpos[3]/2
                            dw    = dpos[2]
                            dh    = dpos[3]
                            det_box = (dxmin, dymin, dw, dh)
                            if overlap_exceed_10pct(det_box, gt_box):
                                # skip
                                continue
                            else:
                                new_filtered.append( (dtid, dpos, dvel, dconf, dfeat) )
                        
                        filtered_list = new_filtered
                        
                        # 再添加这个 GT
                        tid = tid_counter
                        tid_counter+=1
                        x_center = gxmin + 0.5*gw
                        y_center = gymin + 0.5*gh
                        position = np.array([x_center,y_center,gw,gh], dtype=np.float32)
                        velocity = np.array([gvx,gvy], dtype=np.float32)
                        filtered_list.append( (tid, position, velocity, gconf, None) )
            
            detection_list = filtered_list
        
        # 6) 将 detection_list 放入 cells_by_t
        if frame_number not in cells_by_t:
            cells_by_t[frame_number] = []
        cells_by_t[frame_number].extend(detection_list)
        
        # 更新 tid
        tid_counters[video_name] = tid_counter
    
    return videos_cells_by_t



import motmetrics as mm
import glob
import pandas as pd

def read_data(file_path):
    """ Read data from given file path """
    return pd.read_csv(
        file_path,
        header=None,
        names=['frame', 'id', 'left', 'top', 'width', 'height', 'conf', 'x', 'y', 'z'],
        index_col=False
    )

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