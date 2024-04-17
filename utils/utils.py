import random
import torch
import torch.distributed as dist
import os
import logging
import math
import numpy as np
import pickle
from tqdm import tqdm
from datetime import datetime
from argoverse.evaluation import eval_forecasting
from argoverse.map_representation.map_api import ArgoverseMap
import zlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

import torch.backends.cudnn as cudnn


origin_point = None
origin_angle = None

multi_minFDE = []
multi_minADE = []


def multi_agent_metrics(multi_outs, first_time, evaluate):
    global multi_minFDE
    global multi_minADE
    if first_time:
        multi_minFDE = []
        multi_minADE = []

    if evaluate:
        assert len(multi_minFDE) == len(multi_minADE)

        avg_minFDE = sum(multi_minFDE)/len(multi_minFDE)
        avg_minADE = sum(multi_minADE)/len(multi_minADE)
        MR = np.sum(np.array(multi_minFDE) > 2.0)/len(multi_minFDE)

        print("\nMulti Agent Evaluation")
        print(f"minADE: {avg_minADE:.5f}")
        print(f"minFDE: {avg_minFDE:.5f}")
        print(f"MR: {MR:.5f}")
    else:
        batch_size = len(multi_outs)
        for i in range(batch_size):
            # label.shape = (M_c, T_f, 2)
            # output.shape = (M_c, 6, T_f, 2)

            output = multi_outs[i][0]
            label = multi_outs[i][1]
            assert output.shape[0] == label.shape[0]

            norms = torch.norm(
                output[:, :, -1] - label[:, -1].unsqueeze(dim=1), dim=-1)
            # best_ids.shape = (M_c)
            best_ids = torch.argmin(norms, dim=-1)

            # output.shape = (M_c, T_f, 2)
            output = output[torch.arange(len(best_ids)), best_ids]

            # minFDE.shape = (M_c)
            minFDE = torch.norm(output[:, -1] - label[:, -1], dim=-1)

            # minAde.shape = (M_c)
            minADE = torch.mean(torch.norm(output - label, dim=-1), dim=-1)

            assert minADE.shape == minFDE.shape

            multi_minFDE.extend(minFDE.tolist())
            multi_minADE.extend(minADE.tolist())


def get_meta_info(meta_info):
    batch_size = len(meta_info)
    device = meta_info[0].device

    agent_lengths = [len(scene) for scene in meta_info]
    max_agent_num = max(agent_lengths)

    meta_info_tensor = torch.zeros(
        batch_size, max_agent_num, 5, device=device)

    for i, agent_length in enumerate(agent_lengths):
        meta_info_tensor[i, :agent_length] = meta_info[i]

    return meta_info_tensor


def get_masks(agent_lengths, lane_lengths, device):
    max_lane_num = max(lane_lengths)
    max_agent_num = max(agent_lengths)
    batch_size = len(agent_lengths)

    # === === Mask Generation Part === ===
    # === Agent - Agent Mask ===
    # query: agent, key-value: agent
    AA_mask = torch.zeros(
        batch_size, max_agent_num, max_agent_num, device=device)

    for i, agent_length in enumerate(agent_lengths):
        AA_mask[i, :agent_length, :agent_length] = 1
    # === === ===

    # === Agent - Lane Mask ===
    # query: agent, key-value: lane
    AL_mask = torch.zeros(
        batch_size, max_agent_num, max_lane_num, device=device)

    for i, (agent_length, lane_length) in enumerate(zip(agent_lengths, lane_lengths)):
        AL_mask[i, :agent_length, :lane_length] = 1
    # === === ===

    # === Lane - Lane Mask ===
    # query: lane, key-value: lane
    LL_mask = torch.zeros(
        batch_size, max_lane_num, max_lane_num, device=device)

    QL_mask = torch.zeros(
        batch_size, 6, max_lane_num, device=device)

    for i, lane_length in enumerate(lane_lengths):
        LL_mask[i, :lane_length, :lane_length] = 1

        QL_mask[i, :, :lane_length] = 1

    # === === ===

    # === Lane - Agent Mask ===
    # query: lane, key-value: agent
    LA_mask = torch.zeros(
        batch_size, max_lane_num, max_agent_num, device=device)

    for i, (lane_length, agent_length) in enumerate(zip(lane_lengths, agent_lengths)):
        LA_mask[i, :lane_length, :agent_length] = 1

    # === === ===

    masks = [AA_mask, AL_mask, LL_mask, LA_mask]

    # === === === === ===

    return masks, QL_mask


def eval_instance_argoverse(batch_size, pred, pred_probs, mapping, file2pred, file2labels, file2probs, DEs, iter_bar, first_time):
    def get_dis_point_2_points(point, points):
        assert points.ndim == 2
        return np.sqrt(np.square(points[:, 0] - point[0]) + np.square(points[:, 1] - point[1]))
    global method2FDEs
    if first_time:
        method2FDEs = []

    for i in range(batch_size):
        a_pred = pred[i]
        a_prob = pred_probs[i]
        # a_endpoints = all_endpoints[i]
        assert a_pred.shape == (6, 30, 2)
        assert a_prob.shape == (6, )

        file_name_int = int(os.path.split(mapping[i]['file_name'])[1][:-4])
        file2pred[file_name_int] = a_pred
        file2labels[file_name_int] = mapping[i]['origin_labels']
        file2probs[file_name_int] = a_prob

    DE = np.zeros([batch_size, 30])
    for i in range(batch_size):
        origin_labels = mapping[i]['origin_labels']
        FDE = np.min(get_dis_point_2_points(
                origin_labels[-1], pred[i, :, -1, :]))
        method2FDEs.append(FDE)
        for j in range(30):
            DE[i][j] = np.sqrt((origin_labels[j][0] - pred[i, 0, j, 0]) ** 2 + (
                    origin_labels[j][1] - pred[i, 0, j, 1]) ** 2)
    DEs.append(DE)
    miss_rate = 0.0
    miss_rate = np.sum(np.array(method2FDEs) > 2.0) / len(method2FDEs)

    iter_bar.set_description('Iter (MR=%5.3f)' % (miss_rate))


def post_eval(file2pred, file2labels, file2probs, DEs):

    metric_results = eval_forecasting.get_displacement_errors_and_miss_rate(
        file2pred, file2labels, 6, 30, 2.0, file2probs)

    for key in metric_results.keys():
        print(f"{key}_6: {metric_results[key]:.5f}")

    metric_results = eval_forecasting.get_displacement_errors_and_miss_rate(
        file2pred, file2labels, 1, 30, 2.0, file2probs)

    for key in metric_results.keys():
        print(f"{key}_1: {metric_results[key]:.5f}")

    DE = np.concatenate(DEs, axis=0)
    length = DE.shape[1]
    DE_score = [0, 0, 0, 0]
    for i in range(DE.shape[0]):
        DE_score[0] += DE[i].mean()
        for j in range(1, 4):
            index = round(float(length) * j / 3) - 1
            assert index >= 0
            DE_score[j] += DE[i][index]
    for j in range(4):
        score = DE_score[j] / DE.shape[0]
        print(f" {'ADE' if j == 0 else 'DE@1' if j == 1 else 'DE@2' if j == 2 else 'DE@3'}: {score:.5f}")


def batch_init(mapping):
    global origin_point, origin_angle
    batch_size = len(mapping)

    origin_point = np.zeros([batch_size, 2])
    origin_angle = np.zeros([batch_size])
    for i in range(batch_size):
        origin_point[i][0], origin_point[i][1] = rotate(0 - mapping[i]['cent_x'], 0 - mapping[i]['cent_y'],
                                                        mapping[i]['angle'])
        origin_angle[i] = -mapping[i]['angle']


def to_origin_coordinate(points, idx_in_batch):
    for point in points:
        point[0], point[1] = rotate(point[0] - origin_point[idx_in_batch][0],
                                    point[1] - origin_point[idx_in_batch][1], origin_angle[idx_in_batch])


def merge_tensors(tensors, device, hidden_size):
    lengths = []
    for tensor in tensors:
        lengths.append(tensor.shape[0] if tensor is not None else 0)
    res = torch.zeros([len(tensors), max(lengths), hidden_size], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0]] = tensor
    return res, lengths

def merge_attention_tensors(tensors, device):
    lengths = []
    heights = []
    for tensor in tensors:
        lengths.append(tensor.shape[0] if tensor is not None else 0)
        heights.append(tensor.shape[1] if tensor is not None else 0)
    res = torch.zeros([len(tensors), max(lengths), max(heights)], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0],:tensor.shape[1]] = tensor
    return res, lengths


def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


def save_predictions(args, predictions):
    pred_save_path = os.path.join(args.model_save_path, "predictions")
    pickle_file = open(pred_save_path, "wb")
    pickle.dump(predictions, pickle_file,
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle_file.close()


def setup(rank, world_size):
    now = datetime.now()
    s = int(now.second)
    m = int(now.minute)

    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = f"{12300 + s*2 + m*5}"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

def draw_attention_maps(attention_map, save_dir):
    """
    绘制注意力图并保存为图片

    参数:
        attention_map (list): 包含M个数据的注意力图，每个数据是形状为(N, K)的张量
        save_dir (str): 保存图片的目录路径
    """
    # 创建保存图片的目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    M = len(attention_map)  # 数据个数

    # 定义浅蓝和浅红的颜色
    light_blue = mcolors.to_rgb('lightblue')
    light_red = mcolors.to_rgb('lightcoral')

    for i in range(M):
        # 获取当前数据的形状
        N, K = attention_map[i].size()

        # 创建一个新的图像
        plt.figure()

        # 绘制浅蓝和浅红的矩形
        for row in range(N):
            for col in range(K):
                if attention_map[i][row][col] == 1:
                    plt.fill([col, col+1, col+1, col], [row, row, row+1, row+1], color=light_red)
                else:
                    plt.fill([col, col+1, col+1, col], [row, row, row+1, row+1], color=light_blue)

        # 设置坐标轴范围和标题
        plt.xlim(0, K)
        plt.ylim(0, N)
        plt.axis('off')
        plt.title(f"agents num:{N}    lanes num:{K}")

        # 保存图片
        save_path = os.path.join(save_dir, f'attention_map_{i+1}.png')
        plt.savefig(save_path)

        # 关闭图像
        plt.close()

def get_map_and_predictions(file_name_int, ex_list, predictions, file_names_in_ex_list_order=None):
    if file_names_in_ex_list_order is not None:
        idx = file_names_in_ex_list_order.index(file_name_int)
        mapping_i = pickle.loads(zlib.decompress(ex_list[idx]))
        return mapping_i, predictions[file_name_int]
    
    else:
        for mapping_i_comp in ex_list:
            mapping_i = pickle.loads(zlib.decompress(mapping_i_comp))
            if int(mapping_i["file_name"].split("/")[-1][:-4]) == file_name_int:
                return mapping_i, predictions[file_name_int]
        assert False, "No scene with given file index"

    
def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y

am = ArgoverseMap()
def get_lanes(mapping):
    pos_matrix = mapping["pos_matrix"]
    polygons = []
    agent_num = pos_matrix.shape[1]
    
    x = mapping["cent_x"]
    y = mapping["cent_y"]
    city_name = mapping['city_name']
    
    for agent_id in range(agent_num):
        for t_id in range(20):
            pos_x = pos_matrix[t_id, agent_id, 0]
            pos_y = pos_matrix[t_id, agent_id, 1]

            bias_x, bias_y = rotate(pos_x, pos_y, -1*mapping["angle"])
            temp_x, temp_y = (bias_x + x), (bias_y + y)
            
            polygons_i = am.find_local_lane_polygons([temp_x - 100, temp_x + 100, temp_y - 100, temp_y + 100], city_name)
            
            for polygon_i in polygons_i:
                # check if polygon is available
                available = False
                for available_polygone in polygons:
                    if np.all(available_polygone == polygon_i[:, :2]):
                        available = True
                        break
                        
                if available:
                    continue

                polygons.append(polygon_i[:, :2])

        break
           

    polygons = [polygon.copy() for polygon in polygons]
    angle = mapping['angle']
    for index_polygon, polygon in enumerate(polygons):
        for i, point in enumerate(polygon):
            point[0], point[1] = rotate(point[0] - x, point[1] - y, angle)

    polygons = [polygon for polygon in polygons]
    
    return polygons

def plot_scene(mapping_i,predictions_i):
    mapping_i, predictions_i
    mapping_i = mapping_i.copy()
    predictions_pred = predictions_i[0]
    predictions_label = predictions_i[1]
    
    lanes = get_lanes(mapping_i)
    # === Plot Lanes ===
    for lane in lanes:
        plt.plot(lane[:, 0], lane[:, 1], c="gray", alpha=0.25)
        
    pos_matrix = mapping_i["pos_matrix"]
    
    # === Plot Ego Past ===
    plt.plot(pos_matrix[:20, 0, 0], pos_matrix[:20, 0, 1], c="cyan")
    
    # === Plot Other Agents Past ===
    agent_num = pos_matrix.shape[1] - 1
    for j in range(agent_num):
        positions = pos_matrix[:20, j + 1]
        plt.plot(positions[:, 0], positions[:, 1], c="black")
        plt.scatter(positions[-1, 0], positions[-1, 1], marker=".", c="black")
        
    
    # === Plot Ego Futures ===
    angle = mapping_i["angle"]
    x = mapping_i["cent_x"]
    y = mapping_i["cent_y"]
    for predictions_i in predictions_pred:
        for pred_i_j in predictions_i:
            
            # To local coordinate system 
            # for i, point in enumerate(pred_i_j):
            #     point[0], point[1] = rotate(point[0], point[1], angle)
                
            plt.plot(pred_i_j[:, 0], pred_i_j[:, 1], c="green")
            plt.scatter(pred_i_j[-1, 0], pred_i_j[-1, 1], marker="*", c="green", edgecolors="black", s=100, zorder=5)
        
    # === Plot Ego Future (GT) ===
    mgt = predictions_label
    for gt in mgt:
        # for i, point in enumerate(gt):
            # point[0], point[1] = rotate(point[0] , point[1], angle)
        
        plt.plot(gt[:, 0], gt[:, 1], c="red")
        plt.scatter(gt[-1, 0], gt[-1, 1], marker="*", c="red", edgecolors="black", s=100, zorder=6)
    
    origin_x, origin_y = mgt[0][-1]
    plt.axis('equal')
    
    plt.xlim([origin_x - 30, origin_x + 30])
    plt.ylim([origin_y - 30, origin_y + 30])
    plt.axis('off')
    
    fig_name=mapping_i["file_name"].split("/")[-1][:-4]
    plt.savefig(f"exp_pic/{fig_name}_output.png")
    plt.clf()


def plot_attention_map(mapping_i):
    mapping_i = mapping_i.copy()
    
    lanes = get_lanes(mapping_i)
    # === Plot Lanes ===
    for lane in lanes:
        plt.plot(lane[:, 0], lane[:, 1], c="gray", alpha=0.25)
        
    pos_matrix = mapping_i["pos_matrix"]
    
    # === Plot Ego Past ===
    plt.plot(pos_matrix[:20, 0, 0], pos_matrix[:20, 0, 1], c="red")

    # === Plot Other Agents Past ===
    agent_num = pos_matrix.shape[1] - 1
    for j in range(agent_num):
        positions = pos_matrix[:20, j + 1]
        plt.plot(positions[:, 0], positions[:, 1], c="green")

    lane_data=mapping_i["lane_data"]
    attention_map=mapping_i["attention_map"]

    # === Plot Ego Futures ===
    angle = mapping_i["angle"]
    cx = mapping_i["cent_x"]
    cy = mapping_i["cent_y"]

    # 绘制每个向量
    for i,vector in enumerate(lane_data):
        if attention_map[0,i].item()==0:
            continue
        x = [vector[0, -2].tolist()]+vector[:, -4].tolist()
        y = [vector[0, -1].tolist()]+vector[:, -3].tolist()
        # for i,value in enumerate(zip(x,y)):
        #     x[i],y[i]=rotate(value[0], value[1], -1*angle)
        plt.plot(y, x, color='limegreen', alpha=0.25, linewidth=5)
    
    plt.axis('equal')
    
    plt.xlim([- 400, 400])
    plt.ylim([- 400, 400])
    plt.axis('off')
    
    fig_name=mapping_i["file_name"].split("/")[-1][:-4]
    plt.savefig(f"exp_pic/Attention_{fig_name}_output.png")
    plt.clf()

def plot_attention_map_agent(mapping_i):
    mapping_i = mapping_i.copy()
    
    lanes = get_lanes(mapping_i)
        
    pos_matrix = mapping_i["pos_matrix"]
    agent_data = mapping_i["agent_data"]
    
    for a,agent in enumerate(agent_data):
        # === Plot Lanes ===
        for lane in lanes:
            plt.plot(lane[:, 0], lane[:, 1], c="gray", alpha=0.25)

        # === Plot Ego Past ===
        ax=[agent[0, 2].tolist()]+agent[:,0].tolist()
        ay=[agent[0, 3].tolist()]+agent[:,1].tolist()
        plt.plot(ax, ay, c="red")

        lane_data=mapping_i["lane_data"]
        attention_map=mapping_i["attention_map"]

        # === Plot Ego Futures ===
        angle = mapping_i["angle"]
        cx = mapping_i["cent_x"]
        cy = mapping_i["cent_y"]

        # 绘制每个向量
        for i,vector in enumerate(lane_data):
            if attention_map[a,i].item()==0:
                continue
            x = [vector[0, -2].tolist()]+vector[:, -4].tolist()
            y = [vector[0, -1].tolist()]+vector[:, -3].tolist()
            # for i,value in enumerate(zip(x,y)):
            #     x[i],y[i]=rotate(value[0], value[1], -1*angle)
            plt.plot(y, x, color='limegreen', alpha=0.25, linewidth=5)
        
        plt.axis('equal')
        
        plt.xlim([- 150, 150])
        plt.ylim([- 150, 150])
        plt.axis('off')
        
        fig_name=mapping_i["file_name"].split("/")[-1][:-4]
        if not os.path.exists(f"exp_pic/{fig_name}"):
            os.makedirs(f"exp_pic/{fig_name}")
        plt.savefig(f"exp_pic/{fig_name}/Attention_{a}_output.png")
        plt.clf()

def plot_attention(mapping,layer_index,attention_probs):
    if mapping==None:
        return

    for i,mapping_i in enumerate(mapping): 
        # lanes = get_lanes(mapping_i)
        attention_prob=attention_probs[i]
            
        pos_matrix = mapping_i["pos_matrix"]
        agent_data = mapping_i["agent_data"]
        
        agent=agent_data[0]

        lane_data=mapping_i["lane_data"]

        # === Plot Ego Futures ===
        angle = mapping_i["angle"]
        cx = mapping_i["cent_x"]
        cy = mapping_i["cent_y"]

        attention_prob=attention_probs[i]
        fig_name=mapping_i["file_name"].split("/")[-1][:-4]
        if not os.path.exists(f"exp_pic/attention_{fig_name}/{layer_index}"):
            os.makedirs(f"exp_pic/attention_{fig_name}/{layer_index}")

        for n,attention in  enumerate(attention_prob): 
            # === Plot Lanes ===
            # for lane in lanes:
            #     plt.plot(lane[:, 0], lane[:, 1], c="gray", alpha=0.05)

            # === Plot Ego Past ===
            # ax=[agent[0, 2].tolist()]+agent[:,0].tolist()
            # ay=[agent[0, 3].tolist()]+agent[:,1].tolist()
            # plt.plot(ax, ay, c="red")

            # 绘制每个向量
            for i,vector in enumerate(lane_data):
                x = [vector[0, -2].tolist()]+vector[:, -4].tolist()
                y = [vector[0, -1].tolist()]+vector[:, -3].tolist()
                # for i,value in enumerate(zip(x,y)):
                #     x[i],y[i]=rotate(value[0], value[1], -1*angle)
                plt.plot(y, x, color='#006400', alpha=min(1.0,attention[0,i].item()*3))
                
            plt.axis('equal')
                
            plt.xlim([- 75, 75])
            plt.ylim([- 75, 75])
            plt.axis('off')
        
        # === Plot Ego Past ===
        ax=[agent[0, 2].tolist()]+agent[:,0].tolist()
        ay=[agent[0, 3].tolist()]+agent[:,1].tolist()
        plt.plot(ax, ay, c="red")
                
        plt.savefig(f"exp_pic/attention_{fig_name}/{layer_index}/Attention_output.png")
        plt.clf()