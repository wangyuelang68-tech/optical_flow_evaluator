#!/usr/bin/env python
"""
光流处理工具函数
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def load_flow(flow_path):
    """
    加载光流文件（支持 .flo, .npy, .png 格式）
    """
    if flow_path.endswith('.flo'):
        return read_flo_file(flow_path)
    elif flow_path.endswith('.npy'):
        return np.load(flow_path)
    elif flow_path.endswith('.png'):
        # Middlebury 彩色编码格式
        return flow_from_color(flow_path)
    else:
        raise ValueError(f"不支持的格式: {flow_path}")

def save_flow(flow, output_path):
    """
    保存光流文件
    """
    if output_path.endswith('.flo'):
        save_flo_file(flow, output_path)
    elif output_path.endswith('.npy'):
        np.save(output_path, flow)
    elif output_path.endswith('.png'):
        # 保存为彩色图像
        flow_color = flow_to_color(flow)
        cv2.imwrite(output_path, cv2.cvtColor(flow_color, cv2.COLOR_RGB2BGR))
    else:
        raise ValueError(f"不支持的格式: {output_path}")

def read_flo_file(filename):
    """读取 .flo 文件"""
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        
        if magic == 202021.25:  # 老格式
            w = int(np.fromfile(f, np.int32, count=1)[0])
            h = int(np.fromfile(f, np.int32, count=1)[0])
        else:  # 新格式
            f.seek(0)
            magic_bytes = f.read(4)
            if magic_bytes != b'PIEH':
                raise ValueError("Invalid .flo file format")
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
        
        flow = np.fromfile(f, np.float32, count=h * w * 2).reshape(h, w, 2)
    
    return flow

def save_flo_file(flow, filename):
    """保存为 .flo 文件"""
    with open(filename, 'wb') as f:
        # 写入魔法数
        np.array([202021.25], np.float32).tofile(f)
        
        # 写入尺寸
        h, w = flow.shape[:2]
        np.array([w], np.int32).tofile(f)
        np.array([h], np.int32).tofile(f)
        
        # 写入数据
        flow.astype(np.float32).tofile(f)

def flow_to_color(flow, max_flow=None):
    """
    将光流转换为彩色图像
    
    参数:
    ----------
    flow : numpy.ndarray
        光流场，形状 (H, W, 2)
    max_flow : float, optional
        最大光流值，用于归一化
        
    返回:
    ----------
    numpy.ndarray : RGB彩色图像,形状 (H, W, 3)
    """
    h, w = flow.shape[:2]
    
    # 计算幅度和角度
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)
    
    # 归一化幅度
    if max_flow is None:
        max_flow = np.percentile(mag, 95)
    
    if max_flow > 0:
        mag_norm = np.clip(mag / max_flow, 0, 1)
    else:
        mag_norm = np.zeros_like(mag)
    
    # 创建HSV图像
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi)  # 色调 [0, 1]
    hsv[..., 1] = 1.0  # 饱和度
    hsv[..., 2] = mag_norm  # 亮度
    
    # 转换为RGB
    rgb = hsv_to_rgb(hsv)
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb

def flow_from_color(color_img):
    """
    从彩色编码图像恢复光流(Middlebury格式)
    """
    # Middlebury格式：色调表示方向，亮度表示幅度
    hsv = cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # 归一化
    hsv[..., 0] /= 179.0  # OpenCV中H的范围是[0, 179]
    hsv[..., 1] /= 255.0
    hsv[..., 2] /= 255.0
    
    # 恢复角度和幅度
    ang = hsv[..., 0] * 2 * np.pi - np.pi  # [-π, π]
    mag = hsv[..., 2] * 64.0  # Middlebury中最大为64
    
    # 恢复u, v分量
    u = mag * np.cos(ang)
    v = mag * np.sin(ang)
    
    flow = np.stack([u, v], axis=-1)
    return flow

def warp_image(img, flow):
    """
    使用光流扭曲图像
    
    参数:
    ----------
    img : numpy.ndarray
        输入图像
    flow : numpy.ndarray
        光流场，形状 (H, W, 2)
        
    返回:
    ----------
    numpy.ndarray : 扭曲后的图像
    """
    h, w = flow.shape[:2]
    
    # 创建坐标网格
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # 计算扭曲后的坐标
    map_x = (x_coords + flow[:, :, 0]).astype(np.float32)
    map_y = (y_coords + flow[:, :, 1]).astype(np.float32)
    
    # 扭曲图像
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return warped

def visualize_flow_comparison(flow_list, flow_names=None, img1=None, img2=None, 
                              ground_truth=None, figsize=(16, 10)):
    """
    可视化多个光流算法的比较
    
    参数:
    ----------
    flow_list : list
        光流场列表
    flow_names : list, optional
        算法名称列表
    img1 : numpy.ndarray, optional
        第一帧图像
    img2 : numpy.ndarray, optional
        第二帧图像
    ground_truth : numpy.ndarray, optional
        真实光流
    figsize : tuple
        图像大小
    """
    n_flows = len(flow_list)
    
    if flow_names is None:
        flow_names = [f'Flow {i+1}' for i in range(n_flows)]
    
    # 计算需要显示的子图数量
    n_cols = min(3, n_flows + (1 if ground_truth is not None else 0) + (1 if img1 is not None else 0))
    n_rows = (n_flows + (1 if ground_truth is not None else 0) + (1 if img1 is not None else 0) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    idx = 0
    
    # 显示原始图像
    if img1 is not None and idx < len(axes):
        axes[idx].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else img1, cmap='gray')
        axes[idx].set_title('First Frame')
        axes[idx].axis('off')
        idx += 1
    
    if img2 is not None and idx < len(axes):
        axes[idx].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else img2, cmap='gray')
        axes[idx].set_title('Second Frame')
        axes[idx].axis('off')
        idx += 1
    
    # 显示真值（如果有）
    if ground_truth is not None and idx < len(axes):
        gt_color = flow_to_color(ground_truth)
        axes[idx].imshow(gt_color)
        axes[idx].set_title('Ground Truth')
        axes[idx].axis('off')
        idx += 1
    
    # 显示各个算法的结果
    for i, (flow, name) in enumerate(zip(flow_list, flow_names)):
        if idx >= len(axes):
            break
            
        flow_color = flow_to_color(flow)
        axes[idx].imshow(flow_color)
        axes[idx].set_title(name)
        axes[idx].axis('off')
        idx += 1
    
    # 隐藏多余的子图
    for i in range(idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def create_flow_animation(img1, img2, flow, output_path='flow_animation.gif', duration=100):
    """
    创建光流动画
    
    参数:
    ----------
    img1 : numpy.ndarray
        第一帧
    img2 : numpy.ndarray
        第二帧
    flow : numpy.ndarray
        光流场
    output_path : str
        输出路径
    duration : int
        每帧持续时间(ms)
    """
    from PIL import Image
    import imageio
    
    # 创建中间帧
    frames = []
    
    # 添加第一帧
    frames.append(Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)))
    
    # 创建中间扭曲帧
    for alpha in np.linspace(0, 1, 10):
        # 计算中间光流
        interp_flow = flow * alpha
        
        # 扭曲图像
        warped = warp_image(img2, -interp_flow)  # 反向扭曲
        
        frames.append(Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)))
    
    # 添加第二帧
    frames.append(Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)))
    
    # 保存为GIF
    frames[0].save(output_path, save_all=True, append_images=frames[1:], 
                   duration=duration, loop=0)
    
    print(f"动画已保存到: {output_path}")
    return output_path