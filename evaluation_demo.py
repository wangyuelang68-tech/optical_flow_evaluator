#!/usr/bin/env python
"""
光流评估演示脚本
"""

import numpy as np
import cv2
import os
import sys

# 添加当前目录到路径
sys.path.append('.')

from optical_flow_evaluator import OpticalFlowEvaluator, read_flo_file, batch_evaluate_flows
from flow_utils import load_flow, save_flow, warp_image, visualize_flow_comparison

def demo_single_evaluation():
    """单个光流评估演示"""
    print("=" * 80)
    print("单个光流评估演示")
    print("=" * 80)
    
    # 示例数据路径（请修改为你的实际路径）
    flow_path = "out.flo"
    img1_path = "./images/one.png"
    img2_path = "./images/two.png"
    gt_path = None  # 如果没有真值，设置为None
    
    # 检查文件是否存在
    if not os.path.exists(flow_path):
        print(f"错误: 光流文件不存在 - {flow_path}")
        print("请先生成光流文件或修改路径")
        return
    
    # 加载数据
    print("正在加载数据...")
    flow = read_flo_file(flow_path)
    
    img1 = cv2.imread(img1_path) if os.path.exists(img1_path) else None
    img2 = cv2.imread(img2_path) if os.path.exists(img2_path) else None
    gt = read_flo_file(gt_path) if gt_path and os.path.exists(gt_path) else None
    
    # 创建评估器
    evaluator = OpticalFlowEvaluator(
        flow=flow,
        img1=img1,
        img2=img2,
        ground_truth=gt,
        flow_name="SPyNet"  # 你的算法名称
    )
    
    # 执行评估
    print("\n正在评估...")
    results = evaluator.evaluate_all(verbose=True)
    
    # 保存报告
    evaluator.save_report("flow_evaluation_report.txt")
    
    # 可视化结果
    print("\n正在生成可视化...")
    evaluator.visualize_results()
    
    # 显示扭曲图像比较
    if img1 is not None and img2 is not None:
        print("\n显示扭曲图像比较...")
        warped = warp_image(img2, flow)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.title('Original Image 1')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.title('Image 2 Warped to Image 1')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.title('Original Image 2')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return results

def demo_batch_evaluation():
    """批量评估演示"""
    print("=" * 80)
    print("批量光流评估演示")
    print("=" * 80)
    
    # 示例：评估多个光流算法
    # 请根据你的实际情况修改这些路径
    
    # 假设你有多个光流算法的结果
    flow_files = [
        "results/spynet/flow1.flo",
        "results/pwcnet/flow1.flo",
        "results/flownet2/flow1.flo"
    ]
    
    img1_files = [
        "data/frame1.png",
        "data/frame1.png",
        "data/frame1.png"
    ]
    
    img2_files = [
        "data/frame2.png",
        "data/frame2.png",
        "data/frame2.png"
    ]
    
    gt_files = [
        "data/ground_truth.flo",
        "data/ground_truth.flo",
        "data/ground_truth.flo"
    ]
    
    flow_names = [
        "SPyNet",
        "PWC-Net",
        "FlowNet2"
    ]
    
    # 检查文件是否存在
    existing_files = []
    existing_img1 = []
    existing_img2 = []
    existing_gt = []
    existing_names = []
    
    for i in range(len(flow_files)):
        if os.path.exists(flow_files[i]):
            existing_files.append(flow_files[i])
            existing_img1.append(img1_files[i] if os.path.exists(img1_files[i]) else None)
            existing_img2.append(img2_files[i] if os.path.exists(img2_files[i]) else None)
            existing_gt.append(gt_files[i] if os.path.exists(gt_files[i]) else None)
            existing_names.append(flow_names[i])
    
    if not existing_files:
        print("没有找到光流文件，跳过批量评估")
        return None
    
    print(f"找到 {len(existing_files)} 个光流文件进行批量评估")
    
    # 执行批量评估
    results_df = batch_evaluate_flows(
        flow_files=existing_files,
        img1_files=existing_img1,
        img2_files=existing_img2,
        gt_files=existing_gt,
        flow_names=existing_names,
        output_dir="./batch_evaluation_results"
    )
    
    # 显示结果摘要
    print("\n批量评估结果摘要:")
    print(results_df[['flow_name', 'overall_score_score', 'overall_score_grade']].to_string(index=False))
    
    return results_df

def demo_create_synthetic_data():
    """创建合成数据演示"""
    print("=" * 80)
    print("创建合成测试数据")
    print("=" * 80)
    
    # 创建测试图像
    h, w = 256, 256
    
    # 图像1：渐变图像
    img1 = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        img1[i, :, 0] = i  # 红色渐变
        img1[i, :, 1] = 128  # 绿色固定
        img1[i, :, 2] = 255 - i  # 蓝色渐变
    
    # 图像2：向右平移的图像
    shift_x, shift_y = 10, 5  # 平移量
    img2 = np.roll(img1, shift_x, axis=1)
    img2 = np.roll(img2, shift_y, axis=0)
    
    # 创建简单的光流（所有像素向右下角移动）
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[:, :, 0] = shift_x  # 水平移动
    flow[:, :, 1] = shift_y  # 垂直移动
    
    # 保存数据
    os.makedirs("demo_data", exist_ok=True)
    
    cv2.imwrite("demo_data/img1.png", img1)
    cv2.imwrite("demo_data/img2.png", img2)
    
    # 保存光流
    with open("demo_data/flow.flo", 'wb') as f:
        np.array([202021.25], np.float32).tofile(f)
        np.array([w], np.int32).tofile(f)
        np.array([h], np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)
    
    print(f"合成数据已创建:")
    print(f"  图像1: demo_data/img1.png")
    print(f"  图像2: demo_data/img2.png")
    print(f"  光流: demo_data/flow.flo")
    print(f"  平移量: ({shift_x}, {shift_y}) 像素")
    
    # 评估合成数据
    print("\n评估合成数据...")
    evaluator = OpticalFlowEvaluator(
        flow=flow,
        img1=img1,
        img2=img2,
        ground_truth=flow,  # 使用自身作为真值（完美情况）
        flow_name="SyntheticData"
    )
    
    results = evaluator.evaluate_all(verbose=True)
    
    return img1, img2, flow

def main():
    """主函数"""
    print("光流评估工具箱")
    print("版本: 1.0")
    print("=" * 80)
    
    # 创建演示目录
    os.makedirs("evaluation_demo", exist_ok=True)
    os.chdir("evaluation_demo")
    
    while True:
        print("\n请选择操作:")
        print("1. 单个光流评估")
        print("2. 批量光流评估")
        print("3. 创建合成测试数据")
        print("4. 退出")
        
        choice = input("请输入选择 (1-4): ").strip()
        
        if choice == '1':
            demo_single_evaluation()
        elif choice == '2':
            demo_batch_evaluation()
        elif choice == '3':
            demo_create_synthetic_data()
        elif choice == '4':
            print("退出程序")
            break
        else:
            print("无效选择，请重新输入")
        
        input("\n按Enter键继续...")

if __name__ == "__main__":
    # 检查依赖
    try:
        import matplotlib
        import cv2
        import numpy
        import pandas
        print("所有依赖已安装")
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请运行: pip install opencv-python matplotlib numpy pandas scikit-image scipy tqdm")
        sys.exit(1)
    
    main()