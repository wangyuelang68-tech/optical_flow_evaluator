#!/usr/bin/env python
"""
光流评估工具 - 完整调用示例
基于完整的光流评估工具箱
"""

import sys
import os
import numpy as np

# 添加当前目录到路径，确保可以导入自定义模块
sys.path.append('.')

# ==================== 第一步：确保所有依赖已安装 ====================
def check_dependencies():
    """检查所有必需的依赖"""
    required_packages = [
        ('numpy', 'np'),
        ('cv2', 'cv2'),
        ('matplotlib', 'plt'),
        ('skimage', 'skimage'),
        ('pandas', 'pd'),
        ('tqdm', 'tqdm')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(package_name)
            print(f"✅ {package_name} 已安装")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name} 未安装")
    
    if missing_packages:
        print(f"\n缺少以下依赖包,请安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

# ==================== 第二步：导入评估器 ====================
try:
    from optical_flow_evaluator import OpticalFlowEvaluator, read_flo_file, batch_evaluate_flows
    from flow_utils import load_flow, save_flow, warp_image
    print("✅ 成功导入光流评估工具")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保以下文件在当前目录：")
    print("1. optical_flow_evaluator.py")
    print("2. flow_utils.py")
    sys.exit(1)

# ==================== 第三步：主评估函数 ====================
def evaluate_single_flow():
    """评估单个光流文件"""
    print("\n" + "="*80)
    print("单个光流评估")
    print("="*80)
    
    # 配置文件路径（根据你的实际情况修改）
    config = {
        'flow_file': 'C:/Users/86157/Desktop/pytorch-spynet-master/out2.flo',      # 光流文件
        'img1_file': 'C:/Users/86157/Desktop/pytorch-spynet-master/images/three.png',    # 第一帧图像
        'img2_file': 'C:/Users/86157/Desktop/pytorch-spynet-master/images/four.png',     # 第二帧图像
        'gt_file': None,             # 真值文件（如果没有设为None）
        'algorithm_name': 'SPyNet',  # 算法名称
        'output_dir': 'C:/Users/86157/Desktop/pytorch-spynet-master/evaluation_results'  # 输出目录
    }
    
    # 检查文件是否存在
    print("📁 检查文件...")
    for key, value in config.items():
        if key.endswith('_file') and value is not None:
            if os.path.exists(value):
                print(f"  ✅ {key}: {value}")
            else:
                print(f"  ❌ {key}不存在: {value}")
                print(f"  请确保文件存在或修改配置文件")
                return None
    
    try:
        # 1. 加载光流数据
        print("\n📊 加载数据...")
        flow = read_flo_file(config['flow_file'])
        print(f"  光流尺寸: {flow.shape}")
        
        # 2. 加载图像（如果存在）
        img1 = img2 = gt = None
        
        if config['img1_file'] and os.path.exists(config['img1_file']):
            import cv2
            img1 = cv2.imread(config['img1_file'])
            print(f"  图像1尺寸: {img1.shape}")
        
        if config['img2_file'] and os.path.exists(config['img2_file']):
            import cv2
            img2 = cv2.imread(config['img2_file'])
            print(f"  图像2尺寸: {img2.shape}")
        
        if config['gt_file'] and os.path.exists(config['gt_file']):
            gt = read_flo_file(config['gt_file'])
            print(f"  真值尺寸: {gt.shape}")
        
        # 3. 创建评估器
        print(f"\n🔍 创建评估器: {config['algorithm_name']}")
        evaluator = OpticalFlowEvaluator(
            flow=flow,
            img1=img1,
            img2=img2,
            ground_truth=gt,
            flow_name=config['algorithm_name']
        )
        
        # 4. 执行评估
        print("正在评估光流质量...")
        results = evaluator.evaluate_all(verbose=True)
        
        # 5. 保存报告
        os.makedirs(config['output_dir'], exist_ok=True)
        report_file = os.path.join(config['output_dir'], f"{config['algorithm_name']}_report.txt")
        evaluator.save_report(report_file)
        
        # 6. 可视化结果
        evaluator.visualize_results(config['output_dir'])
        
        print(f"\n✅ 评估完成!")
        print(f"   报告已保存: {report_file}")
        print(f"   可视化结果在: {config['output_dir']}/visualizations/")
        
        return results
        
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_multiple_algorithms():
    """比较多个光流算法"""
    print("\n" + "="*80)
    print("多算法比较")
    print("="*80)
    
    # 配置多个算法的结果
    algorithms = [
        {
            'name': 'SPyNet',
            'flow_file': 'results/spynet/out.flo',
            'img1_file': 'images/three.png',
            'img2_file': 'images/four.png',
            'gt_file': None  # 如果有真值可以添加
        },
        {
            'name': 'PWC-Net',
            'flow_file': 'results/pwcnet/out.flo',
            'img1_file': 'images/three.png',
            'img2_file': 'images/four.png',
            'gt_file': None
        },
        # 可以添加更多算法...
    ]
    
    # 过滤掉不存在的文件
    valid_algorithms = []
    for algo in algorithms:
        if os.path.exists(algo['flow_file']):
            valid_algorithms.append(algo)
            print(f"✅ {algo['name']}: {algo['flow_file']}")
        else:
            print(f"❌ {algo['name']} 光流文件不存在: {algo['flow_file']}")
    
    if not valid_algorithms:
        print("没有有效的算法可比较")
        return None
    
    try:
        # 执行批量评估
        flow_files = [algo['flow_file'] for algo in valid_algorithms]
        img1_files = [algo['img1_file'] for algo in valid_algorithms]
        img2_files = [algo['img2_file'] for algo in valid_algorithms]
        gt_files = [algo['gt_file'] for algo in valid_algorithms]
        flow_names = [algo['name'] for algo in valid_algorithms]
        
        output_dir = './comparison_results'
        
        print(f"\n正在比较 {len(valid_algorithms)} 个算法...")
        
        results_df = batch_evaluate_flows(
            flow_files=flow_files,
            img1_files=img1_files,
            img2_files=img2_files,
            gt_files=gt_files,
            flow_names=flow_names,
            output_dir=output_dir
        )
        
        # 打印比较结果
        print("\n" + "="*80)
        print("算法比较结果")
        print("="*80)
        
        import pandas as pd
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        
        # 显示关键指标
        key_columns = ['flow_name', 'overall_score_score', 'overall_score_grade']
        
        if 'ground_truth_mean_epe' in results_df.columns:
            key_columns.extend(['ground_truth_mean_epe', 'ground_truth_accuracy_1px'])
        
        if 'photometric_psnr' in results_df.columns:
            key_columns.extend(['photometric_psnr', 'photometric_mean_error'])
        
        if 'smoothness_avg_gradient' in results_df.columns:
            key_columns.extend(['smoothness_avg_gradient'])
        
        # 只显示存在的列
        available_columns = [col for col in key_columns if col in results_df.columns]
        
        if available_columns:
            print(results_df[available_columns].to_string(index=False))
        else:
            print(results_df.to_string(index=False))
        
        # 保存为Excel以便进一步分析
        excel_path = os.path.join(output_dir, 'comparison_summary.xlsx')
        results_df.to_excel(excel_path, index=False)
        print(f"\n详细比较结果已保存: {excel_path}")
        
        return results_df
        
    except Exception as e:
        print(f"❌ 比较过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def quick_evaluation():
    """快速评估（无需图像）"""
    print("\n" + "="*80)
    print("快速评估（仅光流统计）")
    print("="*80)
    
    flow_file = input("请输入光流文件路径 (默认: out.flo): ").strip()
    if not flow_file:
        flow_file = 'out.flo'
    
    if not os.path.exists(flow_file):
        print(f"❌ 文件不存在: {flow_file}")
        return None
    
    try:
        # 读取光流
        flow = read_flo_file(flow_file)
        h, w = flow.shape[:2]
        
        print(f"\n📊 光流基本信息:")
        print(f"  尺寸: {w} × {h}")
        print(f"  总像素: {w * h:,}")
        
        # 基本统计
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        magnitude = np.sqrt(u**2 + v**2)
        
        print(f"\n📈 统计信息:")
        print(f"  水平分量 (u):")
        print(f"    范围: [{u.min():.3f}, {u.max():.3f}]")
        print(f"    均值: {u.mean():.3f} ± {u.std():.3f}")
        
        print(f"  垂直分量 (v):")
        print(f"    范围: [{v.min():.3f}, {v.max():.3f}]")
        print(f"    均值: {v.mean():.3f} ± {v.std():.3f}")
        
        print(f"  光流幅度:")
        print(f"    范围: [{magnitude.min():.3f}, {magnitude.max():.3f}]")
        print(f"    均值: {magnitude.mean():.3f} ± {magnitude.std():.3f}")
        
        # 质量初步判断
        print(f"\n🔍 质量初步判断:")
        
        if magnitude.max() > 100:
            print(f"  ⚠️  警告: 最大幅度过大 ({magnitude.max():.1f}像素)")
        elif magnitude.max() < 0.5:
            print(f"  ⚠️  警告: 幅度过小，可能无效")
        else:
            print(f"  ✅ 幅度范围正常")
        
        zero_flow_ratio = np.sum(magnitude < 0.1) / magnitude.size
        if zero_flow_ratio > 0.8:
            print(f"  ⚠️  警告: {zero_flow_ratio*100:.1f}% 的区域几乎没有运动")
        else:
            print(f"  ✅ 运动区域比例正常")
        
        return flow
        
    except Exception as e:
        print(f"❌ 快速评估失败: {e}")
        return None

# ==================== 第四步：主菜单 ====================
def main():
    """主函数"""
    print("="*80)
    print("光流评估工具箱 v1.0")
    print("="*80)
    
    # 检查依赖
    if not check_dependencies():
        print("\n请先安装缺少的依赖包,然后重新运行")
        return
    
    while True:
        print("\n" + "="*80)
        print("主菜单")
        print("="*80)
        print("1. 单个光流完整评估")
        print("2. 多算法比较")
        print("3. 快速评估（仅统计）")
        print("4. 创建测试数据")
        print("5. 退出")
        
        choice = input("\n请选择操作 (1-5): ").strip()
        
        if choice == '1':
            results = evaluate_single_flow()
            if results:
                # 询问是否查看详细信息
                view_details = input("\n是否查看详细结果? (y/n): ").strip().lower()
                if view_details == 'y':
                    import json
                    print(json.dumps(results, indent=2, ensure_ascii=False))
        
        elif choice == '2':
            results_df = compare_multiple_algorithms()
        
        elif choice == '3':
            flow = quick_evaluation()
            if flow is not None:
                # 询问是否可视化
                visualize = input("\n是否可视化光流? (y/n): ").strip().lower()
                if visualize == 'y':
                    try:
                        import matplotlib.pyplot as plt
                        
                        magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
                        
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        
                        # 水平分量
                        im1 = axes[0].imshow(flow[:,:,0], cmap='RdBu')
                        axes[0].set_title('Horizontal Flow (u)')
                        plt.colorbar(im1, ax=axes[0])
                        
                        # 垂直分量
                        im2 = axes[1].imshow(flow[:,:,1], cmap='RdBu')
                        axes[1].set_title('Vertical Flow (v)')
                        plt.colorbar(im2, ax=axes[1])
                        
                        # 幅度
                        im3 = axes[2].imshow(magnitude, cmap='hot')
                        axes[2].set_title('Flow Magnitude')
                        plt.colorbar(im3, ax=axes[2])
                        
                        plt.suptitle('Optical Flow Visualization')
                        plt.tight_layout()
                        plt.show()
                        
                    except Exception as e:
                        print(f"可视化失败: {e}")
        
        elif choice == '4':
            print("\n创建测试数据...")
            try:
                from evaluation_demo import demo_create_synthetic_data
                demo_create_synthetic_data()
            except:
                print("创建测试数据功能暂不可用")
        
        elif choice == '5':
            print("退出程序")
            break
        
        else:
            print("无效选择，请重新输入")
        
        input("\n按 Enter 键继续...")

# ==================== 第五步：直接调用示例 ====================
def simple_example():
    """
    最简单的调用示例
    复制这个函数的内容到你的代码中直接使用
    """
    import cv2
    
    # 1. 导入评估器
    from optical_flow_evaluator import OpticalFlowEvaluator, read_flo_file
    
    # 2. 加载你的数据
    flow = read_flo_file("out.flo")           # 你的光流文件
    img1 = cv2.imread("three.png")            # 第一帧
    img2 = cv2.imread("four.png")             # 第二帧
    # gt = read_flo_file("ground_truth.flo")  # 如果有真值
    
    # 3. 创建评估器
    evaluator = OpticalFlowEvaluator(
        flow=flow,
        img1=img1,
        img2=img2,
        ground_truth=None,  # 没有真值设为None
        flow_name="YourAlgorithm"
    )
    
    # 4. 执行评估
    results = evaluator.evaluate_all(verbose=True)
    
    # 5. 保存结果
    evaluator.save_report("my_evaluation_report.txt")
    
    # 6. 可视化
    evaluator.visualize_results()
    
    return results

# ==================== 运行 ====================
if __name__ == "__main__":
    # 如果你想直接运行简单示例，取消下面的注释
    # simple_example()
    
    # 或者运行完整的主菜单
    main()