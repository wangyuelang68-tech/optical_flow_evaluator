#!/usr/bin/env python
"""
光学流质量综合评估工具箱
作者:Yuelang Wang
日期:2026年
功能：提供完整的光流评估指标和可视化
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')

class OpticalFlowEvaluator:
    """
    光学流质量评估器
    
    支持以下评估：
    1. 基本统计信息
    2. 平滑度评估
    3. 光度一致性（无真值情况）
    4. 视觉质量评估
    5. 与真值比较（有真值情况）
    6. 综合质量评分
    """
    
    def __init__(self, flow, img1=None, img2=None, ground_truth=None, flow_name="Unknown"):
        """
        初始化评估器
        
        参数:
        ----------
        flow : numpy.ndarray
            估计的光流场，形状为 (H, W, 2)
        img1 : numpy.ndarray, optional
            第一帧图像
        img2 : numpy.ndarray, optional
            第二帧图像
        ground_truth : numpy.ndarray, optional
            真实光流场，形状为 (H, W, 2)
        flow_name : str
            光流算法名称
        """
        self.flow = flow.astype(np.float32)
        self.img1 = img1
        self.img2 = img2
        self.ground_truth = ground_truth
        self.flow_name = flow_name
        
        self.h, self.w = flow.shape[:2]
        self.results = {}
        self.visualizations = {}
        
        # 验证输入
        self._validate_inputs()
        
    def _validate_inputs(self):
        """验证输入数据"""
        assert len(self.flow.shape) == 3, "光流必须是3维数组 (H, W, 2)"
        assert self.flow.shape[2] == 2, "光流的第三维必须是2 (u, v)"
        
        if self.img1 is not None and self.img2 is not None:
            assert self.img1.shape[:2] == self.flow.shape[:2], "图像尺寸与光流不匹配"
            assert self.img2.shape[:2] == self.flow.shape[:2], "图像尺寸与光流不匹配"
            
        if self.ground_truth is not None:
            assert self.ground_truth.shape == self.flow.shape, "真值尺寸与估计光流不匹配"
    
    def evaluate_all(self, verbose=True):
        """
        执行所有可用的评估
        
        参数:
        ----------
        verbose : bool
            是否打印详细报告
            
        返回:
        ----------
        dict : 所有评估结果
        """
        if verbose:
            print("=" * 80)
            print(f"光学流质量评估报告 - {self.flow_name}")
            print("=" * 80)
            print(f"图像尺寸: {self.w} × {self.h}")
            print(f"总像素数: {self.w * self.h:,}")
        
        # 执行评估
        self._basic_statistics()
        self._smoothness_evaluation()
        
        if self.img1 is not None and self.img2 is not None:
            self._photometric_consistency()
            self._visual_quality_assessment()
        
        if self.ground_truth is not None:
            self._ground_truth_comparison()
        
        # 计算综合评分
        self._compute_overall_score()
        
        # 生成可视化
        self._generate_visualizations()
        
        if verbose:
            self._print_detailed_report()
        
        return self.results
    
    def _basic_statistics(self):
        """计算基本统计信息"""
        u = self.flow[:, :, 0]
        v = self.flow[:, :, 1]
        magnitude = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, u) * 180 / np.pi
        
        # 移除无效值（NaN和无穷大）
        valid_mask = np.isfinite(magnitude) & (magnitude < 1e6)
        
        if not np.any(valid_mask):
            raise ValueError("光流数据无效（全部为NaN或无穷大）")
        
        magnitude_valid = magnitude[valid_mask]
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        
        self.results['basic_stats'] = {
            'u_min': float(np.min(u_valid)),
            'u_max': float(np.max(u_valid)),
            'u_mean': float(np.mean(u_valid)),
            'u_std': float(np.std(u_valid)),
            'v_min': float(np.min(v_valid)),
            'v_max': float(np.max(v_valid)),
            'v_mean': float(np.mean(v_valid)),
            'v_std': float(np.std(v_valid)),
            'magnitude_min': float(np.min(magnitude_valid)),
            'magnitude_max': float(np.max(magnitude_valid)),
            'magnitude_mean': float(np.mean(magnitude_valid)),
            'magnitude_std': float(np.std(magnitude_valid)),
            'magnitude_median': float(np.median(magnitude_valid)),
            'angle_min': float(np.min(angle[valid_mask])),
            'angle_max': float(np.max(angle[valid_mask])),
            'zero_flow_ratio': float(np.sum(magnitude_valid < 0.1) / magnitude_valid.size),
            'large_flow_ratio': float(np.sum(magnitude_valid > 10.0) / magnitude_valid.size),
            'valid_pixel_ratio': float(np.sum(valid_mask) / magnitude.size)
        }
    
    def _smoothness_evaluation(self):
        """评估光流平滑度"""
        u = self.flow[:, :, 0]
        v = self.flow[:, :, 1]
        
        # 计算梯度
        u_grad_x = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3)
        u_grad_y = cv2.Sobel(u, cv2.CV_64F, 0, 1, ksize=3)
        v_grad_x = cv2.Sobel(v, cv2.CV_64F, 1, 0, ksize=3)
        v_grad_y = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=3)
        
        grad_magnitude = np.sqrt(u_grad_x**2 + u_grad_y**2 + v_grad_x**2 + v_grad_y**2)
        
        # 计算二阶导数（拉普拉斯）
        u_laplacian = cv2.Laplacian(u, cv2.CV_64F)
        v_laplacian = cv2.Laplacian(v, cv2.CV_64F)
        laplacian_magnitude = np.sqrt(u_laplacian**2 + v_laplacian**2)
        
        # 检测不连续性
        grad_threshold = np.percentile(grad_magnitude, 95)
        discontinuity_mask = grad_magnitude > grad_threshold
        
        # 区域一致性（局部方差）
        window_size = 5
        u_local_var = cv2.boxFilter(u**2, -1, (window_size, window_size)) - \
                     cv2.boxFilter(u, -1, (window_size, window_size))**2
        v_local_var = cv2.boxFilter(v**2, -1, (window_size, window_size)) - \
                     cv2.boxFilter(v, -1, (window_size, window_size))**2
        local_inconsistency = np.sqrt(u_local_var + v_local_var)
        
        self.results['smoothness'] = {
            'avg_gradient': float(np.nanmean(grad_magnitude)),
            'max_gradient': float(np.nanmax(grad_magnitude)),
            'gradient_std': float(np.nanstd(grad_magnitude)),
            'avg_laplacian': float(np.nanmean(laplacian_magnitude)),
            'discontinuity_ratio': float(np.sum(discontinuity_mask) / discontinuity_mask.size),
            'local_inconsistency_mean': float(np.nanmean(local_inconsistency)),
            'smoothness_score': float(1.0 / (1.0 + np.nanmean(grad_magnitude)))  # 简单评分
        }
    
    def _photometric_consistency(self):
        """光度一致性评估"""
        h, w = self.h, self.w
        
        # 创建扭曲坐标
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        map_x = (x_coords + self.flow[:, :, 0]).astype(np.float32)
        map_y = (y_coords + self.flow[:, :, 1]).astype(np.float32)
        
        # 检查坐标是否在边界内
        valid_mask = (map_x >= 0) & (map_x < w) & (map_y >= 0) & (map_y < h)
        valid_ratio = np.sum(valid_mask) / valid_mask.size
        
        if valid_ratio < 0.9:
            warnings.warn(f"只有 {valid_ratio*100:.1f}% 的像素在扭曲后保持有效")
        
        # 扭曲第二张图像
        img2_warped = cv2.remap(self.img2, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # 转换为灰度图像
        if len(self.img1.shape) == 3:
            img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            img2_warped_gray = cv2.cvtColor(img2_warped, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = self.img1
            img2_warped_gray = img2_warped
        
        # 计算误差
        error = np.abs(img1_gray.astype(np.float32) - img2_warped_gray.astype(np.float32))
        
        # MSE 和 PSNR
        mse = np.mean(error**2)
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            psnr = 100
        
        # 光度误差统计
        error_hist, error_bins = np.histogram(error.flatten(), bins=50, range=(0, 255))
        
        self.results['photometric'] = {
            'mean_error': float(np.mean(error)),
            'median_error': float(np.median(error)),
            'std_error': float(np.std(error)),
            'max_error': float(np.max(error)),
            'psnr': float(psnr),
            'mse': float(mse),
            'valid_warp_ratio': float(valid_ratio),
            'error_histogram': (error_hist, error_bins),
            'warped_image': img2_warped
        }
        
        self.visualizations['warped_image'] = img2_warped
        self.visualizations['error_map'] = error
    
    def _visual_quality_assessment(self):
        """视觉质量评估"""
        if 'warped_image' not in self.results['photometric']:
            return
        
        img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY) if len(self.img1.shape) == 3 else self.img1
        img2_warped = self.results['photometric']['warped_image']
        img2_warped_gray = cv2.cvtColor(img2_warped, cv2.COLOR_BGR2GRAY) if len(img2_warped.shape) == 3 else img2_warped
        
        # SSIM
        ssim_score = ssim(img1_gray, img2_warped_gray, 
                         data_range=img1_gray.max() - img1_gray.min(),
                         win_size=7)
        
        # 互信息
        hist_2d, _, _ = np.histogram2d(img1_gray.flatten(), img2_warped_gray.flatten(), bins=50)
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        # 计算互信息
        mi = 0
        for i in range(len(px)):
            for j in range(len(py)):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
        
        self.results['visual'] = {
            'ssim': float(ssim_score),
            'mutual_information': float(mi),
            'quality_level': self._get_quality_level(ssim_score)
        }
    
    def _ground_truth_comparison(self):
        """与真值比较"""
        u_error = self.flow[:, :, 0] - self.ground_truth[:, :, 0]
        v_error = self.flow[:, :, 1] - self.ground_truth[:, :, 1]
        
        # 端点误差 (EPE)
        epe = np.sqrt(u_error**2 + v_error**2)
        
        # 角度误差 (AAE)
        est_u = self.flow[:, :, 0].flatten()
        est_v = self.flow[:, :, 1].flatten()
        gt_u = self.ground_truth[:, :, 0].flatten()
        gt_v = self.ground_truth[:, :, 1].flatten()
        
        numerator = est_u * gt_u + est_v * gt_v + 1
        denominator = np.sqrt((est_u**2 + est_v**2 + 1) * (gt_u**2 + gt_v**2 + 1))
        denominator = np.clip(denominator, 1e-10, None)  # 避免除零
        angular_error = np.degrees(np.arccos(np.clip(numerator / denominator, -1.0, 1.0)))
        
        # 准确率在不同阈值下
        thresholds = [0.5, 1.0, 2.0, 3.0, 5.0]
        accuracy = {}
        for thresh in thresholds:
            accuracy[f'acc_{thresh}px'] = float(np.sum(epe < thresh) / epe.size)
        
        # 异常值检测
        outlier_threshold = np.percentile(epe, 95)  # 95分位数
        outlier_ratio = np.sum(epe > outlier_threshold) / epe.size
        
        self.results['ground_truth'] = {
            'mean_epe': float(np.mean(epe)),
            'median_epe': float(np.median(epe)),
            'std_epe': float(np.std(epe)),
            'max_epe': float(np.max(epe)),
            'mean_aae': float(np.mean(angular_error)),
            **accuracy,
            'outlier_ratio': float(outlier_ratio),
            'epe_map': epe
        }
        
        self.visualizations['epe_map'] = epe
    
    def _compute_overall_score(self):
        """计算综合质量评分 (0-100)"""
        score_components = {}
        total_weight = 0
        total_score = 0
        
        # 1. 平滑度评分 (权重: 0.2)
        if 'smoothness' in self.results:
            smoothness = self.results['smoothness']['smoothness_score']
            score_components['smoothness'] = smoothness * 100
            total_score += smoothness * 100 * 0.2
            total_weight += 0.2
        
        # 2. 光度一致性评分 (权重: 0.3)
        if 'photometric' in self.results:
            psnr = self.results['photometric']['psnr']
            # 归一化PSNR: 假设PSNR在20-40之间为合理范围
            psnr_score = min(max((psnr - 20) / (40 - 20), 0), 1) * 100
            score_components['photometric'] = psnr_score
            total_score += psnr_score * 0.3
            total_weight += 0.3
        
        # 3. 视觉质量评分 (权重: 0.3)
        if 'visual' in self.results:
            ssim_score = self.results['visual']['ssim']
            visual_score = ssim_score * 100
            score_components['visual'] = visual_score
            total_score += visual_score * 0.3
            total_weight += 0.3
        
        # 4. 与真值比较评分 (权重: 0.2，如果有真值)
        if 'ground_truth' in self.results:
            mean_epe = self.results['ground_truth']['mean_epe']
            # EPE越小越好，转换为分数
            epe_score = max(0, 100 - mean_epe * 10)  # 每像素误差减10分
            score_components['ground_truth'] = epe_score
            total_score += epe_score * 0.2
            total_weight += 0.2
        
        # 如果总权重小于1，调整分数
        if total_weight > 0:
            overall_score = total_score / total_weight
        else:
            overall_score = 0
        
        self.results['overall_score'] = {
            'score': float(overall_score),
            'components': score_components,
            'grade': self._get_grade(overall_score)
        }
    
    def _generate_visualizations(self):
        """生成可视化图像"""
        # 光流彩色编码
        flow_color = self._flow_to_color(self.flow)
        self.visualizations['flow_color'] = flow_color
        
        # 幅度图
        magnitude = np.sqrt(self.flow[:, :, 0]**2 + self.flow[:, :, 1]**2)
        self.visualizations['magnitude'] = magnitude
        
        # 方向图
        angle = np.arctan2(self.flow[:, :, 1], self.flow[:, :, 0])
        self.visualizations['angle'] = angle
    
    def _flow_to_color(self, flow):
        """将光流转换为彩色图像"""
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 归一化幅度
        mag_normalized = np.clip(mag / (np.percentile(mag, 95) + 1e-5), 0, 1)
        
        hsv[..., 0] = ang * 180 / np.pi / 2  # 色调
        hsv[..., 1] = 255  # 饱和度
        hsv[..., 2] = np.uint8(mag_normalized * 255)  # 亮度
        
        flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return flow_color
    
    def _get_quality_level(self, ssim_score):
        """根据SSIM分数判断质量等级"""
        if ssim_score > 0.95:
            return "优秀 (Excellent)"
        elif ssim_score > 0.90:
            return "良好 (Good)"
        elif ssim_score > 0.80:
            return "一般 (Fair)"
        elif ssim_score > 0.70:
            return "较差 (Poor)"
        else:
            return "很差 (Bad)"
    
    def _get_grade(self, score):
        """根据综合评分判断等级"""
        if score >= 90:
            return "A+ (优秀)"
        elif score >= 80:
            return "A (良好)"
        elif score >= 70:
            return "B (一般)"
        elif score >= 60:
            return "C (及格)"
        else:
            return "D (不及格)"
    
    def _print_detailed_report(self):
        """打印详细评估报告"""
        print("\n" + "=" * 80)
        print("详细评估结果")
        print("=" * 80)
        
        # 基本统计
        if 'basic_stats' in self.results:
            stats = self.results['basic_stats']
            print("\n[基本统计信息]")
            print(f"  水平分量 (u): {stats['u_mean']:.3f} ± {stats['u_std']:.3f} "
                  f"[{stats['u_min']:.3f}, {stats['u_max']:.3f}]")
            print(f"  垂直分量 (v): {stats['v_mean']:.3f} ± {stats['v_std']:.3f} "
                  f"[{stats['v_min']:.3f}, {stats['v_max']:.3f}]")
            print(f"  光流幅度: {stats['magnitude_mean']:.3f} ± {stats['magnitude_std']:.3f} "
                  f"[{stats['magnitude_min']:.3f}, {stats['magnitude_max']:.3f}]")
            print(f"  零运动区域: {stats['zero_flow_ratio']*100:.1f}%")
            print(f"  大运动区域 (>10px): {stats['large_flow_ratio']*100:.1f}%")
            print(f"  有效像素比例: {stats['valid_pixel_ratio']*100:.1f}%")
        
        # 平滑度
        if 'smoothness' in self.results:
            smooth = self.results['smoothness']
            print("\n[平滑度评估]")
            print(f"  平均梯度: {smooth['avg_gradient']:.3f}")
            print(f"  不连续区域比例: {smooth['discontinuity_ratio']*100:.1f}%")
            print(f"  局部不一致性: {smooth['local_inconsistency_mean']:.3f}")
            print(f"  平滑度分数: {smooth['smoothness_score']:.3f}")
        
        # 光度一致性
        if 'photometric' in self.results:
            photo = self.results['photometric']
            print("\n[光度一致性]")
            print(f"  平均误差: {photo['mean_error']:.2f}")
            print(f"  中值误差: {photo['median_error']:.2f}")
            print(f"  PSNR: {photo['psnr']:.1f} dB")
            print(f"  MSE: {photo['mse']:.2f}")
            print(f"  有效扭曲比例: {photo['valid_warp_ratio']*100:.1f}%")
        
        # 视觉质量
        if 'visual' in self.results:
            visual = self.results['visual']
            print("\n[视觉质量]")
            print(f"  SSIM: {visual['ssim']:.4f}")
            print(f"  互信息: {visual['mutual_information']:.3f}")
            print(f"  质量等级: {visual['quality_level']}")
        
        # 与真值比较
        if 'ground_truth' in self.results:
            gt = self.results['ground_truth']
            print("\n[与真值比较]")
            print(f"  平均端点误差 (EPE): {gt['mean_epe']:.3f} 像素")
            print(f"  中值端点误差: {gt['median_epe']:.3f} 像素")
            print(f"  平均角度误差 (AAE): {gt['mean_aae']:.2f}°")
            
            print(f"  准确率:")
            for key, value in gt.items():
                if key.startswith('acc_'):
                    print(f"    {key}: {value*100:.1f}%")
            
            print(f"  异常值比例: {gt['outlier_ratio']*100:.1f}%")
        
        # 综合评分
        if 'overall_score' in self.results:
            overall = self.results['overall_score']
            print("\n[综合评估]")
            print(f"  综合评分: {overall['score']:.1f}/100")
            print(f"  等级: {overall['grade']}")
            
            if overall['components']:
                print(f"  各组件分数:")
                for comp, score in overall['components'].items():
                    print(f"    {comp}: {score:.1f}")
        
        print("\n" + "=" * 80)
        print("评估完成!")
        print("=" * 80)
    
    def save_report(self, filename="flow_evaluation_report.txt"):
        """保存评估报告到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"光学流质量评估报告 - {self.flow_name}\n")
            f.write("=" * 80 + "\n")
            f.write(f"评估时间: {np.datetime64('now')}\n")
            f.write(f"图像尺寸: {self.w} × {self.h}\n\n")
            
            import json
            f.write(json.dumps(self.results, indent=2, ensure_ascii=False))
        
        print(f"报告已保存到: {filename}")
    
    def visualize_results(self, save_dir="./visualizations"):
        """可视化所有结果"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(16, 12))
        
        # 1. 光流彩色编码
        plt.subplot(3, 3, 1)
        plt.imshow(self.visualizations['flow_color'])
        plt.title('Color-coded Optical Flow')
        plt.axis('off')
        
        # 2. 光流幅度
        plt.subplot(3, 3, 2)
        plt.imshow(self.visualizations['magnitude'], cmap='hot')
        plt.title('Flow Magnitude')
        plt.colorbar()
        plt.axis('off')
        
        # 3. 扭曲图像
        if 'warped_image' in self.visualizations:
            plt.subplot(3, 3, 3)
            plt.imshow(cv2.cvtColor(self.visualizations['warped_image'], cv2.COLOR_BGR2RGB))
            plt.title('Warped Image (img2 -> img1)')
            plt.axis('off')
        
        # 4. 误差图
        if 'error_map' in self.visualizations:
            plt.subplot(3, 3, 4)
            plt.imshow(self.visualizations['error_map'], cmap='hot')
            plt.title('Photometric Error Map')
            plt.colorbar()
            plt.axis('off')
        
        # 5. EPE图（如果有真值）
        if 'epe_map' in self.visualizations:
            plt.subplot(3, 3, 5)
            plt.imshow(self.visualizations['epe_map'], cmap='hot')
            plt.title('Endpoint Error Map')
            plt.colorbar()
            plt.axis('off')
        
        # 6. 光流分量直方图
        plt.subplot(3, 3, 6)
        u_flat = self.flow[:, :, 0].flatten()
        v_flat = self.flow[:, :, 1].flatten()
        plt.hist(u_flat, bins=50, alpha=0.5, label='u', color='red')
        plt.hist(v_flat, bins=50, alpha=0.5, label='v', color='blue')
        plt.title('Flow Component Distribution')
        plt.xlabel('Flow Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. 光度误差直方图
        if 'photometric' in self.results:
            plt.subplot(3, 3, 7)
            error_hist, error_bins = self.results['photometric']['error_histogram']
            plt.bar(error_bins[:-1], error_hist, width=np.diff(error_bins)[0])
            plt.title('Photometric Error Histogram')
            plt.xlabel('Error Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        
        # 8. 综合评分饼图
        plt.subplot(3, 3, 8)
        if 'overall_score' in self.results and self.results['overall_score']['components']:
            components = self.results['overall_score']['components']
            labels = list(components.keys())
            sizes = list(components.values())
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title('Score Components')
        else:
            plt.text(0.5, 0.5, 'No score components', ha='center', va='center')
            plt.axis('off')
        
        # 9. 综合评分显示
        plt.subplot(3, 3, 9)
        plt.axis('off')
        if 'overall_score' in self.results:
            overall = self.results['overall_score']
            plt.text(0.5, 0.7, f"Overall Score", ha='center', va='center', fontsize=16, fontweight='bold')
            plt.text(0.5, 0.5, f"{overall['score']:.1f}/100", ha='center', va='center', fontsize=36, fontweight='bold')
            plt.text(0.5, 0.3, f"Grade: {overall['grade']}", ha='center', va='center', fontsize=14)
        
        plt.suptitle(f'Optical Flow Evaluation - {self.flow_name}', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # 保存图像
        output_path = os.path.join(save_dir, f'{self.flow_name}_evaluation.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"可视化图像已保存到: {output_path}")
        
        plt.show()


def read_flo_file(filename):
    """
    读取 .flo 文件
    
    参数:
    ----------
    filename : str
        .flo 文件路径
        
    返回:
    ----------
    numpy.ndarray : 光流场，形状为 (H, W, 2)
    """
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


def batch_evaluate_flows(flow_files, img1_files=None, img2_files=None, 
                         gt_files=None, flow_names=None, output_dir="./evaluation_results"):
    """
    批量评估多个光流文件
    
    参数:
    ----------
    flow_files : list
        光流文件路径列表
    img1_files : list, optional
        第一帧图像路径列表
    img2_files : list, optional
        第二帧图像路径列表
    gt_files : list, optional
        真值文件路径列表
    flow_names : list, optional
        光流算法名称列表
    output_dir : str
        输出目录
        
    返回:
    ----------
    pandas.DataFrame : 所有评估结果的汇总
    """
    import os
    import pandas as pd
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for i, flow_file in enumerate(tqdm(flow_files, desc="Evaluating flows")):
        try:
            # 读取光流
            flow = read_flo_file(flow_file)
            
            # 读取图像
            img1 = cv2.imread(img1_files[i]) if img1_files else None
            img2 = cv2.imread(img2_files[i]) if img2_files else None
            
            # 读取真值
            gt = read_flo_file(gt_files[i]) if gt_files else None
            
            # 获取算法名称
            flow_name = flow_names[i] if flow_names else os.path.basename(flow_file).replace('.flo', '')
            
            # 评估
            evaluator = OpticalFlowEvaluator(flow, img1, img2, gt, flow_name)
            results = evaluator.evaluate_all(verbose=False)
            
            # 提取关键指标
            summary = {
                'flow_name': flow_name,
                'file': flow_file,
                'image_size': f"{flow.shape[1]}x{flow.shape[0]}"
            }
            
            # 添加各个维度的指标
            for category, metrics in results.items():
                if category not in ['overall_score', 'photometric'] or category == 'overall_score':
                    for key, value in metrics.items():
                        if isinstance(value, (int, float, str)):
                            summary[f"{category}_{key}"] = value
            
            all_results.append(summary)
            
            # 保存单个评估结果
            evaluator.save_report(os.path.join(output_dir, f"{flow_name}_report.txt"))
            evaluator.visualize_results(os.path.join(output_dir, "visualizations"))
            
        except Exception as e:
            print(f"评估 {flow_file} 时出错: {e}")
            continue
    
    # 创建汇总表格
    df = pd.DataFrame(all_results)
    summary_path = os.path.join(output_dir, "evaluation_summary.csv")
    df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    
    print(f"\n批量评估完成!")
    print(f"评估了 {len(all_results)}/{len(flow_files)} 个光流文件")
    print(f"汇总表格已保存到: {summary_path}")
    
    return df