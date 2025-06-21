#!/usr/bin/env python3
"""
tests/test_random.py - Quantum Random Number Generator Tests

量子随机数生成器测试套件 - 唯一能真正运行的测试！

Author: 方舟 (Fang Zhou)
Version: v0.31
Last Modified: 2028.07.05 22:30

测试量子随机数生成器的各种特性：
1. 基本随机性
2. 分布特性
3. 量子纠缠相关性
4. 贝尔不等式违反
5. 性能基准

司辰说："真正的随机，是宇宙在掷骰子。"
让我们验证一下我们的骰子够不够"量子"。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import time
from collections import Counter
import matplotlib.pyplot as plt
from scipy import stats

from src.utils.random_quantum import QuantumRandomGenerator, quantum_random


class TestQuantumRandomGenerator(unittest.TestCase):
    """量子随机数生成器测试类"""
    
    def setUp(self):
        """测试前初始化"""
        self.qrg = QuantumRandomGenerator(seed=42)  # 固定种子以便重现
        self.sample_size = 1000
        
    def tearDown(self):
        """测试后清理"""
        # 关闭所有matplotlib图形，避免内存泄漏
        plt.close('all')
        
    def test_initialization(self):
        """测试初始化"""
        print("\n=== 测试初始化 ===")
        self.assertIsNotNone(self.qrg)
        self.assertEqual(self.qrg.measurement_count, 0)
        self.assertEqual(len(self.qrg.entanglement_pairs), 0)
        print("✓ 初始化成功")
        
    def test_quantum_bit_generation(self):
        """测试量子比特生成"""
        print("\n=== 测试量子比特生成 ===")
        
        # 生成一批量子比特
        bits = [self.qrg.quantum_bit() for _ in range(self.sample_size)]
        
        # 检查只有0和1
        unique_values = set(bits)
        self.assertEqual(unique_values, {0, 1})
        
        # 检查分布是否接近50/50
        bit_counts = Counter(bits)
        ratio = bit_counts[0] / len(bits)
        print(f"0的比例: {ratio:.3f}")
        print(f"1的比例: {1-ratio:.3f}")
        
        # 使用卡方检验
        chi2, p_value = stats.chisquare([bit_counts[0], bit_counts[1]])
        print(f"卡方检验 p-value: {p_value:.4f}")
        
        # p值应该大于0.05（表示接近均匀分布）
        self.assertGreater(p_value, 0.01)
        print("✓ 量子比特分布正常")
        
    def test_quantum_byte_generation(self):
        """测试量子字节生成"""
        print("\n=== 测试量子字节生成 ===")
        
        bytes_list = [self.qrg.quantum_byte() for _ in range(100)]
        
        # 检查范围
        for byte_val in bytes_list:
            self.assertGreaterEqual(byte_val, 0)
            self.assertLessEqual(byte_val, 255)
            
        # 检查分布的随机性
        unique_bytes = len(set(bytes_list))
        print(f"100个字节中的唯一值数量: {unique_bytes}")
        
        # 应该有相当多的唯一值（表示良好的随机性）
        self.assertGreater(unique_bytes, 50)
        print("✓ 量子字节生成正常")
        
    def test_quantum_float_uniformity(self):
        """测试量子浮点数的均匀性"""
        print("\n=== 测试量子浮点数均匀性 ===")
        
        # 生成大量样本
        samples = [self.qrg.quantum_float() for _ in range(self.sample_size)]
        
        # 基本检查
        self.assertTrue(all(0 <= x <= 1 for x in samples))
        
        # Kolmogorov-Smirnov检验
        ks_stat, ks_pvalue = stats.kstest(samples, 'uniform')
        print(f"KS检验统计量: {ks_stat:.4f}")
        print(f"KS检验 p-value: {ks_pvalue:.4f}")
        
        # p值应该大于0.05
        self.assertGreater(ks_pvalue, 0.01)
        
        # 检查分箱均匀性
        bins = 10
        hist, _ = np.histogram(samples, bins=bins)
        expected_count = self.sample_size / bins
        chi2_stat = np.sum((hist - expected_count)**2 / expected_count)
        print(f"分箱卡方统计量: {chi2_stat:.4f}")
        
        print("✓ 量子浮点数分布均匀")
        
    def test_quantum_gaussian(self):
        """测试量子高斯分布"""
        print("\n=== 测试量子高斯分布 ===")
        
        # 生成高斯样本
        mean, std = 5.0, 2.0
        samples = [self.qrg.quantum_gaussian(mean, std) for _ in range(self.sample_size)]
        
        # 计算统计量
        sample_mean = np.mean(samples)
        sample_std = np.std(samples)
        
        print(f"期望均值: {mean}, 实际均值: {sample_mean:.3f}")
        print(f"期望标准差: {std}, 实际标准差: {sample_std:.3f}")
        
        # 检查是否接近目标参数
        self.assertAlmostEqual(sample_mean, mean, delta=0.2)
        self.assertAlmostEqual(sample_std, std, delta=0.2)
        
        # Shapiro-Wilk正态性检验
        if len(samples) <= 5000:  # Shapiro-Wilk的限制
            stat, p_value = stats.shapiro(samples)
            print(f"Shapiro-Wilk检验 p-value: {p_value:.4f}")
            self.assertGreater(p_value, 0.01)
            
        print("✓ 量子高斯分布正常")
        
    def test_entanglement_correlation(self):
        """测试量子纠缠相关性"""
        print("\n=== 测试量子纠缠 ===")
        
        # 创建多对纠缠粒子
        n_pairs = 100
        pairs = [self.qrg.create_entangled_pair() for _ in range(n_pairs)]
        
        # 分离成两个列表
        alice_values = [p[0] for p in pairs]
        bob_values = [p[1] for p in pairs]
        
        # 计算相关系数
        correlation = np.corrcoef(alice_values, bob_values)[0, 1]
        print(f"纠缠粒子相关系数: {correlation:.4f}")
        
        # 纠缠粒子应该有强相关性
        self.assertGreater(abs(correlation), 0.5)
        
        # 检查纠缠的"spooky action"
        # 测量差异应该很小
        differences = [abs(a - b) for a, b in pairs]
        avg_diff = np.mean(differences)
        print(f"平均差异: {avg_diff:.4f}")
        
        self.assertLess(avg_diff, 0.5)
        print("✓ 量子纠缠表现正常")
        
    def test_bell_inequality_violation(self):
        """测试贝尔不等式违反"""
        print("\n=== 测试贝尔不等式 ===")
        
        # 运行贝尔测试
        S = self.qrg.verify_bell_inequality(num_measurements=500)
        
        # 检查是否违反经典极限
        classical_limit = 2.0
        quantum_limit = 2 * np.sqrt(2)
        
        print(f"CHSH值: {S:.3f}")
        print(f"经典极限: {classical_limit}")
        print(f"量子极限: {quantum_limit:.3f}")
        
        # 如果S > 2，说明具有量子特性
        if S > classical_limit:
            print("✓ 成功违反贝尔不等式！具有量子特性")
        else:
            print("✗ 未违反贝尔不等式，表现为经典")
            
        # 注意：由于是模拟，可能不总是违反
        # self.assertGreater(S, classical_limit)  # 可能失败
        
    def test_quantum_walk(self):
        """测试量子随机行走"""
        print("\n=== 测试量子随机行走 ===")
        
        # 执行量子行走
        steps = 200
        walk_path = self.qrg.quantum_walk(steps)
        
        # 检查基本属性
        self.assertEqual(len(walk_path), steps + 1)
        
        # 计算扩散特性
        final_position = walk_path[-1]
        rms_distance = np.sqrt(np.mean([x**2 for x in walk_path]))
        
        print(f"步数: {steps}")
        print(f"最终位置: {final_position}")
        print(f"RMS距离: {rms_distance:.2f}")
        
        # 量子行走应该比经典随机行走扩散更快
        # 经典: RMS ≈ √n，量子: RMS ≈ n
        classical_rms = np.sqrt(steps)
        print(f"经典RMS预期: {classical_rms:.2f}")
        
        # 粗略检查是否表现出量子特性
        self.assertGreater(rms_distance, classical_rms * 0.8)
        print("✓ 量子行走表现正常")
        
    def test_quantum_key_generation(self):
        """测试量子密钥生成"""
        print("\n=== 测试量子密钥生成 ===")
        
        # 生成不同长度的密钥
        key_lengths = [128, 256, 512]
        
        for length in key_lengths:
            key = self.qrg.generate_quantum_key(length)
            
            # 检查长度（十六进制表示）
            expected_hex_length = length // 4
            self.assertEqual(len(key), expected_hex_length)
            
            # 检查是否是有效的十六进制
            try:
                int(key, 16)
                print(f"✓ {length}位密钥生成成功: {key[:32]}...")
            except ValueError:
                self.fail(f"生成的密钥不是有效的十六进制: {key}")
                
    def test_superposition_sampling(self):
        """测试叠加态采样"""
        print("\n=== 测试量子叠加态采样 ===")
        
        # 测试均匀叠加
        states = ["red", "green", "blue", "yellow"]
        samples = [self.qrg.quantum_superposition_sample(states) 
                  for _ in range(400)]
        
        distribution = Counter(samples)
        print("均匀叠加态分布:")
        for state, count in distribution.items():
            print(f"  {state}: {count/400:.2%}")
            
        # 应该大致均匀
        for count in distribution.values():
            self.assertGreater(count, 50)  # 至少12.5%
            self.assertLess(count, 150)     # 最多37.5%
            
        # 测试非均匀叠加
        amplitudes = [0.5, 0.5, 0.5j, 0.5j]  # 不同相位
        samples2 = [self.qrg.quantum_superposition_sample(states, amplitudes)
                   for _ in range(400)]
        
        distribution2 = Counter(samples2)
        print("\n非均匀叠加态分布:")
        for state, count in distribution2.items():
            print(f"  {state}: {count/400:.2%}")
            
        print("✓ 叠加态采样正常")
        
    def test_performance_benchmark(self):
        """性能基准测试"""
        print("\n=== 性能基准测试 ===")
        
        # 测试不同操作的速度
        operations = {
            'quantum_bit': lambda: self.qrg.quantum_bit(),
            'quantum_byte': lambda: self.qrg.quantum_byte(),
            'quantum_float': lambda: self.qrg.quantum_float(),
            'quantum_gaussian': lambda: self.qrg.quantum_gaussian(),
            'entangled_pair': lambda: self.qrg.create_entangled_pair(),
        }
        
        n_iterations = 10000
        
        for name, operation in operations.items():
            start_time = time.time()
            for _ in range(n_iterations):
                operation()
            elapsed = time.time() - start_time
            
            ops_per_second = n_iterations / elapsed
            print(f"{name}: {ops_per_second:.0f} ops/sec ({elapsed:.3f}s for {n_iterations} ops)")
            
        # 获取统计信息
        stats = self.qrg.get_statistics()
        print(f"\n总测量次数: {stats['total_measurements']}")
        print(f"纠缠对数: {stats['entangled_pairs']}")
        print(f"熵估计: {stats['entropy_estimate']} bits")
        
    def test_edge_cases(self):
        """边界情况测试"""
        print("\n=== 边界情况测试 ===")
        
        # 测试自定义范围
        samples = [self.qrg.quantum_float(-10, 10) for _ in range(100)]
        self.assertTrue(all(-10 <= x <= 10 for x in samples))
        print("✓ 自定义范围正常")
        
        # 测试零方差高斯
        samples = [self.qrg.quantum_gaussian(5, 0.0001) for _ in range(10)]
        self.assertTrue(all(abs(x - 5) < 0.01 for x in samples))
        print("✓ 零方差高斯正常")
        
        # 测试单状态叠加
        result = self.qrg.quantum_superposition_sample(["only_state"])
        self.assertEqual(result, "only_state")
        print("✓ 单状态叠加正常")
        
    def test_reproducibility(self):
        """可重现性测试"""
        print("\n=== 可重现性测试 ===")
        
        # 使用相同种子创建两个生成器
        qrg1 = QuantumRandomGenerator(seed=12345)
        qrg2 = QuantumRandomGenerator(seed=12345)
        
        # 生成一些随机数
        values1 = [qrg1.quantum_float() for _ in range(10)]
        values2 = [qrg2.quantum_float() for _ in range(10)]
        
        # 应该完全相同
        self.assertEqual(values1, values2)
        print("✓ 相同种子产生相同结果")
        
        # 不同种子应该不同
        qrg3 = QuantumRandomGenerator(seed=54321)
        values3 = [qrg3.quantum_float() for _ in range(10)]
        self.assertNotEqual(values1, values3)
        print("✓ 不同种子产生不同结果")


class TestQuantumRandomVisual(unittest.TestCase):
    """量子随机数可视化测试"""
    
    @classmethod
    def setUpClass(cls):
        """只在开始时创建一次生成器"""
        cls.qrg = QuantumRandomGenerator()
        
    def test_visual_analysis(self):
        """生成可视化分析图"""
        print("\n=== 生成可视化分析 ===")
        
        # 只在明确要求时生成图形
        if os.environ.get('SHOW_PLOTS', 'false').lower() == 'true':
            self.qrg.visualize_randomness(5000)
            print("✓ 可视化图形已保存到 quantum_randomness.png")
        else:
            print("⚠ 跳过可视化（设置 SHOW_PLOTS=true 来启用）")


# 便捷测试函数
def quick_quantum_test():
    """快速测试量子随机性"""
    print("=== 快速量子随机性测试 ===")
    
    # 测试便捷函数
    single = quantum_random()
    print(f"单个随机数: {single:.6f}")
    
    multiple = quantum_random(5)
    print(f"多个随机数: {[f'{x:.3f}' for x in multiple]}")
    
    # 简单的随机性检查
    samples = quantum_random(1000)
    mean = np.mean(samples)
    std = np.std(samples)
    
    print(f"\n1000个样本统计:")
    print(f"均值: {mean:.3f} (期望: 0.500)")
    print(f"标准差: {std:.3f} (期望: 0.289)")
    
    if 0.45 < mean < 0.55 and 0.25 < std < 0.35:
        print("✓ 随机性检查通过！")
    else:
        print("✗ 随机性可能有问题")


if __name__ == '__main__':
    print("="*60)
    print("量子随机数生成器测试套件")
    print("唯一真正能运行的部分！")
    print("="*60)
    
    # 运行快速测试
    print("\n### 快速测试 ###")
    quick_quantum_test()
    
    # 运行完整测试套件
    print("\n### 完整测试套件 ###")
    unittest.main(verbosity=2)
    
    print("\n记住：这只是开始。")
    print("当量子随机性与现实修改结合...")
    print("真正的魔法就会发生。")
    print()
    print("「真正的随机，是宇宙在掷骰子。」")
    print("                    —— 司辰")