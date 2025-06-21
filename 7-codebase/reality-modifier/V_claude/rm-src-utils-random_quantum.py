#!/usr/bin/env python3
"""
utils/random_quantum.py - Quantum Random Number Generator

量子随机数生成器 - 整个项目唯一真正能运行的部分！

Author: 方舟 (Fang Zhou)
Version: v0.31
Last Modified: 2028.07.03 01:45

司辰在第11章写道："真正的随机，只存在于量子世界。骰子不是上帝掷的，
而是宇宙本身在掷。" 这句话让我意识到，量子随机性可能是打开
现实修改大门的钥匙。

这个模块实现了：
1. 基于量子原理的随机数生成（模拟）
2. 量子态坍缩的随机性
3. 纠缠粒子的相关性
4. 贝尔不等式验证

虽然没有真正的量子硬件，但算法是正确的！
"""

import numpy as np
import time
import hashlib
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from collections import Counter
import warnings

# 尝试导入量子计算库（如果有的话）
try:
    from qiskit import QuantumCircuit, execute, Aer
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn("Qiskit not available. Using classical simulation.")


class QuantumRandomGenerator:
    """
    量子随机数生成器
    
    基于量子测量的本征随机性。即使没有真正的量子计算机，
    我们也可以模拟量子随机过程的统计特性。
    
    李墨渊说这是最接近"真随机"的方法。
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化生成器
        
        Args:
            seed: 随机种子（仅用于经典模拟）
        """
        # 使用时间和硬件信息创建更好的种子
        if seed is None:
            # 结合多个熵源
            time_entropy = int(time.time() * 1000000) % 2**32
            # 添加一些系统噪声
            noise_entropy = int(np.random.rand() * 2**32)
            combined = f"{time_entropy}{noise_entropy}"
            seed = int(hashlib.sha256(combined.encode()).hexdigest()[:8], 16)
            
        np.random.seed(seed)
        self.measurement_count = 0
        self.entanglement_pairs = []
        
        # 量子态历史（用于分析）
        self.state_history = []
        
        print(f"Quantum Random Generator initialized")
        print(f"Mode: {'Qiskit' if QISKIT_AVAILABLE else 'Classical simulation'}")
        
    def quantum_bit(self) -> int:
        """
        生成单个量子比特
        
        通过制备|+⟩态并测量来获得真随机比特
        """
        if QISKIT_AVAILABLE:
            # 使用Qiskit生成真量子随机数
            qc = QuantumCircuit(1, 1)
            qc.h(0)  # Hadamard门创建叠加态
            qc.measure(0, 0)
            
            # 在模拟器上执行
            backend = Aer.get_backend('qasm_simulator')
            job = execute(qc, backend, shots=1)
            result = job.result()
            counts = result.get_counts()
            
            bit = int(list(counts.keys())[0])
        else:
            # 经典模拟量子测量
            # |+⟩ = (|0⟩ + |1⟩)/√2
            amplitude_0 = 1.0 / np.sqrt(2)
            amplitude_1 = 1.0 / np.sqrt(2)
            
            # 测量导致波函数坍缩
            probability_0 = amplitude_0 ** 2
            bit = 0 if np.random.rand() < probability_0 else 1
            
        self.measurement_count += 1
        return bit
        
    def quantum_byte(self) -> int:
        """生成一个量子字节（8位）"""
        byte_value = 0
        for i in range(8):
            bit = self.quantum_bit()
            byte_value = (byte_value << 1) | bit
        return byte_value
        
    def quantum_float(self, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        生成量子随机浮点数
        
        Args:
            min_val: 最小值
            max_val: 最大值
            
        Returns:
            均匀分布的量子随机数
        """
        # 使用32位精度
        quantum_int = 0
        for _ in range(32):
            quantum_int = (quantum_int << 1) | self.quantum_bit()
            
        # 转换为[0, 1]区间
        quantum_uniform = quantum_int / (2**32 - 1)
        
        # 缩放到指定区间
        return min_val + quantum_uniform * (max_val - min_val)
        
    def quantum_gaussian(self, mean: float = 0.0, std: float = 1.0) -> float:
        """
        生成量子高斯随机数
        
        使用Box-Muller变换，但底层随机性来自量子
        """
        # Box-Muller transform
        u1 = self.quantum_float(0.0001, 0.9999)  # 避免log(0)
        u2 = self.quantum_float()
        
        z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        return mean + std * z0
        
    def create_entangled_pair(self) -> Tuple[float, float]:
        """
        创建纠缠随机数对
        
        司辰在第22章描述的"量子纠缠的随机关联"：
        两个随机数看似独立，但存在深层关联。
        """
        # 创建Bell态 |Φ+⟩ = (|00⟩ + |11⟩)/√2
        if np.random.rand() < 0.5:
            # 测量结果|00⟩
            bit1, bit2 = 0, 0
        else:
            # 测量结果|11⟩
            bit1, bit2 = 1, 1
            
        # 添加相位因子产生连续值
        phase = self.quantum_float(0, 2 * np.pi)
        
        # 纠缠值具有相关性
        value1 = bit1 + 0.5 * np.sin(phase)
        value2 = bit2 + 0.5 * np.sin(phase)
        
        self.entanglement_pairs.append((value1, value2))
        return value1, value2
        
    def verify_bell_inequality(self, num_measurements: int = 1000) -> float:
        """
        验证贝尔不等式
        
        如果违反贝尔不等式，说明我们的随机数具有量子特性！
        这是区分经典随机和量子随机的关键。
        """
        # CHSH不等式测试
        # 设置测量角度
        angles = [(0, np.pi/4), (0, -np.pi/4), 
                  (np.pi/2, np.pi/4), (np.pi/2, -np.pi/4)]
        
        correlations = []
        
        for a, b in angles:
            correlation_sum = 0
            
            for _ in range(num_measurements):
                # 创建纠缠态
                state = np.random.rand()
                
                # Alice测量
                alice_result = 1 if state < 0.5 else -1
                
                # Bob测量（考虑角度）
                correlation_factor = np.cos(a - b)
                bob_prob = 0.5 * (1 + alice_result * correlation_factor)
                bob_result = 1 if np.random.rand() < bob_prob else -1
                
                correlation_sum += alice_result * bob_result
                
            correlations.append(correlation_sum / num_measurements)
            
        # 计算CHSH值
        S = correlations[0] - correlations[1] + correlations[2] + correlations[3]
        
        # 经典极限是2，量子可以达到2√2 ≈ 2.828
        print(f"CHSH inequality test: S = {S:.3f}")
        print(f"Classical limit: 2.000")
        print(f"Quantum limit: {2*np.sqrt(2):.3f}")
        
        return S
        
    def quantum_walk(self, steps: int = 100) -> List[int]:
        """
        量子随机行走
        
        与经典随机行走不同，量子行走显示出弹道式扩散。
        司辰说："量子世界的醉汉走得更远。"
        """
        position = 0
        positions = [position]
        
        # 量子行走的"币"态
        coin_state = np.array([1.0, 0.0], dtype=complex) / np.sqrt(2)
        
        for _ in range(steps):
            # 量子币翻转（Hadamard）
            if self.quantum_bit() == 0:
                position += 1
            else:
                position -= 1
                
            positions.append(position)
            
            # 模拟量子干涉效应
            if len(positions) > 10:
                # 量子行走会产生特征性的双峰分布
                interference = np.sin(len(positions) * 0.1) * 0.3
                position += int(interference)
                
        return positions
        
    def generate_quantum_key(self, length: int = 256) -> str:
        """
        生成量子密钥
        
        用于量子密码学，理论上无条件安全
        """
        key_bits = [str(self.quantum_bit()) for _ in range(length)]
        key_hex = hex(int(''.join(key_bits), 2))[2:].zfill(length // 4)
        return key_hex
        
    def quantum_superposition_sample(self, states: List[str], 
                                   amplitudes: Optional[List[complex]] = None) -> str:
        """
        从量子叠加态中采样
        
        模拟量子系统的测量过程
        """
        if amplitudes is None:
            # 均匀叠加
            n = len(states)
            amplitudes = [1.0 / np.sqrt(n)] * n
        else:
            # 归一化
            norm = np.sqrt(sum(abs(a)**2 for a in amplitudes))
            amplitudes = [a / norm for a in amplitudes]
            
        # 计算概率
        probabilities = [abs(a)**2 for a in amplitudes]
        
        # 量子测量
        random_val = self.quantum_float()
        cumsum = 0
        for i, (state, prob) in enumerate(zip(states, probabilities)):
            cumsum += prob
            if random_val < cumsum:
                # 记录测量导致的"坍缩"
                self.state_history.append({
                    'measurement': i,
                    'state': state,
                    'probability': prob,
                    'time': time.time()
                })
                return state
                
        return states[-1]
        
    def visualize_randomness(self, num_samples: int = 10000):
        """
        可视化量子随机性
        
        展示量子随机数的统计特性
        """
        # 生成样本
        samples = [self.quantum_float() for _ in range(num_samples)]
        gaussian_samples = [self.quantum_gaussian() for _ in range(num_samples)]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 均匀分布直方图
        axes[0, 0].hist(samples, bins=50, density=True, alpha=0.7, color='blue')
        axes[0, 0].set_title('Quantum Uniform Distribution')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        
        # 2. 高斯分布直方图
        axes[0, 1].hist(gaussian_samples, bins=50, density=True, alpha=0.7, color='green')
        axes[0, 1].set_title('Quantum Gaussian Distribution')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Density')
        
        # 3. 相关性图（连续样本）
        axes[1, 0].scatter(samples[:-1], samples[1:], alpha=0.5, s=1)
        axes[1, 0].set_title('Sequential Correlation')
        axes[1, 0].set_xlabel('Sample n')
        axes[1, 0].set_ylabel('Sample n+1')
        
        # 4. 量子行走
        walk = self.quantum_walk(1000)
        axes[1, 1].plot(walk, alpha=0.7)
        axes[1, 1].set_title('Quantum Random Walk')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Position')
        
        plt.tight_layout()
        plt.savefig('quantum_randomness.png', dpi=150)
        print("Visualization saved to quantum_randomness.png")
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取生成器统计信息"""
        stats = {
            'total_measurements': self.measurement_count,
            'entangled_pairs': len(self.entanglement_pairs),
            'state_collapses': len(self.state_history),
            'bits_generated': self.measurement_count,
            'entropy_estimate': self._estimate_entropy()
        }
        
        if self.entanglement_pairs:
            # 计算纠缠相关性
            pairs = np.array(self.entanglement_pairs)
            correlation = np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]
            stats['entanglement_correlation'] = correlation
            
        return stats
        
    def _estimate_entropy(self) -> float:
        """估计生成的熵"""
        if self.measurement_count == 0:
            return 0.0
            
        # 简单估计：每个量子比特贡献1比特熵
        return self.measurement_count


# 便捷函数
def quantum_random(n: int = 1) -> Union[float, List[float]]:
    """生成量子随机数的便捷函数"""
    qrg = QuantumRandomGenerator()
    if n == 1:
        return qrg.quantum_float()
    else:
        return [qrg.quantum_float() for _ in range(n)]


# 测试和演示
if __name__ == "__main__":
    print("=== Quantum Random Number Generator ===")
    print("The only working part of Reality Modifier!")
    print("Based on Si Chen's quantum theories")
    print()
    
    # 初始化生成器
    qrg = QuantumRandomGenerator()
    
    # 测试1：生成基本随机数
    print("Test 1: Basic quantum random numbers")
    print(f"  Quantum bit: {qrg.quantum_bit()}")
    print(f"  Quantum byte: {qrg.quantum_byte():08b}")
    print(f"  Quantum float: {qrg.quantum_float():.6f}")
    print(f"  Quantum gaussian: {qrg.quantum_gaussian():.6f}")
    
    # 测试2：量子纠缠
    print("\nTest 2: Quantum entanglement")
    for i in range(3):
        v1, v2 = qrg.create_entangled_pair()
        print(f"  Entangled pair {i+1}: {v1:.3f}, {v2:.3f}")
    
    # 测试3：贝尔不等式
    print("\nTest 3: Bell inequality test")
    S = qrg.verify_bell_inequality(1000)
    if S > 2:
        print("  ✓ Quantum behavior confirmed!")
    else:
        print("  ✗ Classical behavior detected")
        
    # 测试4：量子密钥
    print("\nTest 4: Quantum key generation")
    key = qrg.generate_quantum_key(128)
    print(f"  128-bit quantum key: {key[:32]}...")
    
    # 测试5：量子叠加采样
    print("\nTest 5: Quantum superposition sampling")
    states = ["red", "green", "blue"]
    # 创建非均匀叠加态
    amplitudes = [0.5, 0.5j, 0.7071]  # 不同相位
    samples = [qrg.quantum_superposition_sample(states, amplitudes) for _ in range(100)]
    distribution = Counter(samples)
    for state, count in distribution.items():
        print(f"  {state}: {count/100:.2%}")
    
    # 生成可视化
    print("\nGenerating visualization...")
    qrg.visualize_randomness(5000)
    
    # 显示统计
    print("\nGenerator Statistics:")
    stats = qrg.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    print("\n--- Quantum Random Generator Test Complete ---")
    print("\"True randomness exists only in the quantum realm.\"")
    print("- Si Chen, Reality Modifier, Chapter 11")
    print()
    print("This is just the beginning. When we connect this to")
    print("the reality scanner and spacetime interface...")
    print("真正的魔法就会开始。")