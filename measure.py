import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
import librosa
from scipy.signal import find_peaks, medfilt

class DPPitchTracker_48k:
    def __init__(self, sr=48000, min_f0=80, max_f0=500, 
                 frame_ms=25, hop_ms=10, channel_mode='avg'):
        """
        参数说明：
            sr: 采样率 (Hz)
            min_f0: 最低基频 (Hz)
            max_f0: 最高基频 (Hz)
            frame_ms: 帧长 (毫秒)
            hop_ms: 帧移 (毫秒)
            channel_mode: 立体声处理方式 ('avg'|'left'|'right')
        """
        self.sr = sr
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.channel_mode = channel_mode
        
        # 计算帧参数
        self.frame_length = int(frame_ms * sr / 1000)
        self.hop_length = int(hop_ms * sr / 1000)
        
        # 基频周期搜索范围
        self.kl = max(10, round(sr / max_f0))  # 最小周期
        self.kr = min(self.frame_length//2, round(sr / min_f0))  # 最大周期

    def _load_audio(self, path):
        """加载48kHz立体声WAV文件"""
        audio, sr = sf.read(path, dtype='float32')
        assert sr == self.sr, f"采样率应为{self.sr}Hz，实际得到{sr}Hz"
        
        # 立体声处理
        if len(audio.shape) == 2:
            if self.channel_mode == 'left':
                audio = audio[:, 0]
            elif self.channel_mode == 'right':
                audio = audio[:, 1]
            else:  # avg
                audio = np.mean(audio, axis=1)
        return audio / (np.max(np.abs(audio)) + 1e-6)

    def _autocorr_48k(self, frame):
        """优化的自相关函数计算"""
        frame = frame - np.mean(frame)
        N = len(frame)
        corr = np.correlate(frame, frame, mode='same')[N//2:]
        return corr / (np.max(corr) + 1e-6)

    def compute_pitch(self, audio_path):
        """核心处理流程"""
        # 1. 加载音频
        audio = self._load_audio(audio_path)
        
        # 2. 分帧
        frames = librosa.util.frame(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        ).T
        
        # 3. 动态规划基频检测
        f0 = self._dp_pitch_tracking(frames)
        
        # 4. 后处理
        f0 = medfilt(f0, kernel_size=5)  # 中值滤波
        f0[(f0 < self.min_f0) | (f0 > self.max_f0)] = 0  # 超出范围置零
        
        # 5. 生成时间戳
        #times = np.arange(len(f0)) * self.hop_length / self.sr
        
        return f0

    def _dp_pitch_tracking(self, frames):
        """动态规划基频跟踪"""
        all_candidates = []
        all_peak_heights = []
        
        for frame in frames:
            # 计算自相关函数
            acf = self._autocorr_48k(frame)
            
            # 寻找候选峰值
            peaks, properties = find_peaks(
                acf[self.kl:self.kr],
                height=0.3,
                prominence=0.5,
                distance=self.kl//2
            )
            candidates = peaks + self.kl
            heights = acf[candidates]
            
            if len(candidates) > 0:
                all_candidates.append(candidates)
                all_peak_heights.append(heights)
            else:
                all_candidates.append(np.array([self.kr]))
                all_peak_heights.append(np.array([0.1]))
        
        # 动态规划路径搜索
        return self._find_optimal_path(all_candidates, all_peak_heights)

    def _find_optimal_path(self, all_candidates, all_heights):
        """动态规划最优路径搜索"""
        n_frames = len(all_candidates)
        if n_frames == 0:
            return np.zeros(0)
        
        # 初始化DP表格
        max_candidates = max(len(c) for c in all_candidates)
        dp_cost = np.full((n_frames, max_candidates), np.inf)
        dp_path = np.zeros((n_frames, max_candidates), dtype=int)
        
        # 第一帧初始化
        dp_cost[0, :len(all_heights[0])] = -np.log(all_heights[0])
        
        # 前向传播
        for i in range(1, n_frames):
            curr_len = len(all_candidates[i])
            prev_len = len(all_candidates[i-1])
            
            for j in range(curr_len):
                curr_period = all_candidates[i][j]
                curr_freq = self.sr / curr_period
                
                min_cost = np.inf
                best_prev = 0
                
                for k in range(prev_len):
                    prev_period = all_candidates[i-1][k]
                    prev_freq = self.sr / prev_period
                    
                    # 代价计算：频率差 + 周期变化惩罚 + 前一帧代价
                    cost = dp_cost[i-1, k] + \
                           np.abs(curr_freq - prev_freq) + \
                           5 * np.abs(np.log2(curr_period/prev_period))
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_prev = k
                
                dp_cost[i, j] = min_cost - np.log(all_heights[i][j])
                dp_path[i, j] = best_prev
        
        # 回溯路径
        best_path = [np.argmin(dp_cost[-1])]
        for i in range(n_frames-1, 0, -1):
            best_path.append(dp_path[i, best_path[-1]])
        best_path.reverse()
        
        # 生成最终轨迹
        f0 = np.zeros(n_frames)
        for i in range(n_frames):
            idx = min(best_path[i], len(all_candidates[i])-1)
            f0[i] = self.sr / all_candidates[i][idx]
        
        return f0

def save_to_excel(times, f0, output_path):
    """将F0序列保存为Excel文件"""
    df = pd.DataFrame({
        'Time (s)': times,
        'Frequency (Hz)': f0,
        'Voiced': f0 > 0  # 添加是否发声标志
    })
    
    # 保存为Excel
    df.to_excel(output_path, index=False, float_format="%.3f")
    print(f"结果已保存到 {output_path}")

# if __name__ == "__main__":
#     # 使用示例
#     input_audio = "./clean/M01/mic_M01_sa1.wav"
#     output_excel = "pitch_results.xlsx"
    
#     # 初始化检测器
#     tracker = DPPitchTracker_48k(
#         sr=48000,
#         min_f0=70,
#         max_f0=600,
#         frame_ms=30,
#         hop_ms=10,
#         channel_mode='avg'
#     )
    
#     # 计算基频
#     times, f0 = tracker.compute_pitch(input_audio)
    
#     # 保存结果
#     save_to_excel(times, f0, output_excel)
    
#     # 打印统计信息
#     voiced_frames = np.sum(f0 > 0)
#     print(f"分析完成，共 {len(f0)} 帧，其中 {voiced_frames} 帧检测到有效基频")
#     print(f"基频范围: {np.min(f0[f0>0]):.1f}-{np.max(f0[f0>0]):.1f} Hz")


def hz_to_cent(f0_array, ref_freq=440.0):
    """将Hz频率数组转换为音分值"""
    cents = np.zeros_like(f0_array)
    valid_mask = f0_array > 0
    cents[valid_mask] = 69 + 12 * np.log2(f0_array[valid_mask] / ref_freq)
    return cents

def load_reference_f0(f0_path):
    """加载.f0文件（仅读取第一列基频值）"""
    try:
        ref_f0 = np.loadtxt(f0_path, usecols=0)
        return ref_f0.astype(float)
    except Exception as e:
        raise ValueError(f"无法读取{f0_path}：{str(e)}")



def frame_aligned_compare(test_f0, ref_f0):
    """
    按帧顺序对齐比较
    返回: {
        'hz_mean_diff': Hz平均绝对差,
        'cent_mean_diff': 音分平均绝对差,
        'hz_std_diff': Hz差值标准差,
        'cent_std_diff': 音分差值标准差,
        'hz_rmse': Hz均方根误差,
        'cent_rmse': 音分均方根误差,
        'corr': 皮尔逊相关系数,
        'voiced_ratio': 有效帧比例,
        'voiced_ratio2': 误检帧比例
    }
    """
    max_len = max(len(test_f0), len(ref_f0))

    padded_test = np.pad(test_f0, (0, max_len - len(test_f0)))
    padded_ref = np.pad(ref_f0, (0, max_len - len(ref_f0)))
    
    test_cent = hz_to_cent(padded_test)
    ref_cent = hz_to_cent(padded_ref)
    
    diff_hz = np.abs(padded_test - padded_ref)
    diff_cent = np.abs(test_cent - ref_cent)
    
    voiced_mask = (padded_test > 0) & (padded_ref > 0)
    voiced_mask2 = (padded_test > 0) & (padded_ref > 0) & (diff_hz >= 0.2*padded_ref)
    insert_error = (padded_test > 80) & (padded_ref==0)
    deletion_error =  (padded_test == 80) & (padded_ref>0)

    results = {
        'hz_mean_diff': np.mean(diff_hz[voiced_mask]) if np.any(voiced_mask) else np.nan,
        'cent_mean_diff': np.mean(diff_cent[voiced_mask]) if np.any(voiced_mask) else np.nan,
        'hz_std_diff': np.std(diff_hz[voiced_mask]) if np.sum(voiced_mask) > 1 else np.nan,
        'cent_std_diff': np.std(diff_cent[voiced_mask]) if np.sum(voiced_mask) > 1 else np.nan,
        'hz_rmse': np.sqrt(np.mean(diff_hz[voiced_mask]**2)) if np.any(voiced_mask) else np.nan,
        'cent_rmse': np.sqrt(np.mean(diff_cent[voiced_mask]**2)) if np.any(voiced_mask) else np.nan,
        'corr': np.corrcoef(padded_test[voiced_mask], padded_ref[voiced_mask])[0,1] if np.sum(voiced_mask) > 1 else np.nan,
        'voiced_ratio': np.mean(voiced_mask),
        'voiced_ratio2': np.mean(voiced_mask2)/np.mean(voiced_mask),
        'insert_error': np.mean(insert_error),
        'delete_error': np.mean(deletion_error)
    }
    
    return results
    

def process_single_pair(wav_path, f0_path):
    """处理单个wav-f0文件对"""
    tracker = DPPitchTracker_48k() ###算法对应类
    test_f0 = tracker.compute_pitch(wav_path)###对应类返回基频序列f0
    ref_f0 = load_reference_f0(f0_path)
    return frame_aligned_compare(test_f0, ref_f0)

def batch_process(wav_dir, f0_dir, output_excel="results_summary.xlsx"):
    """
    批量处理文件夹中的文件
    wav_dir: wav文件目录
    f0_dir: f0文件目录
    output_excel: 结果汇总文件路径
    """
    # 获取匹配的文件对
    wav_files = sorted(Path(wav_dir).glob("*.wav"))
    f0_files = sorted(Path(f0_dir).glob("*.f0"))
    
    if len(wav_files) != len(f0_files):
        print(f"警告: wav文件数({len(wav_files)})与f0文件数({len(f0_files)})不匹配")
    
    # 处理每个文件对
    results = []
    for wav_path, f0_path in zip(wav_files, f0_files):
        print(f"正在处理: {wav_path.name} vs {f0_path.name}")
        try:
            metrics = process_single_pair(wav_path, f0_path)
            metrics['filename'] = wav_path.stem
            results.append(metrics)
        except Exception as e:
            print(f"处理{wav_path}时出错: {str(e)}")
            continue
    
    # 保存汇总结果
    if results:
        df = pd.DataFrame(results)
        # 调整列顺序
        cols = ['filename'] + [c for c in df.columns if c != 'filename']
        df = df[cols]
        df.to_excel(output_excel, index=False)
        print(f"\n处理完成，结果已保存到 {output_excel}")
        
        # 打印总体统计
        print("\n===== 总体统计 =====")
        print(f"平均音分误差: {df['cent_mean_diff'].mean():.2f} ± {df['cent_mean_diff'].std():.2f} cents")
        print(f"平均有效帧比例: {df['voiced_ratio'].mean():.2%}")
        print(f"平均误检率: {df['voiced_ratio2'].mean():.2%}")
    else:
        print("没有成功处理任何文件")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        batch_process(sys.argv[1], sys.argv[2])
    else:
        print("用法: python measure.py <wav文件夹> <f0文件夹>")
        print("示例: python measure.py MIC/M01/ REF/M01/")
        print("结果保存在results_summary.xlsx")