import argparse,os
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.diffraction.xrd import XRDCalculator, WAVELENGTHS
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def broaden_xrd_peaks_fixed_length(peak_x, peak_y, total_length=18000, fwhm=0.2, normalize=True):
    """
    将离散 XRD 峰拓宽为固定长度的连续曲线（如18000）

    参数：
        peak_x: 峰位置（单位：2θ，单位：度）
        peak_y: 峰强度
        total_length: 输出向量长度（如18000）
        fwhm: 峰拓宽的半高宽（单位：度）
        normalize: 是否归一化强度

    返回：
        x: 二θ角数组（shape: [total_length]）
        y: 强度数组（shape: [total_length]）
    """
    x = np.linspace(0, 180, total_length)
    y = np.zeros_like(x)

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # 将FWHM转换为标准差

    for px, py in zip(peak_x, peak_y):
        y += py * np.exp(-(x - px) ** 2 / (2 * sigma ** 2))

    if normalize and y.max() > 0:
        y /= y.max()

    return x, y

def cif2xrd(args):
    data_dir = args.data_dir
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.csv')]
    print(f'待处理文件: {len(files)} files')
    for file in files:
        print(f'处理 {file}')
        origin_df = pd.read_csv(os.path.join(data_dir, file))
        cifs = origin_df['cif'].values
        xrd_array_list=[]
        lattice_type_list=[]
        for cif in tqdm(cifs, desc=f'Generating XRDs for file {file}'):
            parser = CifParser.from_str(cif)
            structure = parser.parse_structures(primitive=True)[0]

            # important to use the conventional structure to ensure
            # that peaks are labelled with the conventional Miller indices
            sga = SpacegroupAnalyzer(structure)
            structure = sga.get_conventional_standard_structure()


            # 获取晶格类型
            lattice_type = sga.get_lattice_type()  # 晶格类型

            # wavelength
            curr_wavelength = WAVELENGTHS[args.wave_source]
            # Create the XRD calculator
            xrd_calc = XRDCalculator(wavelength=curr_wavelength)
            # Calculate the XRD pattern
            pattern = xrd_calc.get_pattern(structure)
            # Create the XRD tensor
            _,xrd_array = broaden_xrd_peaks_fixed_length(pattern.x.tolist(),pattern.y.tolist())
            xrd_array_list.append(xrd_array)
            lattice_type_list.append(lattice_type)
        origin_df['xrd'] = xrd_array_list
        origin_df['lattice_type']=lattice_type_list
        xrd_data = np.vstack(origin_df['xrd'].values)
        tsne = TSNE(n_components=2, random_state=0)
        xrd_tsne = tsne.fit_transform(xrd_data)
        origin_df['xrd_tsne_1'] = xrd_tsne[:, 0]
        origin_df['xrd_tsne_2'] = xrd_tsne[:, 1]
        plt.figure(figsize=(8, 6))
        plt.scatter(origin_df['xrd_tsne_1'], origin_df['xrd_tsne_2'], alpha=0.7)
        plt.title('t-SNE Visualization of XRD Data')
        plt.xlabel('Q-t-SNE Component 1')
        plt.ylabel('Q-t-SNE Component 2')
        plt.grid(True)

        output_path = str(file)+'tsne_visualization.png'  # 你可以根据需要修改路径和文件名
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        origin_df.to_pickle(str(file)+'_Q_tsne_visualization.csv')
        plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate XRD patterns from CIF descriptions')
    parser.add_argument(
        '--data_dir',
        default = 'C:\code\XRD_Visualisation\data\mp_20',
        type=str,
        help='path to input CIF files'
    )
    parser.add_argument(
        '--save_dir',
        default = 'C:\code\XRD_Visualisation\data\mp_20_xrd_gau',
        type=str,
        help='path to save XRD patterns'
    )
    parser.add_argument(
        '--max_theta',
        default=180,
        type=int,
    )
    parser.add_argument(
        '--min_theta',
        default=0,
        type=int,
    )
    parser.add_argument(
        '--wave_source',
        type=str,
        default='CuKa',
        help='What is the wave source?'
    )
    parser.add_argument(
        '--xrd_vector_dim',
        default=4096,
        type=int,
    )
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    cif2xrd(args)