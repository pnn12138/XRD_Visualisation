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

def create_xrd_array(args, pattern):
    wavelength = WAVELENGTHS[args.wave_source]

    # takes in a pattern (in 2theta space) and converts it to an array in Q space
    peak_data = np.zeros(args.xrd_vector_dim)  # Q space
    peak_locations_2_theta = pattern.x.tolist()
    # convert 2theta to theta
    peak_locations_theta = [0.5 * theta for theta in peak_locations_2_theta]
    # convert theta to Q
    peak_locations_Q = [4 * np.pi * np.sin(np.radians(theta)) / wavelength for theta in peak_locations_theta]
    peak_values = pattern.y.tolist()

    # convert min and max theta to Q
    min_Q = 4 * np.pi * np.sin(np.radians(args.min_theta / 2)) / wavelength
    max_Q = 4 * np.pi * np.sin(np.radians(args.max_theta / 2)) / wavelength
    for i2 in range(len(peak_locations_Q)):
        q = peak_locations_Q[i2]
        height = peak_values[i2] / 100
        scaled_location = int(args.xrd_vector_dim * (q - min_Q) / (max_Q - min_Q))
        peak_data[scaled_location] = max(peak_data[scaled_location], height)  # just in case really close

    return peak_data

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
            xrd_array = create_xrd_array(args, pattern)
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
        origin_df.to_csv(str(file)+'_Q_tsne_visualization.csv')
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
        default = 'C:\code\XRD_Visualisation\data\mp_20_xrd',
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