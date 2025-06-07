import argparse,os
import pandas as pd
import numpy as np
from pymatgen.analysis.diffraction.xrd import WAVELENGTHS
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

def augment_xrdStrip(curr_xrdStrip,sinc_filt, return_both=False, do_not_sinc_gt_xrd=False):
    """
    Input:
    -> curr_xrdStrip: XRD pattern of shape (self.n_presubsample,)
    -> return_both: if True, return (bothFiltered, rawSincFiltered), only valid if self.xrd_filter == 'both';
        if False, return based on self.xrd_filter
    Output:
    -> if return_both=False,
        returns curr_xrdStrip augmented by peak broadening (sinc and/or gaussian) & vertical Gaussian perturbations;
        with shape (self.n_postsubsample,); in range [0, 1]
    -> if return_both=True,
        returns (bothFiltered, rawSincFiltered); where bothFiltered has both sinc filter & gaussian filter,
        rawSincFiltered has only sinc filter
    """

    xrd = curr_xrdStrip.numpy()


    if do_not_sinc_gt_xrd:  # it comes from experimental data, which is already broadened!
        sinc_filtered = xrd
    else:  # this is synthetic data: need to broaden it
        sinc_filtered = sinc_filter(xrd,sinc_filt)
    filtered = gaussian_filter(sinc_filtered)
    sinc_only_presubsample =sinc_filtered
    # presubsamples
    filtered_presubsample = filtered

    # postsubsampling
    filtered_postsubsampled = post_process_filtered_xrd(filtered)

    # postsubsampling
    sinc_only_postsubsample = post_process_filtered_xrd(sinc_filtered)

    return filtered_postsubsampled, sinc_only_postsubsample, filtered_presubsample, sinc_only_presubsample

def post_process_filtered_xrd(filtered):
    # scale
    filtered = filtered / np.max(filtered)
    filtered = np.maximum(filtered, np.zeros_like(filtered))
    # sample it
    assert filtered.shape == (4096,)
    filtered = sample(filtered)
    return filtered
def sample(x):
    step_size = int(np.ceil(len(x) / 512))
    x_subsample = [np.max(x[i:i + step_size]) for i in range(0, len(x), step_size)]
    return np.array(x_subsample)

def sinc_filter(x,sinc_filt):
    filtered = np.convolve(x, sinc_filt, mode='same')
    return filtered

def gaussian_filter(x):
    filtered = gaussian_filter1d(x,
                sigma=np.random.uniform(
                    low=4096 * (1e-2, 1.1e-2)[0],
                    high=4096 * (1e-2, 1.1e-2)[1]
                ),
                mode='constant', cval=0)
    return filtered
def process_xrd(args):

    data_dir = args.data_dir
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.csv')]
    print(f'待处理文件: {len(files)} files')
    #初始化Q空间
    nanomaterial_size = 50
    n_presubsample = 4096
    n_postsubsample = 512
    wavesource = 'CuKa'
    wavelength = WAVELENGTHS[wavesource]
    min_theta = 0 / 2
    max_theta = 180 / 2
    Q_min = 4 * np.pi * np.sin(np.radians(min_theta)) / wavelength
    Q_max = 4 * np.pi * np.sin(np.radians(max_theta)) / wavelength
    # phase shift for sinc filter = half of the signed Q range
    phase_shift = (Q_max - Q_min) / 2
    # compute Qs
    Qs = np.linspace(Q_min, Q_max,n_presubsample)
    Qs_shifted = Qs - phase_shift
    plt.figure(figsize=(9,6))
    sinc_filt = nanomaterial_size * (np.sinc(nanomaterial_size * Qs_shifted / np.pi) ** 2)

    plt.plot(sinc_filt)
    plt.show()
    horizontal_noise_range = (1e-2, 1.1e-2),  # (1e-3, 1.1e-3)
    vertical_noise = 1e-3,

    for file in files:
        df = pd.read_pickle(os.path.join(data_dir,file))
        xrd_df=df["xrd"]
        for curr_xrd in tqdm(xrd_df):
            """list_data = curr_xrd.strip("[]").split()

            # 转换为浮点数列表
            float_list = [float(item) for item in list_data]

            # 转换为 numpy 数组
            curr_xrd = np.array(float_list)"""
            print(curr_xrd)
            curr_xrd = curr_xrd.reshape((n_presubsample,))
            df['rawXRD'] = sample(curr_xrd.numpy()) # need to downsample first
            # have sinc with gaussian filter & sinc w/out gaussian filter
            curr_xrd, sinc_only_xrd, curr_xrd_presubsample, sinc_only_xrd_presubsample = augment_xrdStrip(curr_xrd,sinc_filter,return_both=True, do_not_sinc_gt_xrd=False)
            curr_xrd["xrdd"] = curr_xrd
            curr_xrd['sincOnly'] = sinc_only_xrd
            curr_xrd['sincOnlyPresubsample'] = sinc_only_xrd_presubsample
            curr_xrd['xrdPresubsample'] = curr_xrd_presubsample






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process XRD patterns with SINC and gau')
    parser.add_argument(
        '--data_dir',
        default='C:\code\XRD_Visualisation\data\mp_20_xrd',
        type=str,
        help='path to input CIF files'
    )
    parser.add_argument(
        '--save_dir',
        default='C:\code\XRD_Visualisation\data\mp_20_xrd_sinc_gau',
        type=str,
        help='path to input CIF files'
    )
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    process_xrd(args)