import argparse
import io
import os
import zipfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict
import sys
import importlib

# ------------------- Import diagnostics -------------------

_import_errors = {}

def _try_import(module_name: str):
    try:
        return importlib.import_module(module_name), None
    except Exception as e:
        return None, str(e)

# Core libraries
np, err = _try_import('numpy')
if err: _import_errors['numpy'] = err
pd, err = _try_import('pandas')
if err: _import_errors['pandas'] = err
cv2, err = _try_import('cv2')
if err: _import_errors['opencv-python (cv2)'] = err

# scikit-image
skimage_mod, err = _try_import('skimage')
if err:
    _import_errors['scikit-image'] = err
else:
    try:
        from skimage import color, exposure, filters, util
        from skimage.measure import shannon_entropy
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
    except Exception as e:
        _import_errors['scikit-image'] = str(e)

# matplotlib
mpl, err = _try_import('matplotlib')
if err:
    _import_errors['matplotlib'] = err
else:
    try:
        mpl.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        _import_errors['matplotlib'] = str(e)

# Fail if essentials are missing
essential = {'numpy', 'opencv-python (cv2)', 'scikit-image'}
failed_essential = [k for k in _import_errors.keys() if k in essential]
if failed_essential:
    print('Essential packages failed to import. Please install them:')
    print('python -m pip install numpy pandas opencv-python scikit-image matplotlib pillow')
    sys.exit(1)

# ----------------------------- Modules -----------------------------

class DatasetLoader:
    IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

    def extract_images(self) -> List[Tuple[str, np.ndarray]]:
        images = []
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.lower().endswith(self.IMAGE_EXT):
                    continue
                try:
                    data = z.read(info)
                    img_array = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        from PIL import Image
                        img = np.array(Image.open(io.BytesIO(data)).convert('RGB'))
                    if img.ndim == 3:
                        if img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = img.astype(np.float32)
                    if img.max() > 1.1:
                        img /= 255.0
                    images.append((info.filename, img))
                except Exception as e:
                    print(f"Warning: Failed to read {info.filename}: {e}")
        return images

class ColorizationModule:
    TARGET_SHAPE = (512, 512)

    @staticmethod
    def _standardize(img_gray: np.ndarray) -> np.ndarray:
        out = cv2.resize((img_gray * 255).astype(np.uint8), ColorizationModule.TARGET_SHAPE)
        return out.astype(np.float32) / 255.0

    @staticmethod
    def _to_rgb(img_gray: np.ndarray) -> np.ndarray:
        return np.stack([img_gray, img_gray, img_gray], axis=-1)

    def clahe(self, img_gray: np.ndarray):
        img = self._standardize(img_gray)
        c = cv2.createCLAHE(2.0, (8,8)).apply((img*255).astype(np.uint8))
        return self._to_rgb(c.astype(np.float32)/255.0)

    def heatmap(self, img_gray: np.ndarray, cmap='jet'):
        img = self._standardize(img_gray)
        return plt.get_cmap(cmap)(img)[:, :, :3].astype(np.float32)

    def lut_color(self, img_gray: np.ndarray, colormap=cv2.COLORMAP_AUTUMN):
        img = self._standardize(img_gray)
        c = cv2.applyColorMap((img*255).astype(np.uint8), colormap)
        return cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

    def gamma_correction(self, img_gray: np.ndarray):
        return self._to_rgb(exposure.adjust_gamma(self._standardize(img_gray), 1.2))

    def edge_enhanced(self, img_gray: np.ndarray):
        img = self._standardize(img_gray)
        edges = cv2.Canny((img*255).astype(np.uint8), 50, 150).astype(np.float32)/255.0
        base = self._to_rgb(img)
        base[:, :, 0] += edges
        return np.clip(base, 0, 1)

class EvaluationModule:
    @staticmethod
    def _gray(img):
        if img.ndim == 3: return color.rgb2gray(img).astype(np.float32)
        return img.astype(np.float32)

    def compute_metrics(self, orig, proc):
        og, pr = self._gray(orig), self._gray(proc)
        diff = np.abs(og - pr)
        return {
            'ssim': ssim(og, pr, data_range=1.0),
            'psnr': psnr(og, pr, data_range=1.0),
            'entropy': shannon_entropy((pr*255).astype(np.uint8)),
            'mean_diff': float(diff.mean()),
            'diff_map': diff
        }

# -------------------------- Execution logic --------------------------

def process_single_image(name: str, img_gray: np.ndarray, output_dir: str) -> List[Dict]:
    cm, em = ColorizationModule(), EvaluationModule()
    orig_std = cm._standardize(img_gray)
    orig_rgb = cm._to_rgb(orig_std)

    methods = {
        'CLAHE': cm.clahe(img_gray),
        'Heatmap_Jet': cm.heatmap(img_gray, 'jet'),
        'LUT_Autumn': cm.lut_color(img_gray),
        'Gamma': cm.gamma_correction(img_gray),
        'EdgeEnhanced': cm.edge_enhanced(img_gray),
    }

    # VISUALIZATION LOGIC
    safe_name = name.replace('/', '_').replace('\\', '_')
    base_name = os.path.splitext(safe_name)[0]
    vis_path = os.path.join(output_dir, f"{base_name}_grid.png")
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
    axes[0].imshow(orig_rgb); axes[0].set_title('Original')
    axes[1].imshow(methods['CLAHE']); axes[1].set_title('CLAHE')
    axes[2].imshow(methods['Heatmap_Jet']); axes[2].set_title('Heatmap')
    axes[3].imshow(methods['EdgeEnhanced']); axes[3].set_title('Edges')
    
    # Diff Map for Heatmap
    m_eval = em.compute_metrics(orig_rgb, methods['Heatmap_Jet'])
    axes[4].imshow(m_eval['diff_map'], cmap='hot'); axes[4].set_title('Diff (Heatmap)')
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    fig.savefig(vis_path)
    plt.close(fig)

    res_list = []
    for mname, mimg in methods.items():
        metrics = em.compute_metrics(orig_rgb, mimg)
        res_list.append({
            'image': name,
            'method': mname,
            'ssim': metrics['ssim'],
            'psnr': metrics['psnr'],
            'entropy': metrics['entropy'],
            'mean_diff': metrics['mean_diff']
        })
    return res_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', dest='zip', required=False)
    parser.add_argument('--output', default='outputs')
    args = parser.parse_args()

    zip_path = args.zip if args.zip else input('Enter zip path: ').strip()
    os.makedirs(args.output, exist_ok=True)

    images = DatasetLoader(zip_path).extract_images()
    if not images:
        print("No images found."); return

    all_res = []
    print(f"Processing {len(images)} images and generating visual grids...")
    for name, img in images:
        all_res.extend(process_single_image(name, img, args.output))

    df = pd.DataFrame(all_res)

    # Scoring logic
    def compute_score(row):
        return (0.5 * row['ssim'] + 0.3 * (row['psnr'] / 50.0) - 0.2 * row['mean_diff'])

    df['score'] = df.apply(compute_score, axis=1)

    # Ranking and CSV Generation
    df.to_csv(os.path.join(args.output, 'results.csv'), index=False)
    
    ranked = df.sort_values(['image', 'score'], ascending=[True, False])
    ranked.to_csv(os.path.join(args.output, 'ranked_results.csv'), index=False)

    best_method_per_image = ranked.groupby('image').first().reset_index()
    best_method_per_image.to_csv(os.path.join(args.output, 'best_method_per_image.csv'), index=False)

    summary = df.groupby('method').mean(numeric_only=True).reset_index()
    summary.to_csv(os.path.join(args.output, 'summary_stats.csv'), index=False)

    # Safety Flags
    df['risk_flag'] = (df['entropy'] > df['entropy'].mean()) & (df['ssim'] < 0.85)
    df[df['risk_flag']].to_csv(os.path.join(args.output, 'potentially_misleading_cases.csv'), index=False)

    print(f"✅ Complete. {len(images)} images processed. Grids and CSVs saved in '{args.output}'.")

if __name__ == '__main__':
    main()