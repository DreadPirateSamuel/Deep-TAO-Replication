import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress INFO and WARNING logs
import numpy as np
import pandas as pd
import fitsio
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
import glob
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Z-scale algorithm from utils.py (for visualization)
MAX_REJECT = 0.5
MIN_NPIXELS = 5
GOOD_PIXEL = 0
BAD_PIXEL = 1
KREJ = 2.5
MAX_ITERATIONS = 5

def zsc_sample(image, maxpix, bpmask=None, zmask=None):
    nc = image.shape[0]
    nl = image.shape[1]
    stride = max(1.0, np.sqrt((nc - 1) * (nl - 1) / float(maxpix)))
    stride = int(stride)
    samples = image[::stride, ::stride].flatten()
    if isinstance(samples, np.ma.MaskedArray):
        samples = samples.compressed()
    elif isinstance(samples, np.ndarray):
        samples = samples[np.isfinite(samples)]
    return samples[:maxpix]

def zsc_compute_sigma(flat, badpix, npix):
    goodpixels = np.where(badpix == GOOD_PIXEL)
    sumz = flat[goodpixels].sum()
    sumsq = (flat[goodpixels] * flat[goodpixels]).sum()
    ngoodpix = len(goodpixels[0])
    if ngoodpix == 0:
        mean = None
        sigma = None
    elif ngoodpix == 1:
        mean = sumz
        sigma = None
    else:
        mean = sumz / ngoodpix
        temp = sumsq / (ngoodpix - 1) - sumz * sumz / (ngoodpix * (ngoodpix - 1))
        sigma = np.sqrt(temp) if temp > 0 else 0.0
    return ngoodpix, mean, sigma

def zsc_fit_line(samples, npix, krej, ngrow, maxiter):
    xscale = 2.0 / (npix - 1)
    xnorm = np.arange(npix) * xscale - 1.0
    ngoodpix = npix
    minpix = max(MIN_NPIXELS, int(npix * MAX_REJECT))
    last_ngoodpix = npix + 1
    badpix = np.zeros(npix, dtype="int32")
    for niter in range(maxiter):
        if ngoodpix >= last_ngoodpix or ngoodpix < minpix:
            break
        goodpixels = np.where(badpix == GOOD_PIXEL)
        sumx = xnorm[goodpixels].sum()
        sumxx = (xnorm[goodpixels] * xnorm[goodpixels]).sum()
        sumxy = (xnorm[goodpixels] * samples[goodpixels]).sum()
        sumy = samples[goodpixels].sum()
        sum = len(goodpixels[0])
        delta = sum * sumxx - sumx * sumx
        intercept = (sumxx * sumy - sumx * sumxy) / delta
        slope = (sum * sumxy - sumx * sumy) / delta
        fitted = xnorm * slope + intercept
        flat = samples - fitted
        ngoodpix, mean, sigma = zsc_compute_sigma(flat, badpix, npix)
        threshold = sigma * krej
        badpix[np.where(flat < -threshold)] = BAD_PIXEL
        badpix[np.where(flat > threshold)] = BAD_PIXEL
        kernel = np.ones(ngrow, dtype="int32")
        badpix = np.convolve(badpix, kernel, mode='same')
        ngoodpix = len(np.where(badpix == GOOD_PIXEL)[0])
    zstart = intercept - slope
    zslope = slope * xscale
    return ngoodpix, zstart, zslope

def zscale(image, nsamples=1000, contrast=0.25, bpmask=None, zmask=None):
    samples = zsc_sample(image, nsamples, bpmask, zmask)
    npix = len(samples)
    samples.sort()
    zmin = samples[0]
    zmax = samples[-1]
    center_pixel = (npix - 1) // 2
    median = samples[center_pixel] if npix % 2 == 1 else 0.5 * (samples[center_pixel] + samples[center_pixel + 1])
    minpix = max(MIN_NPIXELS, int(npix * MAX_REJECT))
    ngrow = max(1, int(npix * 0.01))
    ngoodpix, zstart, zslope = zsc_fit_line(samples, npix, KREJ, ngrow, MAX_ITERATIONS)
    if ngoodpix < minpix:
        z1 = zmin
        z2 = zmax
    else:
        if contrast > 0:
            zslope = zslope / contrast
        z1 = max(zmin, median - (center_pixel - 1) * zslope)
        z2 = min(zmax, median + (npix - center_pixel) * zslope)
    return z1, z2

def load_fits_data(fits_path, max_images=10):
    try:
        data, header = fitsio.read(fits_path, header=True)
        images = []
        mjds = np.array(data['MJD'], dtype=np.float64)  # Convert to float
        exts = np.arange(2, min(header['N_Images'] + 2, max_images + 2))
        for ext in exts:
            img = fitsio.read(fits_path, ext=ext)
            # Normalize image to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-10)
            images.append(img)
        return images, mjds[:len(exts)-2], header
    except Exception as e:
        print(f"Error loading {fits_path}: {e}")
        return None, None, None

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def replicate_deep_tao(transients_dir, non_transients_dir, output_dir, epochs=10, batch_size=16, subset_size=2000):
    """
    Replicate DEEP-TAO study by training and evaluating a CNN on transient and non-transient datasets.
    
    Parameters:
    -----------
    transients_dir : str
        Path to directory containing transient FITS files (e.g., TAO_transients-master/data/).
    non_transients_dir : str
        Path to directory containing non-transient FITS files (e.g., TAO_non-transients-master/data/NON/).
    output_dir : str
        Directory to save model and results.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    subset_size : int
        Number of FITS files to use per class (for computational constraints).
    
    Returns:
    --------
    None
        Saves model, results, and visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define classes and check for SN
    classes = ['AGN', 'BZ', 'CV', 'OTHER', 'NON-TRANSIENT']
    if os.path.exists(os.path.join(transients_dir, 'SN')):
        classes.insert(0, 'SN')
    else:
        print("Warning: SN class not found in transients_dir. DEEP-TAO paper includes SN.")
    
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    #Load dataset with balancing
    X, y = [], []
    for cls in classes[:-1]:  # Transient classes
        fits_files = glob.glob(f"{transients_dir}/{cls}/*.fits")
        if not fits_files:
            print(f"No files found for class {cls} in {transients_dir}/{cls}/")
            continue
        np.random.shuffle(fits_files)
        fits_files = fits_files[:subset_size//num_classes]
        for fits_path in fits_files:
            images, mjds, header = load_fits_data(fits_path)
            if images:
                X.extend(images)
                y.extend([class_to_idx[cls]] * len(images))
    
    #Non-transient class
    fits_files = glob.glob(f"{non_transients_dir}/*.fits")
    if not fits_files:
        print(f"No files found for NON-TRANSIENT in {non_transients_dir}/")
    else:
        np.random.shuffle(fits_files)
        fits_files = fits_files[:subset_size//num_classes]
        for fits_path in fits_files:
            images, mjds, header = load_fits_data(fits_path)
            if images:
                X.extend(images)
                y.extend([class_to_idx['NON-TRANSIENT']] * len(images))
    
    if not X:
        print("No data loaded. Check dataset paths and FITS files.")
        return
    
    #Verify image dimensions
    shapes = [img.shape for img in X]
    if len(set(shapes)) > 1:
        print(f"Warning: Inconsistent image dimensions: {set(shapes)}. Resizing may be needed.")
    
    #Preprocessig data
    X = np.array(X)[..., np.newaxis]  # Add channel dimension
    y = tf.keras.utils.to_categorical(y, num_classes)
    
    #Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #Building and training the model
    input_shape = X_train.shape[1:]
    model = build_cnn_model(input_shape, num_classes)
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)]
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), callbacks=callbacks, verbose=1)
    
    #Evaluating model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    f1_scores = f1_score(y_test_classes, y_pred_classes, average=None)
    
    #Saving results :)
    results = pd.DataFrame({
        'Class': classes,
        'F1_Score': f1_scores
    })
    results.to_csv(f"{output_dir}/f1_scores.csv", index=False)
    print("F1 Scores:\n", results)
    
    #Plotting matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
    
    #Visualize sample images
    sample_fits = []
    for cls in classes[:-1]:
        fits_files = glob.glob(f"{transients_dir}/{cls}/*.fits")
        if fits_files:
            sample_fits.append(fits_files[0])
    fits_files = glob.glob(f"{non_transients_dir}/*.fits")
    if fits_files:
        sample_fits.append(fits_files[0])
    
    if sample_fits:
        fig = plt.figure(figsize=(15, 5))
        for i, fits_path in enumerate(sample_fits):
            images, mjds, header = load_fits_data(fits_path, max_images=1)
            if images:
                plt.subplot(1, len(sample_fits), i+1)
                zmin, zmax = zscale(images[0])
                plt.imshow(images[0], vmin=zmin, vmax=zmax, cmap='gray')
                mjd = float(mjds[0]) if len(mjds) > 0 else 0.0
                plt.title(f"{header.get('CRTS_ID', 'Unknown')}\nMJD: {mjd:.2f}")
                plt.axis('off')
        plt.suptitle('Sample Transient and Non-Transient Images')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sample_images.png")
        plt.close()
    
    #Saving model
    model.save(f"{output_dir}/deep_tao_model.keras")
    
    #Used to compare with paper
    paper_f1_avg = 0.5458  # From TAO-Net in related work
    print(f"Paper's Average F1-Score: {paper_f1_avg:.4f}")
    print(f"Your Average F1-Score: {np.mean(f1_scores):.4f}")

if __name__ == "__main__":
    # Paths relative to my local /home/sam06/CAP4613PRA/
    TRANSIENTS_DIR = "TAO_transients-master/data"
    NON_TRANSIENTS_DIR = "TAO_non-transients-master/data/NON"
    OUTPUT_DIR = "my_deep_tao_results"
    replicate_deep_tao(TRANSIENTS_DIR, NON_TRANSIENTS_DIR, OUTPUT_DIR, epochs=10, batch_size=16, subset_size=2000)