import numpy as np 
import cv2
import os.path as path
import sys
from scipy import ndimage as ndi

G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
       0, 0, 0], dtype=np.bool)

G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
       0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0], dtype=np.bool)

def bwmorph(image, n_iter=None):
    # check parameters
    if n_iter is None:
        n = -1
    elif n_iter <= 0:
        raise ValueError('n_iter must be > 0')
    else:
        n = n_iter
    
    # check that we have a 2d binary image, and convert it
    # to uint8
    skel = np.array(image).astype(np.uint8)
    
    if skel.ndim != 2:
        raise ValueError('2D array required')
    if not np.all(np.in1d(image.flat,(0,1))):
        raise ValueError('Image contains values other than 0 and 1')

    # neighborhood mask
    mask = np.array([[ 8,  4,  2],
                     [16,  0,  1],
                     [32, 64,128]],dtype=np.uint8)

    # iterate either 1) indefinitely or 2) up to iteration limit
    while n != 0:
        before = np.sum(skel) # count points before thinning
        
        # for each subiteration
        for lut in [G123_LUT, G123P_LUT]:
            # correlate image with neighborhood mask
            N = ndi.correlate(skel, mask, mode='constant')
            # take deletion decision from this subiteration's LUT
            D = np.take(lut, N)
            # perform deletion
            skel[D] = 0
            
        after = np.sum(skel) # coint points after thinning
        
        if before == after:
            # iteration had no effect: finish
            break
            
        # count down to iteration limit (or endlessly negative)
        n -= 1
    
    return skel.astype(np.bool)

def drd(im, im_gt):
    height, width = im.shape
    neg = np.zeros(im.shape)
    neg[im_gt!=im] = 1
    y, x = np.unravel_index(np.flatnonzero(neg), im.shape)

    n = 2
    m = n*2+1
    W = np.zeros((m,m), dtype=np.uint8)
    W[n,n] = 1.
    W = cv2.distanceTransform(1-W, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    W[n,n] = 1.
    W = 1./W
    W[n,n] = 0.
    W /= W.sum()

    nubn = 0.
    block_size = 8
    for y1 in range(0, height, block_size):
        for x1 in range(0, width, block_size):
            y2 = min(y1+block_size-1,height-1)
            x2 = min(x1+block_size-1,width-1)
            block_dim = (x2-x1+1)*(y1-y1+1)
            block = 1-im_gt[y1:y2, x1:x2]
            block_sum = np.sum(block)
            if block_sum>0 and block_sum<block_dim:
                nubn += 1

    drd_sum= 0.
    tmp = np.zeros(W.shape)
    for i in range(min(1,len(y))):
        tmp[:,:] = 0 

        x1 = max(0, x[i]-n)
        y1 = max(0, y[i]-n)
        x2 = min(width-1, x[i]+n)
        y2 = min(height-1, y[i]+n)

        yy1 = y1-y[i]+n
        yy2 = y2-y[i]+n
        xx1 = x1-x[i]+n
        xx2 = x2-x[i]+n

        tmp[yy1:yy2+1,xx1:xx2+1] = np.abs(im[y[i],x[i]]-im_gt[y1:y2+1,x1:x2+1])
        tmp *= W

        drd_sum += np.sum(tmp)
    
    if nubn != 0:
        return drd_sum/nubn
    else:
        return 1

def evaluate_metrics(im, im_gt):
    height, width = im.shape
    npixel = height*width

    im[im>0] = 1
    im_gt[im_gt>0] = 1

    sk = bwmorph(1-im_gt)
    im_sk = np.ones(im_gt.shape)
    im_sk[sk] = 0

    kernel = np.ones((3,3), dtype=np.uint8)
    im_dil = cv2.erode(im_gt, kernel)
    im_gtb = im_gt-im_dil
    im_gtbd = cv2.distanceTransform(1-im_gtb, cv2.DIST_L2, 3)

    ptp = np.zeros(im_gt.shape)
    ptp[(im==0) & (im_sk==0)] = 1
    numptp = ptp.sum()

    tp = np.zeros(im_gt.shape)
    tp[(im==0) & (im_gt==0)] = 1
    numtp = tp.sum()

    tn = np.zeros(im_gt.shape)
    tn[(im==1) & (im_gt==1)] = 1

    fp = np.zeros(im_gt.shape)
    fp[(im==0) & (im_gt==1)] = 1
    numfp = fp.sum()

    fn = np.zeros(im_gt.shape)
    fn[(im==1) & (im_gt==0)] = 1
    numfn = fn.sum()

    if numtp + numfp != 0:
        precision = numtp / (numtp + numfp)
    else:
        precision = 1

    if numtp + numfn != 0:
        recall = numtp / (numtp + numfn)
    else:
        recall = 1
    precall = numptp / np.sum(1-im_sk)
    
    if recall+precision != 0:
        fmeasure = (2*recall*precision)/(recall+precision)
    else:
        fmeasure = 1
    
    if precall+precision != 0:
        pfmeasure = (2*precall*precision)/(precall+precision)
    else:
        pfmeasure = 1

    mse = (numfp+numfn)/npixel
    if mse != 0:
        psnr = 10.*np.log10(1./mse)
    else:
        psnr = 100

    im_dn = im_gtbd.copy()
    im_dn[fn==0] = 0

    im_dp = im_gtbd.copy()
    im_dp[fp==0] = 0

    return psnr, fmeasure, pfmeasure, drd(im, im_gt)

if __name__ == "__main__":
    if len(sys.argv)<3:
        print(sys.argv[0],"input-image ground-truth-image")
        sys.exit(1)
    if not (path.exists(sys.argv[1]) and path.exists(sys.argv[2])):
        print("file not found")
        sys.exit(1)
    im = cv2.imread(sys.argv[1],0)
    im_gt = cv2.imread(sys.argv[2], 0)

    psnr, fmeasure, pfmeasure, drd = evaluate_metrics(im, im_gt)
    print("psnr", psnr, "fmeasure", fmeasure, "pfmeasure", pfmeasure, "drd", drd)