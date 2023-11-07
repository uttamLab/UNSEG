"""
UNSEG is an algorithm for UNsupervised SEGmentation of nuclei and cells in
fluorescent images of tissues containing both nucleus and cell membrane markers.

@author: Uttam Lab
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.filters import rank, threshold_multiotsu, gaussian, sobel
from skimage.segmentation import mark_boundaries, watershed, find_boundaries
from skimage.morphology import erosion, dilation, disk, thin
from skimage.util import img_as_ubyte
from scipy import ndimage
import time
import cv2
from sklearn.cluster import KMeans

###############################################################################
### AUXILIARY FUNCTIONS
###############################################################################
def plot_img(img, tlt='', cmp='gray'):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,15))
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=None, hspace=None)
    ax.imshow(img, cmap=cmp)
    ax.set_title(tlt)
    ax.axis('on')
    plt.show()

def scale_img(img, norm='0:1'):
    img = img.astype('float64')
    i_min = np.min(img)
    i_max = np.max(img)
    if i_max-i_min > 0:
        if norm == '-1:1':
            img = 2*(img-i_min)/(i_max-i_min)-1
        else:
            img = (img-i_min)/(i_max-i_min)
    return img

def compute_conv(a):
    h, w = a.shape
    conv = 0
    
    c = np.roll(a, (1,0), axis=(1,0))
    c[:,0] = 0
    conv = conv + c
    
    c = np.roll(a, (1,-1), axis=(1,0))
    c[:,0] = 0
    c[h-1,:] = 0
    conv = conv + c
    
    c = np.roll(a, (0,-1), axis=(1,0))
    c[h-1,:] = 0
    conv = conv + c
    
    c = np.roll(a, (-1,-1), axis=(1,0))
    c[:,w-1] = 0
    c[h-1,:] = 0
    conv = conv + c
    
    c = np.roll(a, (-1,0), axis=(1,0))
    c[:,w-1] = 0
    conv = conv + c
    
    c = np.roll(a, (-1,1), axis=(1,0))
    c[:,w-1] = 0
    c[0,:] = 0
    conv = conv + c
    
    c = np.roll(a, (0,1), axis=(1,0))
    c[0,:] = 0
    conv = conv + c
    
    c = np.roll(a, (1,1), axis=(1,0))
    c[:,0] = 0
    c[0,:] = 0
    conv = conv + c
    
    conv = conv + a
    return conv

def gradient_adaptive_smoothing(img, k=1):
    """Gradient Adaptive Smoothing"""
    gas = scale_img(img)
    u = sobel(gas)
    u = np.exp(-(u*u)/(2*k**2))
    gas = u*gas
    gas = compute_conv(gas)
    u = compute_conv(u)
    gas = gas / u
    return gas

def affv(img, k=5, t=0.5):
    """Local Mean Suppression Filter"""
    h = img.shape[0]
    w = img.shape[1]
    ker = np.ones((2*k+1,), dtype='int32')
    row = np.ones((1,w), dtype='int32')
    img_avr = np.zeros((h,w), dtype='float64')
    norm = np.zeros((h,w), dtype='int32')
    for i in range(h):
        img_avr[i,:] = np.convolve(img[i,:], ker, mode='same')
        norm[i,:] = np.convolve(row[0,:], ker, mode='same')
    for j in range(w):
        img_avr[:,j] = np.convolve(img_avr[:,j], ker, mode='same')
        norm[:,j] = np.convolve(norm[:,j], ker, mode='same')
    img_avr = np.divide(img_avr, norm)
    norm = img < t*img_avr
    img_avr = img.copy()
    img_avr[norm] = 0
    return img_avr

def distribution_function(img, x, f, dn, mt='fg'):
    """Distribution of Probabilities"""
    if mt == 'fg':
        vs = 0
        n = dn+1
    elif mt == 'bg':
        vs = 1
        n = np.max(x)+1
    pb = np.zeros((img.shape[0],img.shape[1]), dtype='float64')
    for i in range(n):
        m = img == i
        inx = np.argwhere(x == i)
        if len(inx) == 1:
            inx = inx[0,0]
            pb[m] = f[inx]
        elif len(inx) == 0:
            inx_m = np.ma.masked_less(x, i, copy=True)
            if np.ma.count_masked(inx_m) == 0:
                inx = np.nan
                val = vs
            else:
                inx = np.ma.argmin(inx_m)
                inx = inx - 1
                val = f[inx]
            pb[m] = val
    return pb

def find_contour(cc, num='one'):
    """Outer Contour of Connected Component"""
    contours, _  = cv2.findContours(cc, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    num_contours = len(contours)
    if num_contours == 1:
        ind = 0
    else:
        nc, ind = 0, 0
        for i in range(num_contours):
            if len(contours[i][:,0,0]) > nc:
                nc, ind = len(contours[i][:,0,0]), i
    max_contour = contours[ind]
    if num == 'one':
        return max_contour
    else:
        contours = list(contours)
        contours.pop(ind)
        return max_contour, contours

def remove_selfintersection(cc):
    cc = erosion(cc, disk(2))
    cc = dilation(cc, disk(2))
    cc = ndimage.binary_fill_holes(cc)
    return cc

def test_selfintersection(cc, contour='no'):
    cnt = find_contour((cc).astype('uint8'))
    hull = cv2.convexHull(cnt, returnPoints=False)
    try:
        defects = cv2.convexityDefects(cnt, hull)
        test_res = True
    except:
        defects = 0
        test_res = False
    if contour == 'no':
        return test_res, defects
    else:
        cnt = cnt[:,0,:]
        return test_res, defects, cnt

def add_layer(cc, b=1):
    h, w = cc.shape
    ccb = np.zeros((h+2*b, w+2*b), dtype='bool')
    ccb[b:h+b, b:w+b] = cc
    return ccb
  
def remove_layer(ccb, b=1):
    h, w = ccb.shape
    cc = np.zeros((h-2*b, w-2*b), dtype='float64')
    cc = ccb[b:h-b, b:w-b]
    return cc

def get_component_indices(arr):
    n = np.unique(arr)
    n = list(n)
    if 0 in n:
        n.remove(0)
    return n

def dynamic_ws(gbm0, lbm0):
    """Perturbation Analysis of Seed Points for Watershed Segmentation"""
    m_thr = 0
    f_thr = 25
    a_g = np.count_nonzero(gbm0)
    a_l = np.count_nonzero(lbm0)
    if a_g == 0 or a_l == 0:
        return np.zeros((gbm0.shape), dtype='int32'), 100
    da = 100*(a_g-a_l)/a_g
    dt0 = ndimage.distance_transform_edt(add_layer(lbm0))
    dt0 = remove_layer(dt0)
    dt0_avr = np.mean(dt0[dt0>0])
    dt0_avr_f = np.floor(dt0_avr)
    dt0_mask = dt0 > dt0_avr
    prop0 = label(dt0_mask, background=False, connectivity=2)
    prop0 = regionprops(prop0)
    n_sp = 0
    for i in range(len(prop0)):
       if prop0[i].area > m_thr:
           n_sp = n_sp + 1
    sp = np.zeros((n_sp,2), dtype='int64')
    j = 0
    for i in range(len(prop0)):
        if prop0[i].area > m_thr:
            pp0 = prop0[i]
            hh1, ww1, hh2, ww2 = pp0.bbox
            temp0 = np.zeros((hh2-hh1,ww2-ww1), dtype='float64')
            temp0[pp0.image] = dt0[hh1:hh2, ww1:ww2][pp0.image]
            yy, xx = np.unravel_index(np.argmax(temp0, axis=None), temp0.shape)
            yy, xx = yy + hh1, xx + ww1
            sp[j,:] = (yy, xx)
            j = j + 1
    shifts = ((0,0), (-1,0), (1,0), (0,-1), (0,1))
    areas = np.zeros((n_sp, 5), dtype='int64')
    for j in range(5):
        dx = int(shifts[j][0] * dt0_avr_f)
        dy = int(shifts[j][1] * dt0_avr_f)
        markers = np.zeros(dt0.shape, dtype='int32')
        for i in range(sp.shape[0]):
            markers[sp[i,0]+dy, sp[i,1]+dx] = i + 1
        seg0 = watershed(-dt0, markers, mask=gbm0)
        for i in range(n_sp):
            areas[i,j] = np.count_nonzero(seg0 == i+1)
        if j == 0:
            seg_0 = seg0.copy()
    areas_n = areas[:,1:].copy()
    for i in range(areas.shape[0]):
        areas_n[i,:] = areas_n[i,:] - areas[i,0]
    dyn_channel_namesange = np.zeros((areas.shape[0],5), dtype='int32')
    dyn_channel_namesange[:,0] = np.count_nonzero(areas_n > 0, axis=1)
    dyn_channel_namesange[:,1] = np.count_nonzero(areas_n == 0, axis=1)
    dyn_channel_namesange[:,2] = np.count_nonzero(areas_n < 0, axis=1)
    dyn_channel_namesange[:,3] = np.count_nonzero(areas[:,1:] < f_thr , axis=1)
    dyn_channel_namesange[:,4] = dyn_channel_namesange[:,0] + dyn_channel_namesange[:,1] - dyn_channel_namesange[:,2]
    sp_fxd = dyn_channel_namesange[:,1] == 4
    sp_uns = np.logical_and(dyn_channel_namesange[:,3]>0, dyn_channel_namesange[:,4]<=0)
    sp_stb = np.logical_not(np.logical_or(sp_fxd, sp_uns))
    seg_final = np.zeros((gbm0.shape), dtype='int32')
    n_cc = 0
    mask = np.zeros(gbm0.shape, dtype='bool')
    for i in range(sp_fxd.shape[0]):
        if sp_fxd[i]:
            seg_final[seg_0 == i+1] = n_cc+1
            n_cc = n_cc + 1
        else:
            mask[seg_0==i+1] = True 
    markers = np.zeros(gbm0.shape, dtype='int32')
    j = n_cc + 1
    for i in np.where(sp_stb == True)[0]:
        markers[sp[i,0], sp[i,1]] = j
        j = j + 1
    prop_mask = label(mask, background=False, connectivity=2)
    prop_mask = regionprops(prop_mask)
    sp_add = []
    for p in prop_mask:
        h1, w1, h2, w2 = p.bbox
        if np.count_nonzero(markers[h1:h2,w1:w2][p.image] != 0) == 0:
            temp0 = np.zeros((h2-h1,w2-w1), dtype='float64')
            temp0[p.image] = dt0[h1:h2, w1:w2][p.image]
            yy, xx = np.unravel_index(np.argmax(temp0, axis=None), temp0.shape)
            yy, xx = yy + h1, xx + w1
            sp_add.append((yy,xx))
            markers[yy, xx] = j
            j = j + 1
    sp_add = np.array(sp_add, dtype='int64')
    sp_add = sp_add.reshape(-1,2)
    seg0 = watershed(-dt0, markers, mask=mask)
    inds = get_component_indices(seg0)
    for i in inds:
        seg_final[seg0 == i] = i
    return seg_final, da

def boundary_cuts(seg, mask):
    bnds = find_boundaries(seg, connectivity=1, mode='outer', background=0)
    bnds = np.logical_and(mask, bnds)
    seg0 = seg*np.logical_not(bnds).astype('int32')
    return seg0

def add_boundary(cc, b):
    h, w = cc.shape
    ccb = np.zeros((h+2*b, w+2*b), dtype='bool')
    ccb[b:h+b, b:w+b] = cc
    return ccb

def remove_boundary(ccb, b):
    h, w = ccb.shape
    cc = np.zeros((h-2*b, w-2*b), dtype='bool')
    cc = ccb[b:h-b, b:w-b]
    return cc

def adaptive_morphological_filter(cc, d):
    d = int(np.round(d))
    ccb = add_boundary(cc, d)
    ccb = dilation(ccb, disk(d))
    ccb = erosion(ccb, disk(d))
    ccb = remove_boundary(ccb, d)
    return ccb

def morphological_smoothing(cc, d=5):
        cc = add_boundary(cc, b=d+2)
        cc = dilation(cc, disk(d))
        cc = erosion(cc, disk(d))
        cc = erosion(cc, disk(d))
        cc = dilation(cc, disk(d))
        cc = remove_boundary(cc, b=d+2)
        return cc

def gradient_smoothing(d, n, l):
    v = np.ones((n-1), dtype='float64')
    m = np.diag(v, k=1) - np.diag(v, k=-1)
    m[0,n-1] = -1
    m[n-1,0] = 1
    d = np.matmul(np.linalg.inv(np.identity(n) + l*np.matmul(m.T, m)), d)
    return d

def convexity_test(cc, cp_thr=5):
    test_result, defects, cnt = test_selfintersection(cc, contour='yes')
    if test_result:
        if defects is None:
            cp = np.array([], dtype='float64').reshape(-1,4)
            d_max = 0
        else:
            cp = defects[defects[:,0,3]>cp_thr*256,0,:]
            cp = cp.reshape(-1,4)
            d_max = int(np.round(np.max(defects[:,0,3])/256))
    else:
        cp = np.array([], dtype='float64').reshape(-1,4)
        d_max = np.nan
    return test_result, cp, d_max, cnt

def virtual_ws_cuts(img, cc, bbox, mask_local_ternary, cnt, cp, dist_transform='DT', large=False):
    """Watershed with Virtual Cuts"""
    h1, w1, h2, w2 = bbox
    lbm0 = np.logical_and(cc, np.logical_not(mask_local_ternary[h1:h2,w1:w2,0]))
    a_g = np.count_nonzero(cc)
    a_l = np.count_nonzero(lbm0)
    da = 100*(a_g-a_l)/a_g
    if da > 0.5:
        cc_wo_cy5 = np.logical_and(np.logical_not(mask_local_ternary[h1:h2,w1:w2,0]), ndimage.binary_fill_holes(cc))
    else:
        cc_wo_cy5 = np.logical_and(np.logical_not(mask_local_ternary[h1:h2,w1:w2,0]), cc)
    dt0 = ndimage.distance_transform_edt(add_layer(cc_wo_cy5))
    dt0 = remove_layer(dt0)
    if dist_transform == 'GDT':
        dt0 = scale_img(dt0)
        dt0 = dt0 * np.exp(1*(1-img[h1:h2,w1:w2,0]))
    
    def vc_seg(ii=0):
        x1, x2 = cnt[cp[ii,0],0], cnt[cp[ii,1],0]
        y1, y2 = cnt[cp[ii,0],1], cnt[cp[ii,1],1]
        x0, y0 = cnt[cp[ii,2],0], cnt[cp[ii,2],1]
        dx = np.abs(x1-x2)
        dy = np.abs(y1-y2)
        if dx < dy:
            x = np.linspace(0, w2-w1-1, w2-w1).astype('int32')
            y = y0 - (x2-x1)*(x-x0)/(y2-y1)
            y = np.round(y)
            y = y.astype('int32')
            t = np.logical_and(y>=0, y<h2-h1)
        else:
            y = np.linspace(0, h2-h1-1, h2-h1).astype('int32')
            x = x0 - (y2-y1)*(y-y0)/(x2-x1)
            x = np.round(x)
            x = x.astype('int32')
            t = np.logical_and(x>=0, x<w2-w1)
        x = x[t]
        y = y[t]
        path = np.ones((h2-h1,w2-w1), dtype='uint8')
        xy = np.array([x, y]).T
        xy = xy.reshape((-1,1,2))
        path = cv2.polylines(path, [xy], False, (0,0,0), 2)
        path = path.astype('bool')
        path = np.logical_and(cc, path)
        cc_cut_reg = label(path, background=False, connectivity=2)
        cc_cut_reg = regionprops(cc_cut_reg)
        if len(cc_cut_reg) < 2:
            seg0 = cc.copy()
            seg0 = seg0.astype('int32')
        else:
            sp_0 = np.zeros((len(cc_cut_reg),3), dtype='float64')
            j = 0
            for p in cc_cut_reg:
                hh1, ww1, hh2, ww2 = p.bbox
                temp0 = np.zeros((hh2-hh1,ww2-ww1), dtype='float64')
                temp0[p.image] = dt0[hh1:hh2, ww1:ww2][p.image]
                yy, xx = np.unravel_index(np.argmax(temp0, axis=None), temp0.shape)
                yy, xx = yy + hh1, xx + ww1
                sp_0[j,:] = (yy, xx, dt0[yy,xx])
                j = j + 1
            if sp_0.shape[0] > 2:
                sorted_index = np.argsort(sp_0[:,2])
                sorted_index = sorted_index[::-1]
                sp_0 = sp_0[sorted_index]
                sp_0 = sp_0[0:2,:]
            markers = np.zeros(dt0.shape, dtype='int32')
            for i in range(sp_0.shape[0]):
                markers[int(sp_0[i,0]), int(sp_0[i,1])] = i + 1
            seg0 = watershed(-dt0, markers, mask=cc)
        return seg0
    
    def min_area(seg):
        prop = regionprops(seg)
        n_ws = len(prop)
        a = np.zeros((n_ws,1), dtype='int32')
        for i in range(n_ws):
            a[i,0] = prop[i].filled_area
        a_min = np.min(a)
        if n_ws == 1:
            a_min = 0
        return a_min
    
    if large:
        seg0 = vc_seg(ii=0)
        a_min_0 = min_area(seg0)
        seg1 = vc_seg(ii=1)
        a_min_1 = min_area(seg1)
        if a_min_0 >= a_min_1:
            return seg0
        else:
            return seg1
    else:
        seg0 = vc_seg(ii=0)
        return seg0

def mask_statistics(mask):
    properties = regionprops(mask)
    a = []
    c = []
    s = []
    for p in properties:
        a.append(p.area)
        c.append(p.solidity)
        s.append(((p.perimeter)**2)/(4*np.pi*p.area))
    a = np.array(a)
    c = np.array(c)
    s = np.array(s)
    mask_status = {'n_total':a.shape[0], 'convexity_avr':np.mean(c), 'shape_avr':np.mean(s),
                   'area_avr':np.mean(a), 'area_std':np.std(a), 'area_min':np.min(a), 'area_max':np.max(a)}
    return mask_status

###############################################################################
### PROCESSING FUNCTIONS
###############################################################################
def normalize_img(intensity):
    """Image with Scaled Channels (0 nuclei channel, 1 cell membrane channel)"""
    h = intensity.shape[0]
    w = intensity.shape[1]
    c = intensity.shape[2]
    img = np.zeros((h,w,c), dtype='float64')
    for i in range(c):
        img[:,:,i] = scale_img(intensity[:,:,i])
    return img

def intensity2rgb(img, return_rgb=False):
    """Create RGB Image Using Nuclei (0) and Cell (1) channel_namesannels"""
    h = img.shape[0]
    w = img.shape[1]
    img_rgb = np.zeros((h,w,3), dtype='float64')
    img_rgb[:,:,0] = img[:,:,1]
    img_rgb[:,:,2] = img[:,:,0]
    if return_rgb:
        return img_rgb
    else:
        plot_img(img_rgb, tlt='Two-channel Image')

def calc_empirical_probabilities(img, sigma=3):
    """Nuclei and Cell Empirical Foreground Probabilities"""
    n_otsu = 3
    dn = 255
    n_channel_names = img.shape[2]
    global_otsu_mask = np.zeros((img.shape), dtype='bool')
    thr = np.zeros((n_channel_names,n_otsu-1), dtype='float64')
    for i in range(n_channel_names):
        img_temp = scale_img(gaussian(img[:,:,i], sigma=sigma))
        thr[i,:] = threshold_multiotsu(img_temp, classes=n_otsu)
        global_otsu_mask[:,:,i] = img_temp > thr[i,0]
    p_em = np.zeros((img.shape), dtype='float64')
    for i in range(n_channel_names):
        img_temp = np.floor(dn*img[:,:,i]).astype('uint8')
        if i == 0:
            f_nuc_fg = img_temp[global_otsu_mask[:,:,i]]
            x_nuc_fg, s = np.unique(f_nuc_fg, return_index=False, return_inverse=False, return_counts=True, axis=None)
            n = f_nuc_fg.shape[0]
            f_nuc_fg = np.cumsum(s)/n
            p_em[:,:,i] = distribution_function(img_temp, x_nuc_fg, f_nuc_fg, dn, mt='fg')
        elif i == 1:
            f_cel_fg = img_temp[global_otsu_mask[:,:,i]]
            x_cel_fg, s = np.unique(f_cel_fg, return_index=False, return_inverse=False, return_counts=True, axis=None)
            n = f_cel_fg.shape[0]
            f_cel_fg = np.cumsum(s)/n
            p_em[:,:,i] = distribution_function(img_temp, x_cel_fg, f_cel_fg, dn, mt='fg')
    return p_em

def calc_global_binary_mask(img, nk=[5,10,20,40], t0=0.5):
    """Foreground Masks"""
    n = img.shape[2]
    mask_global = np.ones((img.shape), dtype='bool')
    for i in range(n):
        for k in nk:
            likelihood = affv(img[:,:,i], k=k, t=t0)
            mask_global[:,:,i] = np.logical_and(mask_global[:,:,i], likelihood != 0)
    return mask_global

def calc_local_binary_mask(img, r0=5, k0=1, smoothing=True):
    """Local Masks"""
    n = img.shape[2]
    temp_img = np.zeros((img.shape), dtype='uint8')
    for i in range(n):
        if smoothing:
            temp_img[:,:,i] = img_as_ubyte(scale_img(gradient_adaptive_smoothing(img[:,:,i], k=k0)))
        else:
            temp_img[:,:,i] = img_as_ubyte(img[:,:,i])
    local_otsu_thr = np.zeros((img.shape), dtype='uint8')
    for i in range(n):
        local_otsu_thr[:,:,i] = rank.otsu(temp_img[:,:,i], disk(r0)) # for old skimage versions use: selem=disk(r0)
    local_otsu_mask = np.zeros((img.shape), dtype='bool')
    for i in range(n):
        local_otsu_mask[:,:,i] = temp_img[:,:,i] > local_otsu_thr[:,:,i]
    return local_otsu_mask

def remove_background(mask_local, mask_global, p_em, pct):
    """Use Empirical Propability Threshold to Remove Background in Foreground and Local Masks (Clarification of Masks)"""
    c = mask_global.shape[2]
    for i in range(c):
        if pct[i] == None:
            thr_prb = threshold_multiotsu(p_em[:,:,i], classes=3)
            thr_prb = thr_prb[0]
        else:
            thr_prb = pct[i]
        temp_mask = p_em[:,:,i] > thr_prb
        print('\tpropability threshold (channel {0}) = {1:.03f}'.format(i, thr_prb))
        mask_global[:,:,i] = np.logical_and(mask_global[:,:,i], temp_mask)
        mask_local[:,:,i] = np.logical_and(mask_local[:,:,i], mask_global[:,:,i])
    return mask_local, mask_global

def calc_likelihood(img, mask_local, mask_global):
    """Likelihood Function (not vectorized version)"""
    (h, w, c) = img.shape
    i_min = np.zeros((c,1), dtype='float64')
    for i in range(c):
        i_min[i,0] = np.min(img[mask_local[:,:,i],i])
        print('\tmin intensity (channel {0}) = {1:.4f}'.format(i, i_min[i,0]))
    t_t = np.zeros((c,1), dtype='float64')
    for i in range(c):
        t_t[i,0] = threshold_multiotsu(img[:,:,0], classes=2)[0]
        #t_t[i,0] = threshold_multiotsu(img[:,:,i], classes=2)[0]
    img_likelihood = np.zeros((img.shape[0], img.shape[1]), dtype='float64')
    for i in range(h):
        for j in range(w):
            if img[i,j,0] > i_min[0,0] or img[i,j,1] > i_min[1,0]:
                i2 = t_t[1]*img[i,j,0]
                i1 = t_t[0]*img[i,j,1]
                img_likelihood[i,j] = -(i2-i1)/(i2+i1)
    m1 = img_likelihood < 0
    m2 = np.logical_and(mask_global[:,:,0], m1)
    m = m1 != m2
    m1 = img_likelihood > 0
    m2 = np.logical_and(mask_global[:,:,1], m1)
    img_likelihood[m] = 0
    m = m1 != m2
    img_likelihood[m] = 0
    return img_likelihood

def calc_aposteriori_probabilities(p_em, mask_local, likelihood):
    """Local & A Posteriori Probabilities to Compute Bayes Masks"""
    lt = 0
    p_local = np.zeros((p_em.shape), dtype='float64')
    p_apriori = np.zeros((p_em.shape), dtype='float64')
    p_global = np.zeros((p_em.shape), dtype='float64')
    for i in range(p_em.shape[2]):
        p_local[mask_local[:,:,i],i] = p_em[mask_local[:,:,i],i]
        p_apriori[:,:,i] = p_em[:,:,i]
        p_apriori[mask_local[:,:,i],i] = 1
        if i == 0:
            m = likelihood < -lt
        elif i == 1:
            m = likelihood > lt
        p_global[m,i] = p_apriori[m,i] + np.abs(likelihood[m]) - p_apriori[m,i] * np.abs(likelihood[m])
    return p_local, p_global

def calc_ternary_mask(img, prob, met='Argmax'):
    """Calculation of Ternary (Bayes-like) Mask Using K-Means or Argmax method"""
    mask_ternary = np.zeros((img.shape[0],img.shape[1],3), dtype='uint8')
    if met == 'Kmeans':
        n_clusters = 3
        x = np.zeros((img.shape[0]*img.shape[1],2), dtype='float64')
        pa = np.sum(prob, axis=2)
        m = pa > 0
        x[m.reshape(-1,),0] = (prob[m,0]/pa[m]).reshape(-1,)
        x[m.reshape(-1,),1] = (prob[m,1]/pa[m]).reshape(-1,)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(x)
        mask_kmeans = kmeans.labels_
        mask_kmeans = mask_kmeans.reshape(img.shape[0], img.shape[1])
        i_n = np.zeros((n_clusters,), dtype='float64')
        for i in range(n_clusters):
            i_n[i] = np.mean(img[mask_kmeans==i,0])
        i_n = np.argmax(i_n)
        i_c = np.zeros((n_clusters,), dtype='float64')
        for i in range(n_clusters):
            i_c[i] = np.mean(img[mask_kmeans==i,1])
        i_c = np.argmax(i_c)
        mask_ternary[mask_kmeans==i_n,2] = 255
        mask_ternary[mask_kmeans==i_c,0] = 255
    elif met == 'Argmax':
        bg = np.sum(prob, axis=2)
        bg = bg == 0
        nc = np.argmax(prob, axis=2)
        mask_ternary[nc==0,2] = 255
        mask_ternary[nc==1,0] = 255
        mask_ternary[bg,:] = 0
    return mask_ternary

def convexity_analysis(mask_ternary, area_thr=20, cp_thr=5):
    """Convert Global Ternary Mask into Initial Nuclei Segmentation, List of Nuclei Clusters and List of Small Objects"""
    props = label(mask_ternary[:,:,2] == 255, background=False, connectivity=2)
    props = regionprops(props)
    nuc_seg = np.zeros((mask_ternary.shape[0],mask_ternary.shape[1]), dtype='int32')
    nuc_seg_features = []
    nuc_clust = []
    nuc_clust_features = [[],[],[]]
    small_obj = []
    clusters_self = []
    mask_info = {'removed':0, 'self-intersection':0, 'rectangle':0, 'convex':0, 'cluster':0}
    nuclei_clusters_info = {'nuclei':{}, 'clusters':{}}
    i = 1
    for p in props:
        h1, w1, h2, w2 = p.bbox
        if p.filled_area < area_thr:
            cc_type = 'removed'
        else:
            cnt = find_contour((p.filled_image).astype('uint8'))
            hull = cv2.convexHull(cnt, returnPoints=False)
            try:
                defects = cv2.convexityDefects(cnt, hull)
                if defects is None:
                    cc_type = 'rectangle'
                    nuc_seg[h1:h2, w1:w2][p.filled_image] = i
                    i = i + 1
                    nuc_seg_features.append([p.filled_area])
                else:
                    cp = defects[defects[:,0,3]>cp_thr*256,0,:]
                    n_cp = cp.shape[0]
                    if n_cp == 0:
                        cc_type = 'convex'
                        nuc_seg[h1:h2, w1:w2][p.filled_image] = i
                        i = i + 1
                        nuc_seg_features.append([p.filled_area])
                    else:
                        cc_type = 'cluster'
                        nuc_clust.append((p.image, p.bbox))
                        nuc_clust_features[0].append(n_cp)
                        nuc_clust_features[1].append(cp[:,3]/256)
                        nuc_clust_features[2].append(p.area)
            except:
                cc_type = 'self-intersection'
                clusters_self.append((p.filled_image, p.bbox))
        mask_info[cc_type] = mask_info[cc_type] + 1
    while len(clusters_self) > 0:
        cl = clusters_self[0]
        cc = cl[0]
        h1, w1, h2, w2 = cl[1]
        clusters_self.remove(cl)
        cc = remove_selfintersection(cc)
        props = label(cc, background=False, connectivity=2)
        props = regionprops(props)
        if len(props) > 0:
            for p in props:
                hh1, ww1, hh2, ww2 = p.bbox
                test_res, defects = test_selfintersection(p.image)
                if p.filled_area < area_thr:
                    cc_type = 'removed'
                else:
                    dh1 = h1 + hh1
                    dh2 = dh1 - hh1 + hh2
                    dw1 = w1 + ww1
                    dw2 = dw1 - ww1 + ww2
                    if test_res:
                        if defects is None:
                            cc_type = 'rectangle'
                            nuc_seg[dh1:dh2, dw1:dw2][p.filled_image] = i
                            i = i + 1
                            nuc_seg_features.append([p.filled_area])
                        else:
                            cp = defects[defects[:,0,3]>cp_thr*256,0,:]
                            n_cp = cp.shape[0]
                            if n_cp == 0:
                                cc_type = 'convex'
                                nuc_seg[dh1:dh2, dw1:dw2][p.filled_image] = i
                                i = i + 1
                                nuc_seg_features.append([p.filled_area])
                            else:
                                cc_type = 'cluster'
                                nuc_clust.append((p.image, (dh1, dw1, dh2, dw2)))
                                nuc_clust_features[0].append(n_cp)
                                nuc_clust_features[1].append(cp[:,3]/256)
                                nuc_clust_features[2].append(p.area)
                    else:
                        cc_type = 'self-intersection'
                        clusters_self.append((p.filled_image, (dh1, dw1, dh2, dw2)))
            mask_info[cc_type] = mask_info[cc_type] + 1
    nuc_seg_features = np.array(nuc_seg_features)
    n_avr = np.mean(nuc_seg_features[:,0])
    n_std = np.std(nuc_seg_features[:,0])
    n_max = np.max(nuc_seg_features[:,0])
    n_min = np.min(nuc_seg_features[:,0])
    n_area_thr = int(np.min([n_max-n_std, n_avr/4, n_min+n_std]))
    
    props = regionprops(nuc_seg)
    nuc_seg = np.zeros((mask_ternary.shape[0],mask_ternary.shape[1]), dtype='int32')
    nuc_seg_features = []
    i = 1
    for p in props:
        h1, w1, h2, w2 = p.bbox
        if p.area >= n_area_thr:
            ccf = adaptive_morphological_filter(p.filled_image, cp_thr-1)
            nuc_seg[h1:h2,w1:w2][ccf] = i
            nuc_seg_features.append(p.area)
            i = i + 1
        else:
            ccf = adaptive_morphological_filter(p.filled_image, cp_thr-1)
            small_obj.append((ccf, (h1, w1, h2, w2)))
    mask_info['NS'] = np.max(nuc_seg)
    mask_info['SO'] = len(small_obj)
    nuc_seg_features = np.array(nuc_seg_features)
    a_avr = np.array(nuc_clust_features[2])
    a_avr = np.mean(a_avr)
    a_avr = int(np.floor(a_avr))
    n_cp_avr = np.array(nuc_clust_features[0])
    n_cp_avr = np.mean(n_cp_avr)
    n_cp_avr = int(np.floor(n_cp_avr))
    d_cp_avr = []
    for dd in nuc_clust_features[1]:
        for ddd in list(dd):
            d_cp_avr.append(ddd)
    d_cp_avr = np.array(d_cp_avr)
    d_cp_avr = np.mean(d_cp_avr)
    nuclei_clusters_info['nuclei']['number'] = len(nuc_seg_features)
    nuclei_clusters_info['nuclei']['min'] = np.min(nuc_seg_features)
    nuclei_clusters_info['nuclei']['max'] = np.max(nuc_seg_features)
    nuclei_clusters_info['nuclei']['avr'] = np.mean(nuc_seg_features)
    nuclei_clusters_info['nuclei']['std'] = np.std(nuc_seg_features)
    nuclei_clusters_info['nuclei']['data'] = nuc_seg_features
    nuclei_clusters_info['nuclei']['nuclei area threshold'] = n_area_thr
    nuclei_clusters_info['clusters']['Number'] = len(nuc_clust_features[0])
    nuclei_clusters_info['clusters']['AvrArea'] = a_avr
    nuclei_clusters_info['clusters']['Ncp'] = n_cp_avr
    nuclei_clusters_info['clusters']['Dcp'] = d_cp_avr
    print('\t---------------------------------------------------\n'+
          '\tNon-convexity Threshold = {0:.0f} pxs\n'.format(cp_thr)+
          '\tMin Area Threshold = {0:.0f} pxs\n'.format(area_thr)+
          '\tSmall Object Area Threshold = {0:.0f} pxs\n'.format(n_area_thr)+
          '\t---------------------------------------------------\n'+
          '\tSummary for Nuclei Global Ternary Mask\n'+
          '\t---------------------------------------------------\n'+
          '\t\t- Number of removed components = {0} (Area < {1} pxs)\n'.format(mask_info['removed'], area_thr)+
          '\t\t- Number of components with self-intersections = {0} ({1})\n'.format(mask_info['self-intersection'], len(clusters_self))+
          '\t\t- Number of rectangle components = {0}\n'.format(mask_info['rectangle'])+
          '\t\t- Number of convex components = {0}\n'.format(mask_info['convex'])+
          '\t\t-----------------------------------------------\n'+
          '\t\t- Nuclei Segmentation (NS)\n'+
          '\t\t\tNumber of nuclei = {0}\n'.format(nuclei_clusters_info['nuclei']['number'])+
          '\t\t\tMean nuclei area = {0:.1f} pxs\n'.format(nuclei_clusters_info['nuclei']['avr'])+
          '\t\t\tSTD of nuclei area = {0:.1f} pxs\n'.format(nuclei_clusters_info['nuclei']['std'])+
          '\t\t\tMin nuclei area = {0} pxs\n'.format(nuclei_clusters_info['nuclei']['min'])+
          '\t\t\tMax nuclei area = {0} pxs\n'.format(nuclei_clusters_info['nuclei']['max'])+
          '\t\t-----------------------------------------------\n'+
          '\t\t- Nuclei Clusters (NC)\n'+
          '\t\t\tNumber of clusters = {0}\n'.format(nuclei_clusters_info['clusters']['Number'])+
          '\t\t\tMean cluster area = {0:.1f} pxs\n'.format(nuclei_clusters_info['clusters']['AvrArea'])+
          '\t\t\tMean number of CPs = {0:.1f}\n'.format(nuclei_clusters_info['clusters']['Ncp'])+
          '\t\t\tMean defect distance = {0:.1f} pxs\n'.format(nuclei_clusters_info['clusters']['Dcp'])+
          '\t\t-----------------------------------------------\n'+
          '\t\t- Small Objects (SO)\n'+
          '\t\t\tNumber of objects = {0} (Area < {1} pxs)\n'.format(mask_info['SO'], n_area_thr)+
          '\t---------------------------------------------------')
    return nuc_seg, nuc_clust, small_obj, nuclei_clusters_info

def process_clusters(img, mask_local_ternary, clusters, nuc_seg, clust_small, nuclei_clusters_info, cp_thr=5, area_thr=20, cy5_thr=25, dist_tr='DT'):
    """Processing of Nuclei Clusters Using Pertubated Watershed and Watershed with Virtual Cuts"""
    print('\ta. Pertubated Watershed')
    try:
        otsu_3 = threshold_multiotsu(nuclei_clusters_info['nuclei']['data'], classes=3)
        a1 = otsu_3[0]
    except:
        a1 = np.min(nuclei_clusters_info['nuclei']['data'])
    a2 = np.max([2*nuclei_clusters_info['nuclei']['avr'], nuclei_clusters_info['nuclei']['max']])
    a3 = np.min([nuclei_clusters_info['nuclei']['avr']+3*nuclei_clusters_info['nuclei']['std'], 2*nuclei_clusters_info['nuclei']['max']])
    c1 = nuclei_clusters_info['clusters']['Dcp']
    clust_unknown = []
    n_nuc = np.max(nuc_seg)
    while len(clusters) > 0:
        cl = clusters[0]
        cc = cl[0]
        h1, w1, h2, w2 = cl[1]
        clusters.remove(cl)
        gbm0 = np.zeros((h2-h1, w2-w1), dtype='bool')
        gbm0[cc] = True
        lbm0 = np.logical_and(gbm0, np.logical_not(mask_local_ternary[h1:h2,w1:w2,0]))
        seg0, da = dynamic_ws(gbm0, lbm0)
        seg0 = boundary_cuts(seg0, gbm0)
        pp = regionprops(seg0)
        n_ws = len(pp)
        if n_ws == 1:
            p = pp[0]
            hh1, ww1, hh2, ww2 = p.bbox
            test_result, cp, d_max, cnt = convexity_test(p.filled_image, cp_thr=cp_thr)
            n_cp = cp.shape[0]
            if n_cp == 0:
                cp_max = 0
            else:
                cp_max = np.max(cp[:,3])/256
            if test_result:
                if p.filled_area < a1:
                    if p.filled_area > area_thr:
                        ccf = adaptive_morphological_filter(p.filled_image, d_max)
                        clust_small.append((ccf, (h1+hh1, w1+ww1, h1+hh2, w1+ww2)))
                else:
                    if n_cp >= 2 and da <= cy5_thr and (cp_max > c1 or p.filled_area > a3):
                        clust_unknown.append((p.image, (h1+hh1, w1+ww1, h1+hh2, w1+ww2)))
                    else:
                        n_nuc = n_nuc + 1
                        ccf = adaptive_morphological_filter(p.filled_image, d_max)
                        nuc_seg[h1+hh1:h1+hh2, w1+ww1:w1+ww2][ccf] = n_nuc
                        #nuc_seg[h1:h2, w1:w2][hh1:hh2, ww1:ww2][ccf] = n_nuc
            else:
                print('\t\tWARNING: Self-intersection in Perturbed Watershed: Nsp = 1')
                cc_wosi = remove_selfintersection(p.image)
                cc_wosi_props = label(cc_wosi, background=False, connectivity=2)
                cc_wosi_props = regionprops(cc_wosi_props)
                if len(cc_wosi_props) > 0:
                    for p_wosi in cc_wosi_props:
                        hhh1, www1, hhh2, www2 = p_wosi.bbox
                        if p_wosi.area > area_thr:
                            clusters.append((p_wosi.image, (h1+hh1+hhh1, w1+ww1+www1, h1+hh1+hhh2, w1+ww1+www2)))
        elif n_ws > 1:
            for p in pp:
                hh1, ww1, hh2, ww2 = p.bbox
                test_result, cp, d_max, cnt = convexity_test(p.filled_image, cp_thr=cp_thr)
                n_cp = cp.shape[0]
                if n_cp == 0:
                    cp_max = 0
                else:
                    cp_max = np.max(cp[:,3])/256
                if test_result:
                    if p.filled_area < a1:
                        if p.filled_area > area_thr:
                            ccf = adaptive_morphological_filter(p.filled_image, d_max)
                            clust_small.append((ccf, (h1+hh1, w1+ww1, h1+hh2, w1+ww2)))
                    elif p.filled_area < a2:
                        if n_cp >= 2 and cp_max > c1:
                            clusters.append((p.image, (h1+hh1, w1+ww1, h1+hh2, w1+ww2)))
                        else:
                            n_nuc = n_nuc + 1
                            ccf = adaptive_morphological_filter(p.filled_image, d_max)
                            nuc_seg[h1+hh1:h1+hh2, w1+ww1:w1+ww2][ccf] = n_nuc
                    else:
                        clusters.append((p.image, (h1+hh1, w1+ww1, h1+hh2, w1+ww2)))
                else:
                    print('\t\tWARNING: Self-intersection in Perturbed Watershed: Nsp > 1')
                    cc_wosi = remove_selfintersection(p.image)
                    cc_wosi_props = label(cc_wosi, background=False, connectivity=2)
                    cc_wosi_props = regionprops(cc_wosi_props)
                    if len(cc_wosi_props) > 0:
                        for p_wosi in cc_wosi_props:
                            hhh1, www1, hhh2, www2 = p_wosi.bbox
                            if p_wosi.area > area_thr:
                                clusters.append((p_wosi.image, (h1+hh1+hhh1, w1+ww1+www1, h1+hh1+hhh2, w1+ww1+www2)))
        else:
            print('\t\tWARNING: in Perturbed Watershed the number of seed points is not positive, Nsp = {0}'.format(n_ws))
    print('\t\t- Nuclei Segmentation: {0}\n'.format(n_nuc)+
          '\t\t- Nuclei Clusters\t: {0}\n'.format(len(clust_unknown))+
          '\t\t- Small Objects\t: {0}'.format(len(clust_small)))
    print('\tb. Virtual Cuts')
    while len(clust_unknown) > 0:
        cl = clust_unknown[0]
        cc = cl[0]
        h1, w1, h2, w2 = cl[1]
        clust_unknown.remove(cl)
        test_result, cp, d_max, cnt = convexity_test(cc, cp_thr=cp_thr)
        n_cp = cp.shape[0]
        a_cc = np.count_nonzero(ndimage.binary_fill_holes(cc))
        if n_cp == 0:
            cp_max = 0
        else:
            cp_max = np.max(cp[:,3])/256
            cp = cp[cp[:,3].argsort()]
            cp = cp[::-1]
        if n_cp > 0:
            cp = cp[cp[:,3].argsort()]
            cp = cp[::-1]
        if test_result:
            lbm0 = np.logical_and(cc, np.logical_not(mask_local_ternary[h1:h2,w1:w2,0]))
            a_g = np.count_nonzero(cc)
            a_l = np.count_nonzero(lbm0)
            da = 100*(a_g-a_l)/a_g
            if n_cp >= 2 and (cp_max > c1 or a_cc > a3 or (da < 0.001 and a_cc > 2*a1)):
                if a_cc > 2*a3:
                    vl = True
                else:
                    vl = False
                seg0 = virtual_ws_cuts(img, cc, (h1, w1, h2, w2), mask_local_ternary, cnt, cp, dist_transform=dist_tr, large=vl)
                pp = regionprops(seg0)
                n_ws = len(pp)
                if n_ws == 1:
                    p = pp[0]
                    hh1, ww1, hh2, ww2 = p.bbox
                    if p.filled_area < a1:
                        if p.filled_area > area_thr:
                            ccf = adaptive_morphological_filter(p.filled_image, d_max)
                            clust_small.append((ccf, (h1+hh1, w1+ww1, h1+hh2, w1+ww2)))
                    else:
                        n_nuc = n_nuc + 1
                        ccf = adaptive_morphological_filter(p.filled_image, d_max)
                        nuc_seg[h1+hh1:h1+hh2, w1+ww1:w1+ww2][ccf] = n_nuc
                elif n_ws == 2:
                    a = np.zeros((n_ws,1), dtype='int32')
                    for i in range(n_ws):
                        a[i,0] = pp[i].filled_area
                    a_min = np.min(a)
                    if a_min < a1:
                        n_nuc = n_nuc + 1
                        ccf = ndimage.binary_fill_holes(cc)
                        ccf = adaptive_morphological_filter(ccf, d_max)
                        nuc_seg[h1:h2, w1:w2][ccf] = n_nuc
                    else:
                        seg0 = boundary_cuts(seg0, cc)
                        pp = regionprops(seg0)
                        for p in pp:
                            hh1, ww1, hh2, ww2 = p.bbox
                            clust_unknown.append((p.image, (h1+hh1, w1+ww1, h1+hh2, w1+ww2)))
                else:
                    print('\t\tWARNING: in Virtual Cuts the number of seed points is neither 1 nor 2')
            else:
                if a_cc < a1:
                    if a_cc > area_thr:
                        ccf = adaptive_morphological_filter(cc, d_max)
                        clust_small.append((ccf, (h1, w1, h2, w2)))
                else:
                    n_nuc = n_nuc + 1
                    ccf = adaptive_morphological_filter(cc, d_max)
                    nuc_seg[h1:h2, w1:w2][ccf] = n_nuc
        else:
            print('Self-intersection in Virtual Cuts')
            cc_wosi = remove_selfintersection(p.image)
            cc_wosi_props = label(cc_wosi, background=False, connectivity=2)
            cc_wosi_props = regionprops(cc_wosi_props)
            if len(cc_wosi_props) > 0:
                for p_wosi in cc_wosi_props:
                    hhh1, www1, hhh2, www2 = p_wosi.bbox
                    if p_wosi.area > area_thr:
                        clust_unknown.append((p_wosi.image, (h1+hh1+hhh1, w1+ww1+www1, h1+hh1+hhh2, w1+ww1+www2)))
    print('\t\t- Nuclei Segmentation: {0}\n'.format(n_nuc)+
          '\t\t- Nuclei Clusters\t: {0}\n'.format(len(clust_unknown))+
          '\t\t- Small Objects\t: {0}'.format(len(clust_small)))
    return nuc_seg, clust_small

def process_small_objects(clust_small, nuc_seg, nuclei_clusters_info, h, w):
    """Processing of Small Connected Components"""
    print('\tc. Adding of Small Objects')
    
    def wide_mask(cc, d, h1, w1, h2, w2, h, w):
        hh1 = np.max([0, h1-d])
        ww1 = np.max([0, w1-d])
        hh2 = np.min([h, h2+d])
        ww2 = np.min([w, w2+d])
        ccb = np.zeros((hh2-hh1, ww2-ww1), dtype='bool')
        ccb[h1-hh1:h2-hh1, w1-ww1:w2-ww1][cc] = True
        ccb = dilation(ccb, disk(d))
        mask_wide = nuc_seg[hh1:hh2, ww1:ww2][ccb]
        mw = np.count_nonzero(mask_wide)
        return mw
    
    d1 = 3 
    small_area = []
    for i in range(len(clust_small)):
        cl = clust_small[i]
        cc = cl[0]
        small_area.append(np.count_nonzero(cc))
    small_area = np.array(small_area, dtype='int32')
    inds = np.argsort(small_area)
    inds = inds[::-1]
    otsu_3 = threshold_multiotsu(small_area, classes=3)
    a_max = np.max([otsu_3[1], nuclei_clusters_info['nuclei']['min']])
    a_min = np.max([otsu_3[0], nuclei_clusters_info['nuclei']['min']/2])
    print('\t\t* Min Area Threshold = {0}'.format(a_min))
    print('\t\t* Max Area Threshold = {0}'.format(a_max))
    n_nuc = np.max(nuc_seg)
    for i in inds:
        cl = clust_small[i]
        cc = cl[0]
        h1, w1, h2, w2 = cl[1]
        ai = np.count_nonzero(cc)
        mw1 = wide_mask(cc, d1, h1, w1, h2, w2, h, w)
        mn = np.count_nonzero(nuc_seg[h1:h2, w1:w2][cc])
        if mn == 0 and ai >= a_max:
            n_nuc = n_nuc + 1
            nuc_seg[h1:h2, w1:w2][cc] = n_nuc
        elif mw1 == 0 and ai >= a_min:
            n_nuc = n_nuc + 1
            nuc_seg[h1:h2, w1:w2][cc] = n_nuc
    print('\t\t- Nuclei Segmentation: {0}\n'.format(n_nuc)+
          '\t\t- Nuclei Clusters\t: 0\n'+
          '\t\t- Small Objects\t: {0}'.format(len(clust_small)))
    return nuc_seg

def filter_nuclei_mask(mask):
    l = 2
    properties = regionprops(mask)
    new_mask = np.zeros(mask.shape, dtype='int32')
    for p in properties:
        h1, w1, h2, w2 = p.bbox
        h = h2 - h1
        w = w2 - w1
        cc0 = new_mask[h1:h2,w1:w2] == 0
        y0, x0 = p.centroid
        cnt = find_contour((p.filled_image).astype('uint8'))
        cnt = cnt[:,0,:]
        n_theta = cnt.shape[0]
        r = np.sqrt((cnt[:,0]-x0)**2 + (cnt[:,1]-y0)**2)
        theta = np.arctan2(cnt[:,1]-y0, cnt[:,0]-x0)*180/np.pi
        r_smooth = gradient_smoothing(r, n_theta, l)
        x_smooth = (np.round(x0+r_smooth*np.cos(theta*np.pi/180))).astype('int32')
        x_smooth = x_smooth.reshape(-1,1)
        x_smooth = np.where(x_smooth < 0, 0, x_smooth)
        x_smooth = np.where(x_smooth >= w, w-1, x_smooth)
        y_smooth = (np.round(y0+r_smooth*np.sin(theta*np.pi/180))).astype('int32')        
        y_smooth = y_smooth.reshape(-1,1)
        y_smooth = np.where(y_smooth < 0, 0, y_smooth)
        y_smooth = np.where(y_smooth >= h, h-1, y_smooth)
        cc = np.zeros((h,w), dtype='uint8')
        pts = np.concatenate((x_smooth, y_smooth), axis=1)
        cc = cv2.fillPoly(cc, pts=[pts], color=(1,1,1))
        cc = cc.astype('bool')
        cc = np.logical_and(cc0, cc)
        prop = label(cc, background=False, connectivity=2)
        prop = regionprops(prop)
        if len(prop) == 1:
            new_mask[h1:h2,w1:w2][cc] = p.label
        elif len(prop) == 0:
            new_mask[h1:h2,w1:w2][p.filled_image] = p.label
        else:
            new_mask[h1:h2,w1:w2][p.filled_image] = p.label
    n0 = np.max(new_mask)
    n1 = list(np.unique(new_mask))
    if 0 in n1:
        n1.remove(0)
    n1 = len(n1)
    if n0 != n1:
        print('\tWARNING: Nuclei Segmentation Labeling Is Not Continuous')
        #new_mask = label(new_mask, background=0, connectivity=2)
    mask_status = mask_statistics(new_mask)
    print('\t---------------------------------------------------\n'+
          '\tSummary for Nuclei Segmentation\n'+
          '\t---------------------------------------------------\n'+
          '\t\t- Number of nuclei = {0}\n'.format(mask_status['n_total'])+
          '\t\t-----------------------------------------------\n'+
          '\t\t- Mean nuclei area = {0:.1f} pxs\n'.format(mask_status['area_avr'])+
          '\t\t- STD of nuclei area = {0:.1f} pxs\n'.format(mask_status['area_std'])+
          '\t\t- Min nuclei area = {0} pxs\n'.format(mask_status['area_min'])+
          '\t\t- Max nuclei area = {0} pxs\n'.format(mask_status['area_max'])+
          '\t\t-----------------------------------------------\n'+
          '\t\t- Mean nuclei convexity = {0:.4f}\n'.format(mask_status['convexity_avr'])+
          '\t\t- Mean nuclei compactness = {0:.4f}\n'.format(mask_status['shape_avr'])+
          '\t---------------------------------------------------')
    return new_mask, mask_status

### Initialization of Cell Segmentation
def initialize_cell_segmentation(nuc_seg):
    cell_seg = nuc_seg.copy()
    nuclei_prop_list = regionprops(cell_seg)
    nuclei_prop = {}
    for p in nuclei_prop_list:
        nuclei_prop[p.label] = p
    nuclei = list(nuclei_prop.keys())
    n_nuclei = len(nuclei)
    return cell_seg, nuclei_prop, nuclei, n_nuclei

### Skeletonization
def get_skeleton_mod(mask):
    skeleton = thin(mask)
    skeleton = skeleton.astype('bool')
    skeleton = dilation(skeleton, disk(1))
    return skeleton

def get_mask_cuts_mod(mask, area_threshold_cell):
    mask_cells = np.zeros((mask.shape[0], mask.shape[1]), dtype='bool')
    mm = label(mask, background=False, connectivity=2)
    mm = regionprops(mm)
    for m in mm:
        cc = ndimage.binary_fill_holes(m.image)
        a = np.count_nonzero(cc)
        if a >= area_threshold_cell:
            h1, w1, h2, w2 = m.bbox
            mask_cells[h1:h2, w1:w2][m.image] = True
    skeleton = get_skeleton_mod(mask_cells)
    mask_cells = np.logical_and(mask_cells, np.logical_not(skeleton))
    skeleton = get_skeleton_mod(mask)
    return mask_cells, skeleton

### Euler Number Filter
def euler_number_filter_mod(mask_cell_lab, mask_cells_cuts, nuclei, n_nuclei, small_area=50, area_ratio_thr=0.65):
    n_cells = n_nuclei
    mask_cell_cuts_prop = label(mask_cells_cuts, background=False, connectivity=2)
    mask_cell_cuts_prop = regionprops(mask_cell_cuts_prop)
    for p in mask_cell_cuts_prop:
        if p.euler_number != 1:
            a1 = p.area
            cc0 = ndimage.binary_fill_holes(p.image)
            a2 = np.count_nonzero(cc0)
            if a2 >= small_area and a1/a2 < area_ratio_thr:
                h1, w1, h2, w2 = p.bbox
                n_unique, n_counts = np.unique(mask_cell_lab[h1:h2, w1:w2][cc0], return_counts=True)
                n_inds = np.argsort(n_counts)[::-1]
                n = 0
                for ind in n_inds:
                    if ind == 0:
                        continue
                    else:
                        n = n_unique[ind]
                        break
                cc = np.zeros((h2-h1,w2-w1), dtype='bool')
                cc[cc0] = True
                if n == 0:
                    mask_allowed = mask_cell_lab[h1:h2,w1:w2] == 0
                    n_cells = n_cells + 1
                    n0 = n_cells
                else:
                    mask_allowed = np.logical_or(mask_cell_lab[h1:h2,w1:w2] == 0, mask_cell_lab[h1:h2,w1:w2] == n)
                    n0 = n
                    nuclei.remove(n)
                cc = morphological_smoothing(cc, d=2)
                cc = ndimage.binary_fill_holes(cc)
                cc = np.logical_and(mask_allowed, cc)
                if n_inds.shape[0] > 2:
                    e8 = label(cc, background=False, connectivity=2)
                    e8 = regionprops(e8)
                    e8 = e8[0]
                    e8 = e8.euler_number
                    if e8 != 1 and n != 0:
                        nuclei.append(n)
                        continue
                mask_cell_lab[h1:h2,w1:w2][cc==True] = n0
    return mask_cell_lab, nuclei, n_cells

### Isolated Nuclei Filter
def extend_cell(cc, cc_bbox=(0,0,10,20), mask_hw=(50,100), d=9):
    h1, w1, h2, w2 = cc_bbox
    dh = h2 - h1
    dw = w2 - w1
    h, w = mask_hw
    if h1 >= d:
        y1 = d
    else:
        y1 = h1
    if w1 >= d:
        x1 = d
    else:
        x1 = w1
    if h-h2 >= d:
        y2 = d
    else:
        y2 = h-h2
    if w-w2 >= d:
        x2 = d
    else:
        x2 = w-w2
    ccb = np.zeros((y1+dh+y2,x1+dw+x2), dtype='bool')
    ccb[y1:y1+dh, x1:x1+dw] = cc
    ccb = dilation(ccb, disk(d))
    ccb_bbox = (h1-y1, w1-x1, h2+y2, w2+x2)
    return ccb, ccb_bbox

def isolated_nuclei_filter(mask_cell_lab, mask_cells, nuclei_prop, nuclei, u0=9):
    nuclei_new = nuclei.copy()
    for n in nuclei:
        p = nuclei_prop[n]
        h1, w1, h2, w2 = p.bbox
        cc = np.zeros((h2-h1,w2-w1), dtype='bool')
        cc[p.image] = True
        cc, ext_bbox = extend_cell(cc, cc_bbox=(h1,w1,h2,w2), mask_hw=(mask_cell_lab.shape[0],mask_cell_lab.shape[1]), d=u0)
        h1, w1, h2, w2 = ext_bbox
        n_cc = np.sum(np.logical_and(cc, mask_cells[h1:h2,w1:w2]))
        if n_cc < 25:
            mask_allowed = np.logical_or(mask_cell_lab[h1:h2,w1:w2] == n, mask_cell_lab[h1:h2,w1:w2] == 0)
            cc = np.logical_and(mask_allowed, cc)
            cc = erosion(cc, disk(3))
            cc = dilation(cc, disk(3))
            cc = np.logical_and(mask_allowed, cc)
            mask_cell_lab[h1:h2,w1:w2][cc==True] = n
            nuclei_new.remove(n)
    return mask_cell_lab, nuclei_new

### Threshold Filter
def regularize_cell_boundary(d, thr=10):
    def smooth_jump(z, m, n, thr=10):
        smooth = False
        z1 = np.roll(z, shift=-1)
        dz = z1 - z
        z_abs_max = np.max(np.abs(dz))
        if z_abs_max > thr:
            smooth = True
            i = np.argmax(np.abs(dz))
            if dz[i] > 0:
                if i < n-1:
                    j = i + 1
                else:
                    j = 0
                m[j] = True
                z[j] = z[i]
            else:
                if i < n-1:
                    j = i + 1
                else:
                    j = 0
                m[i] = True
                z[i] = z[j]
        return z, m, smooth
    
    d_cell = d.copy()
    n = d.shape[0]
    m = np.zeros((n,), dtype='bool')
    for i in range(n):
        d_cell, m, smooth = smooth_jump(d_cell, m, n, thr=thr)
        if not smooth:
            break
    d_ind = (m == True).astype('int32')
    if np.max(d_ind) == 0:
        return d_cell, thr
    else:
        ind_start = (d_ind - np.roll(d_ind, shift=-1)) == -1
        ind_end = (d_ind - np.roll(d_ind, shift=1)) == -1
        ind_start_i = np.argwhere(ind_start)
        ind_end_i = np.argwhere(ind_end)
        if ind_start_i[0] >= ind_end_i[0]:
            ind_end_i = np.roll(ind_end_i, shift=-1)
            ind_start_i[-1,0] = ind_start_i[-1,0]-d_ind.shape[0]
        for i in range(ind_start_i.shape[0]):
            t1 = ind_start_i[i,0]
            x1 = d[t1]
            t2 = ind_end_i[i,0]
            x2 = d[t2]
            for t in range(t1+1, t2):
                x = x1 + (t-t1)*(x2-x1)/(t2-t1)
                d_cell[t] = x
        return d_cell, thr

def xiyi_2_xnyn(xi, yi, w, h):
    bnd = False
    xn = int(np.round(xi))
    yn = int(np.round(yi))
    if xn <= 0:
        xn = 0
        bnd = True
    elif xn >= w-1:
        xn = w-1
        bnd = True
    if yn <= 0:
        yn = 0
        bnd = True
    elif yn >= h-1:
        yn = h-1
        bnd = True
    return xn, yn, bnd

def threshold_filter(mask_cell_lab, mask_skeleton, nuclei_prop, nuclei):
    n_theta = 72
    l = 2
    d = 5
    theta = (2*np.pi/n_theta)*np.linspace(0, n_theta-1, n_theta)
    h, w = mask_cell_lab.shape
    mask_base = mask_cell_lab.copy()
    mask_base[mask_skeleton == True] = -1
    for n in nuclei:
        x_y_d = np.zeros((n_theta,8), dtype='float64')
        p = nuclei_prop[n]
        hn1, wn1, hn2, wn2 = p.bbox
        y0, x0 = p.centroid
        for i in range(n_theta):
            xi = x0
            yi = y0
            xm = int(np.round(x0))
            ym = int(np.round(y0))
            xn = int(np.round(x0))
            yn = int(np.round(y0))
            flag = True
            if theta[i] == np.pi/2:
                while mask_base[yn,xn] == n or mask_base[yn,xn] == 0:
                    xi = xi
                    yi = yi + 1
                    xn, yn, bnd = xiyi_2_xnyn(xi, yi, w, h)
                    if bnd:
                        break
                    if mask_base[ym,xm] == n and mask_base[yn,xn] != n and flag:
                        flag = False
                        xm = int(np.round(xi))
                        ym = int(np.round(yi-1))
            elif theta[i] == 3*np.pi/2:
                while mask_base[yn,xn] == n or mask_base[yn,xn] == 0:
                    xi = xi
                    yi = yi - 1
                    xn, yn, bnd = xiyi_2_xnyn(xi, yi, w, h)
                    if bnd:
                        break
                    if mask_base[ym,xm] == n and mask_base[yn,xn] != n and flag:
                        flag = False
                        xm = int(np.round(xi))
                        ym = int(np.round(yi+1))
            else:
                while mask_base[yn,xn] == n or mask_base[yn,xn] == 0:
                    xi = xi + np.cos(theta[i])
                    yi = y0 + np.tan(theta[i])*(xi-x0)
                    xn, yn, bnd = xiyi_2_xnyn(xi, yi, w, h)
                    if bnd:
                        break
                    if mask_base[yn,xn] != n and flag:
                        flag = False
                        xm = int(np.round(xi-np.cos(theta[i])))
                        ym = int(np.round(y0+np.tan(theta[i])*(xi-np.cos(theta[i])-x0)))
            if flag:
                x_y_d[i,0] = xn
                x_y_d[i,1] = yn
            else:
                x_y_d[i,0] = xm
                x_y_d[i,1] = ym
            x_y_d[i,2] = xn
            x_y_d[i,3] = yn
            if mask_base[yn,xn] == -1:
                x_y_d[i,4] = xn
                x_y_d[i,5] = yn
            else:
                x_y_d[i,4] = x_y_d[i,0]
                x_y_d[i,5] = x_y_d[i,1]
            x_y_d[i,6] = np.sqrt((x_y_d[i,4]-x_y_d[i,0])**2 + (x_y_d[i,5]-x_y_d[i,1])**2)
            x_y_d[i,7] = np.sqrt((x_y_d[i,4]-x0)**2 + (x_y_d[i,5]-y0)**2)
        zn = x_y_d[:,7]
        zn1 = np.roll(zn, shift=-1)
        dz = zn1 - zn
        dz_min = np.min(dz)
        dz_max = np.max(dz)
        if np.max(np.abs([dz_min, dz_max])) > 0.25*p.equivalent_diameter:
            d_cell, d_thr = regularize_cell_boundary(x_y_d[:,7], thr=0.5*p.equivalent_diameter)
        else:
            d_cell = x_y_d[:,7]
        d_cell = gradient_smoothing(d_cell, n_theta, l)
        x_cell = (np.round(x0+d_cell*np.cos(theta))).astype('int64')
        x_cell = x_cell.reshape(-1,1)
        x_cell = np.where(x_cell < 0, 0, x_cell)
        x_cell = np.where(x_cell >= w, w-1, x_cell)
        y_cell = (np.round(y0+d_cell*np.sin(theta))).astype('int64')
        y_cell = y_cell.reshape(-1,1)
        y_cell = np.where(y_cell < 0, 0, y_cell)
        y_cell = np.where(y_cell >= h, h-1, y_cell)
        hc1 = np.min(y_cell[:,0])
        hc2 = np.max(y_cell[:,0]) + 1
        wc1 = np.min(x_cell[:,0])
        wc2 = np.max(x_cell[:,0]) + 1
        h1 = np.min([hn1, hc1])
        h2 = np.max([hn2, hc2])
        w1 = np.min([wn1, wc1])
        w2 = np.max([wn2, wc2])
        cc = np.zeros((h2-h1,w2-w1), dtype='uint8')
        cc[p.coords[:,0]-h1, p.coords[:,1]-w1] = 1
        pts = np.concatenate((x_cell-wc1, y_cell-hc1), axis=1)
        cc = cv2.fillPoly(cc, pts=[pts], color=(1,1,1))
        cc = cc.astype('bool')
        cc = morphological_smoothing(cc, d=d)
        mask_allowed = np.logical_or(mask_cell_lab[h1:h2,w1:w2] == n, mask_cell_lab[h1:h2,w1:w2] == 0)
        cc = np.logical_and(mask_allowed, cc)
        mask_cell_lab[h1:h2,w1:w2][cc==True] = n
    return mask_cell_lab

###############################################################################
#    SEGMENTATION FUNCTIONS
###############################################################################
def nuclei_segmentation(intensity,
                        area_threshold=20,
                        convexity_threshold=4,
                        cell_marker_threshold=25,
                        dist_tr='GDT',
                        sigma0=3,
                        k0=1,
                        r0=5,
                        pct=[0.01, 0.01],
                        nk=[5, 10, 20, 40],
                        t0=0.5,
                        ternary_met='Argmax',
                        visualization=False):
    global_time = time.time()
    print('\nNUCLEI SEGMENTATION...\n')
    
    print(' 0. Preprocessing...')
    local_time = time.time()
    channel_names = ('Nuclei','Cell Membranes')
    img = normalize_img(intensity)
    if visualization:
        intensity2rgb(img, return_rgb=False)
    print('\t-processing time = {0:.2f} seconds'.format(time.time()-local_time))
    
    print(' 1. Computing of A Priori Probabilities...')
    local_time = time.time()
    p_em = calc_empirical_probabilities(img, sigma=sigma0)
    if visualization:
        for i in range(len(channel_names)):
            plot_img(p_em[:,:,i], tlt='{0}: A Priori Probability'.format(channel_names[i]))
    print('\t-processing time = {0:.2f} seconds'.format(time.time()-local_time))
    
    print(' 2. Computing of A Priori Global and Local (Binary) Masks...')
    local_time = time.time()
    mask_global_binary = calc_global_binary_mask(img, nk=nk, t0=t0)
    mask_local_binary = calc_local_binary_mask(img, r0=r0, k0=k0, smoothing=True)
    mask_local_binary, mask_global_binary = remove_background(mask_local_binary, mask_global_binary, p_em, pct)
    if visualization:
        for i in range(len(channel_names)):
            plot_img(mask_global_binary[:,:,i], tlt='{0}: A Priori Global Mask'.format(channel_names[i]))
            plot_img(mask_local_binary[:,:,i], tlt='{0}: A Priori Local Mask'.format(channel_names[i]))
    print('\t-processing time = {0:.2f} seconds'.format(time.time()-local_time))
    
    print(' 3. Computing of Likelihood...')
    local_time = time.time()
    likelihood = calc_likelihood(img, mask_local_binary, mask_global_binary)
    if visualization:
        plot_img(likelihood, tlt='Likelihood Function', cmp='bwr')
    print('\t-processing time = {0:.2f} seconds'.format(time.time()-local_time))
        
    print(' 4. Computing of A Posteriori Global and Local (Ternary) Masks...')
    local_time = time.time()
    p_local, p_global = calc_aposteriori_probabilities(p_em, mask_local_binary, likelihood)
    del p_em, mask_global_binary, mask_local_binary, likelihood
    mask_local_ternary = calc_ternary_mask(img, p_local, met=ternary_met)
    mask_global_ternary = calc_ternary_mask(img, p_global, met=ternary_met)
    print('\tclustering method: {0}'.format(ternary_met))
    if visualization:
        plot_img(mask_local_ternary, tlt='A Posteriori Local Mask ({0})'.format(ternary_met))
        plot_img(mask_global_ternary, tlt='A Posteriori Global Mask ({0})'.format(ternary_met))
    del p_local, p_global
    print('\t-processing time = {0:.2f} seconds'.format(time.time()-local_time))
    
    print(' 5. Convexity Analysis...\n\tformation of:\n'+
    '\t -initial Nuclei Segmentation (NS)\n\t -list of Nuclei Clusters (NC)\n\t -list of Small Objects (SO)')
    local_time = time.time()
    nuc_seg, nuc_clust, small_obj, nuclei_clusters_info = convexity_analysis(mask_global_ternary, area_thr=area_threshold, cp_thr=convexity_threshold)
    print('\t-processing time = {0:.2f} seconds'.format(time.time()-local_time))
    
    print(' 6. Processing of Nuclei Clusters and Small Objects...')
    local_time = time.time()
    nuc_seg, small_obj = process_clusters(img, mask_local_ternary, nuc_clust, nuc_seg, small_obj, nuclei_clusters_info, 
                                          cp_thr=convexity_threshold, area_thr=area_threshold, cy5_thr=cell_marker_threshold, dist_tr=dist_tr)
    nuc_seg = process_small_objects(small_obj, nuc_seg, nuclei_clusters_info, img.shape[0], img.shape[1])
    print('\t-processing time = {0:.2f} seconds'.format(time.time()-local_time))
    
    print(' 7. Shape Smoothing of Segmented Nuclei...')
    local_time = time.time()
    nuc_seg, nuc_seg_status = filter_nuclei_mask(nuc_seg)
    print('\t-processing time = {0:.2f} seconds'.format(time.time()-local_time))
    
    print('\nNUCLEI SEGMENTATION COMPLETED: processing time = {0:.2f} seconds'.format(time.time()-global_time))
    return nuc_seg, nuc_seg_status, mask_local_ternary, mask_global_ternary

def nuclei_cell_segmentation(intensity,
                             area_threshold=20,
                             convexity_threshold=4,
                             cell_marker_threshold=25,
                             dist_tr='GDT',
                             sigma0=3,
                             k0=1,
                             r0=5,
                             pct=[0.01, 0.01],
                             nk=[5, 10, 20, 40],
                             t0=0.5,
                             ternary_met='Argmax',
                             visualization=False,
                             area_ratio_threshold=0.65,
                             dilation_radius=9):
    """Performes unsupervised segmentation of nuclei and then cells successively.
    
    Parameters
    ----------
    intensity : 3D (height, width, channels) numpy array.
        Intensities of nuclei marker (channel index = 0) and cell membrane marker (channel index = 1).
    area_threshold : int (positive).
        The minimal possible area of a nucleus in pixels.
    convexity_threshold : int (positive)
        The largest allowable deviation (in pixels) of a boundary point from its convex hull for a convex component.
    cell_marker_threshold : int (positive).
        Defines whether a cell membrane marker is present in a cluster of nuclei.
    dist_tr : string, one of "DT" or "GDT".
        Defines geometrical distance transform (DT) or gradient distance transform (GDT) in the Virtual Cuts.
    sigma0 : float (positive).
        The standard deviation of the Gaussian filter.
    k0 : float (positive).
        The degree of smoothing of the Gradient Adaptive Filter.
    r0 : int (positive).
        The disk kernel radius of the local Otsu method.
    pct : list of two positive floats, [float (positive), float (positive)].
        The background a priori probability thresholds for every of two channels.
    nk : list of positive integers, [int (positive), int (positive), ...].
        The list of kernel sizes in the Local Mean Suppression Filter.
    t0 : float (positive).
        The intensity threshold in the Local Mean Suppression Filter.
    ternary_met : string, one of "Argmax" or "Kmeans".
        Defines the algorithm to compute a posteriori local and global masks.
    visualization : bool.
        Plots a priori probabilities, a posteriori local and global masks,
        contrast based likelihood function, and nuclei and cell segmentations.
    area_ratio_threshold : float (positive).
        The area threshold for a cell without a nucleus.
    dilation_radius : int (positive).
        The nucleus mask is morphologically dilated by this amount for cells without cell membrane marker expression.
    
    Returns
    -------
    nuc_seg : 2D (height, width) numpy 'int32' array.
        Mask of segmented nuclei.
    cel_seg : 2D (height, width) numpy 'int32' array.
        Mask of segmented cells.
    n_nuclei : int.
        Number of segmented nuclei.
    n_cells : int.
        Number of segmented cells."""
    global_time = time.time()
    nuc_seg, nuc_seg_status, mask_local_ternary, mask_global_ternary = nuclei_segmentation(intensity,
                                                                                           area_threshold=area_threshold,
                                                                                           convexity_threshold=convexity_threshold,
                                                                                           cell_marker_threshold=cell_marker_threshold,
                                                                                           dist_tr=dist_tr,
                                                                                           sigma0=sigma0,
                                                                                           k0=k0,
                                                                                           r0=r0,
                                                                                           pct=pct,
                                                                                           nk=nk,
                                                                                           t0=t0,
                                                                                           ternary_met=ternary_met,
                                                                                           visualization=visualization)
    print('\nCELL SEGMENTATION...\n')
    
    print(' 8. Formation of Initial Cell Segmentation...')
    local_time = time.time()
    cel_seg, nuclei_prop, nuclei, n_nuclei = initialize_cell_segmentation(nuc_seg)
    print('\tinitial number of segmented cells = {0}'.format(n_nuclei))
    print('\t-processing time = {0:.2f} seconds'.format(time.time()-local_time))
    
    print(' 9. Skeletonization of Cell Membrane Global Ternary Mask...')
    local_time = time.time()
    mask_membraines = mask_global_ternary[:,:,0] == 255
    area_threshold_cell = np.max([0.5*nuc_seg_status['area_avr'], nuc_seg_status['area_avr']-nuc_seg_status['area_std']])
    print('\tMinimal Area of a Cell without Nucleus = {0:.1f} pxs'.format(area_threshold_cell))
    mask_membraines_cuts, mask_skeleton = get_mask_cuts_mod(mask_membraines, area_threshold_cell)
    print('\t-processing time = {0:.2f} seconds'.format(time.time()-local_time))
    
    print('10. Euler Number Filtering...')
    local_time = time.time()
    cel_seg, nuclei, n_cells = euler_number_filter_mod(cel_seg, mask_membraines_cuts, nuclei, n_nuclei,
                                                       small_area=area_threshold_cell, area_ratio_thr=area_ratio_threshold)
    print('\tfinal number of segmented cells = {0}'.format(n_cells))
    print('\t-processing time = {0:.2f} seconds'.format(time.time()-local_time))
    
    print('11. Isolated Nuclei Filtering...')
    local_time = time.time()
    cel_seg, nuclei = isolated_nuclei_filter(cel_seg, mask_membraines, nuclei_prop, nuclei, u0=dilation_radius)
    print('\t-processing time = {0:.2f} seconds'.format(time.time()-local_time))
    
    print('12. Threshold Filtering and Smoothing of Cell Segmentation...')
    local_time = time.time()
    cel_seg = threshold_filter(cel_seg, mask_skeleton, nuclei_prop, nuclei)
    print('\t-processing time = {0:.2f} seconds'.format(time.time()-local_time))
    
    if visualization:
        print('13. Plotting of Nuclei and Cell Segmentations...')
        local_time = time.time()
        img = normalize_img(intensity)
        img = intensity2rgb(img, return_rgb=True)
        plot_img(mark_boundaries(img, nuc_seg, color=(1,1,1)), tlt='Nuclei Segmentation: number of nuclei = {0}'.format(n_nuclei))
        plot_img(mark_boundaries(img, cel_seg, color=(0,1,0)), tlt='Cell Segmentation: number of cells = {0}'.format(n_cells))
        print('\t-processing time = {0:.2f} seconds'.format(time.time()-local_time))
        
    print('\nCELL SEGMENTATION COMPLETED: total processing time = {0:.2f} seconds'.format(time.time()-global_time))
    return nuc_seg, cel_seg, n_nuclei, n_cells
