import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from .source_pos import gauss
from ..lib import util
from ..lib.plots import figure_filetype
import scipy.stats as stats


def lc_nodriftcorr(meta, wave_1d, optspec, optmask=None):
    '''Plot a 2D light curve without drift correction. (Fig 3101)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    wave_1d : ndarray
        Wavelength array with trimmed edges depending on xwindow and ywindow
        which have been set in the S3 ecf
    optspec : ndarray
        The optimally extracted spectrum.
    optmask : ndarray (1D), optional
        A mask array to use if optspec is not a masked array. Defaults to None
        in which case only the invalid values of optspec will be masked.

    Returns
    -------
    None
    '''
    normspec = util.normalize_spectrum(meta, optspec, optmask=optmask)
    wmin = wave_1d.min()
    wmax = wave_1d.max()
    vmin = 0.97
    vmax = 1.03

    plt.figure(3101, figsize=(8, 8))
    plt.clf()
    plt.imshow(normspec, origin='lower', aspect='auto',
               extent=[wmin, wmax, 0, meta.n_int], vmin=vmin, vmax=vmax,
               cmap=plt.cm.RdYlBu_r)
    plt.title(f"MAD = {int(np.round(meta.mad_s3, 0))} ppm")
    plt.ylabel('Integration Number')
    plt.xlabel(r'Wavelength ($\mu m$)')
    plt.colorbar(label='Normalized Flux')
    plt.tight_layout()
    fname = f'figs{os.sep}fig3101-2D_LC'+figure_filetype
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def image_and_background(data, meta, log, m):
    '''Make image+background plot. (Figs 3301)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    m : int
        The file number.

    Returns
    -------
    None
    '''
    log.writelog('  Creating figures for background subtraction',
                 mute=(not meta.verbose))

    intstart, subdata, submask, subbg = (data.attrs['intstart'],
                                         data.flux.values,
                                         data.mask.values,
                                         data.bg.values)
    xmin, xmax = data.flux.x.min().values, data.flux.x.max().values
    ymin, ymax = data.flux.y.min().values, data.flux.y.max().values

    iterfn = range(meta.n_int)
    if meta.verbose:
        iterfn = tqdm(iterfn)
    for n in iterfn:
        plt.figure(3301, figsize=(8, 8))
        plt.clf()
        plt.suptitle(f'Integration {intstart + n}')
        plt.subplot(211)
        plt.title('Background-Subtracted Flux')
        max = np.max(subdata[n] * submask[n])
        plt.imshow(subdata[n]*submask[n], origin='lower', aspect='auto',
                   vmin=0, vmax=max/10, extent=[xmin, xmax, ymin, ymax])
        plt.colorbar()
        plt.ylabel('Detector Pixel Position')
        plt.subplot(212)
        plt.title('Subtracted Background')
        median = np.median(subbg[n])
        std = np.std(subbg[n])
        plt.imshow(subbg[n], origin='lower', aspect='auto', vmin=median-3*std,
                   vmax=median+3*std, extent=[xmin, xmax, ymin, ymax])
        plt.colorbar()
        plt.ylabel('Detector Pixel Position')
        plt.xlabel('Detector Pixel Position')
        plt.tight_layout()
        file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))
                                       + 1))
        int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
        fname = (f'figs{os.sep}fig3301_file{file_number}_int{int_number}' +
                 '_ImageAndBackground'+figure_filetype)
        plt.savefig(meta.outputdir+fname, dpi=300)
        if not meta.hide_plots:
            plt.pause(0.2)


def drift_2D(data, meta):
    '''Plot the fitted 2D drift. (Fig 3102)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    '''
    plt.figure(3102, figsize=(8, 6))
    plt.clf()
    plt.subplot(211)
    for p in range(2):
        iscans = np.where(data.scandir.values == p)[0]
        plt.plot(iscans, data.drift2D[iscans, 1], '.')
    plt.ylabel(f'Drift Along y ({data.drift2D.drift_units})')
    plt.subplot(212)
    for p in range(2):
        iscans = np.where(data.scandir.values == p)[0]
        plt.plot(iscans, data.drift2D[iscans, 0], '.')
    plt.ylabel(f'Drift Along x ({data.drift2D.drift_units})')
    plt.xlabel('Frame Number')
    plt.tight_layout()
    fname = f'figs{os.sep}fig3102_Drift2D{figure_filetype}'
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def optimal_spectrum(data, meta, n, m):
    '''Make optimal spectrum plot. (Figs 3302)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    n : int
        The integration number.
    m : int
        The file number.

    Returns
    -------
    None
    '''
    intstart, stdspec, optspec, opterr = (data.attrs['intstart'],
                                          data.stdspec.values,
                                          data.optspec.values,
                                          data.opterr.values)

    plt.figure(3302)
    plt.clf()
    plt.suptitle(f'1D Spectrum - Integration {intstart + n}')
    plt.semilogy(data.stdspec.x.values, stdspec[n], '-', color='C1',
                 label='Standard Spec')
    plt.errorbar(data.stdspec.x.values, optspec[n], yerr=opterr[n], fmt='-',
                 color='C2', ecolor='C2', label='Optimal Spec')
    plt.ylabel('Flux')
    plt.xlabel('Detector Pixel Position')
    plt.legend(loc='best')
    plt.tight_layout()
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    fname = (f'figs{os.sep}fig3302_file{file_number}_int{int_number}' +
             '_Spectrum'+figure_filetype)
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def source_position(meta, x_dim, pos_max, m,
                    isgauss=False, x=None, y=None, popt=None,
                    isFWM=False, y_pixels=None, sum_row=None, y_pos=None):
    '''Plot source position for MIRI data. (Figs 3303)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    x_dim : int
        The number of pixels in the y-direction in the image.
    pos_max : float
        The brightest row.
    m : int
        The file number.
    isgauss : bool; optional
        Used a guassian centring method.
    x : type; optional
        Unused.
    y : type; optional
        Unused.
    popt : list; optional
        The fitted Gaussian terms.
    isFWM : bool; optional
        Used a flux-weighted mean centring method.
    y_pixels : 1darray; optional
        The indices of the y-pixels.
    sum_row : 1darray; optional
        The sum over each row.
    y_pos : float; optional
        The FWM central position of the star.

    Returns
    -------
    None

    Notes
    -----
    History:

    - 2021-07-14: Sebastian Zieba
        Initial version.
    - Oct 15, 2021: Taylor Bell
        Tidied up the code a bit to reduce repeated code.
    '''
    plt.figure(3303)
    plt.clf()
    plt.plot(y_pixels, sum_row, 'o', label='Data')
    if isgauss:
        x_gaussian = np.linspace(0, x_dim, 500)
        gaussian = gauss(x_gaussian, *popt)
        plt.plot(x_gaussian, gaussian, '-', label='Gaussian Fit')
        plt.axvline(popt[1], ls=':', label='Gaussian Center', c='C2')
        plt.xlim(pos_max-meta.spec_hw, pos_max+meta.spec_hw)
    elif isFWM:
        plt.axvline(y_pos, ls='-', label='Weighted Row')
    plt.axvline(pos_max, ls='--', label='Brightest Row', c='C3')
    plt.ylabel('Row Flux')
    plt.xlabel('Row Pixel Position')
    plt.legend()
    plt.tight_layout()
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    fname = ('figs'+os.sep+f'fig3303_file{file_number}_source_pos' +
             figure_filetype)
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def profile(meta, profile, submask, n, m):
    '''Plot weighting profile from optimal spectral extraction routine. (Figs 3304)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    profile : ndarray
        Fitted profile in the same shape as the data array.
    submask : ndarray
        Outlier mask.
    n : int
        The current integration number.
    m : int
        The file number.

    Returns
    -------
    None
    '''
    profile = np.ma.masked_invalid(profile)
    submask = np.ma.masked_invalid(submask)
    mask = np.logical_or(np.ma.getmaskarray(profile),
                         np.ma.getmaskarray(submask))
    profile = np.ma.masked_where(mask, profile)
    submask = np.ma.masked_where(mask, submask)
    vmax = 0.05*np.ma.max(profile*submask)
    vmin = np.ma.min(profile*submask)
    plt.figure(3304)
    plt.clf()
    plt.suptitle(f"Profile - Integration {n}")
    plt.imshow(profile*submask, aspect='auto', origin='lower',
               vmax=vmax, vmin=vmin)
    plt.ylabel('Relative Pixel Postion')
    plt.xlabel('Relative Pixel Position')
    plt.tight_layout()
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    fname = (f'figs{os.sep}fig3304_file{file_number}_int{int_number}_Profile' +
             figure_filetype)
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def subdata(meta, i, n, m, subdata, submask, expected, loc):
    '''Show 1D view of profile for each column. (Figs 3501)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    i : int
        The column number.
    n : int
        The current integration number.
    m : int
        The file number.
    subdata : ndarray
        Background subtracted data.
    submask : ndarray
        Outlier mask.
    expected : ndarray
        Expected profile
    loc : ndarray
        Location of worst outliers.

    Returns
    -------
    None
    '''
    ny, nx = subdata.shape
    plt.figure(3501)
    plt.clf()
    plt.suptitle(f'Integration {n}, Columns {i}/{nx}')
    plt.plot(np.arange(ny)[np.where(submask[:, i])[0]],
             subdata[np.where(submask[:, i])[0], i], 'bo')
    plt.plot(np.arange(ny)[np.where(submask[:, i])[0]],
             expected[np.where(submask[:, i])[0], i], 'g-')
    plt.plot((loc[i]), (subdata[loc[i], i]), 'ro')
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    col_number = str(i).zfill(int(np.floor(np.log10(nx))+1))
    fname = (f'figs{os.sep}fig3501_file{file_number}_int{int_number}' +
             f'_col{col_number}_subdata'+figure_filetype)
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.1)

#Photometry

def phot_lc(meta, data):
    """
    Plots the flux as a function of time.
    """
    plt.figure(3103)
    plt.clf()
    plt.suptitle('Photometric light curve')
    plt.errorbar(data.time, data['aplev'], yerr=data['aperr'], c='k', fmt='.')
    plt.ylabel('Flux')
    plt.xlabel('Time')
    plt.tight_layout()
    fname = (f'figs{os.sep}fig3103-1D_LC' + figure_filetype)
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def phot_bg(meta, data):
    """
    Plots the background flux as a function of time.
    """
    plt.figure(3305)
    plt.clf()
    plt.suptitle('Photometric background light curve')
    plt.errorbar(data.time, data['skylev'], yerr=data['skyerr'], c='k', fmt='.')
    plt.ylabel('Flux')
    plt.xlabel('Time')
    plt.tight_layout()
    fname = (f'figs{os.sep}fig3305-1D_BG_LC' + figure_filetype)
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def phot_centroid(meta, data):
    """
    Plots the (x, y) centroids and (sx, sy) the Gaussian 1-sigma half-widths as a function of time.
    """
    plt.figure(3306)
    plt.clf()
    plt.suptitle('Centroid positions over time')

    plt.subplot(411)
    plt.plot(data.time, data.centroid_x-np.mean(data.centroid_x))
    plt.ylabel('Delta x')

    plt.subplot(412)
    plt.plot(data.time, data.centroid_y-np.mean(data.centroid_y))
    plt.ylabel('Delta y')

    plt.subplot(413)
    plt.plot(data.time, data.centroid_sy-np.mean(data.centroid_sx))
    plt.ylabel('Delta sx')

    plt.subplot(414)
    plt.plot(data.time, data.centroid_sy-np.mean(data.centroid_sy))
    plt.ylabel('Delta sy')
    plt.xlabel('Time')
    plt.tight_layout()
    fname = (f'figs{os.sep}fig3306-Centroid' + figure_filetype)
    plt.savefig(meta.outputdir + fname, dpi=250)
    if not meta.hide_plots:
        plt.pause(0.2)


def phot_centroid_fgc(img, x, y, sx, sy, i, m, meta):
    plt.figure(3502)
    plt.clf()
    plt.suptitle('Centroid gaussian fit')
    fig, ax = plt.subplots(2,2, figsize=(8,8))
    fig.delaxes(ax[1,1])
    ax[0,0].imshow(img, vmax=5e3, origin='lower', aspect='auto')

    ax[1,0].plot(range(len(np.sum(img, axis=0))), np.sum(img, axis=0))
    x_plot = np.linspace(0, len(np.sum(img, axis=0)))
    ax[1,0].plot(x_plot, stats.norm.pdf(x_plot, x, sx)/np.max(stats.norm.pdf(x_plot, x, sx))*np.max(np.sum(img, axis=0)))

    ax[0,1].plot(np.sum(img, axis=1), range(len(np.sum(img, axis=1))))
    y_plot = np.linspace(0, len(np.sum(img, axis=1)))
    ax[0,1].plot(stats.norm.pdf(y_plot, y, sy)/np.max(stats.norm.pdf(y_plot, y, sy))*np.max(np.sum(img, axis=1)), y_plot)

    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(i).zfill(int(np.floor(np.log10(meta.n_int))+1))
    plt.tight_layout()
    fname = (f'figs{os.sep}fig3502_file{file_number}_int{int_number}_centroid_fgc' + figure_filetype)
    plt.savefig(meta.outputdir + fname, dpi=250)
    if not meta.hide_plots:
        plt.pause(0.2)


def phot_2D_frame(meta, m, i, data):
    """
    Plots the 2D frame together with the centroid position and apertures.
    """
    plt.figure(3307)
    plt.clf()
    plt.suptitle('2D frame with centroid and apertures')
    flux, centroid_x, centroid_y = data.flux[i], data.centroid_x[i], data.centroid_y[i]
    plt.imshow(flux, vmax=5e3, origin='lower')
    plt.scatter(centroid_x, centroid_y, marker='x', s=25, c='r', label='centroid')
    #alphas = np.zeros(image.shape)
    #alphas[np.where(skyann == True)] = 0.9
    #plt.imshow(~skyann, origin='lower', alpha=alphas, cmap='magma')
    #alphas = np.zeros(image.shape)
    #alphas[np.where(apmask == True)] = 0.4
    circle1 = plt.Circle((centroid_x, centroid_y), meta.photap, color='r', fill=False, lw=3, alpha=0.7, label='target aperture')
    circle2 = plt.Circle((centroid_x, centroid_y), meta.skyin, color='w', fill=False, lw=4, alpha=0.8, label='sky aperture')
    circle3 = plt.Circle((centroid_x, centroid_y), meta.skyout, color='w', fill=False, lw=4, alpha=0.8)
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)
    plt.gca().add_patch(circle3)
    #plt.imshow(~apmask, origin='lower', alpha=alphas)
    plt.xlim(0, flux.shape[1])
    plt.ylim(0, flux.shape[0])
    plt.xlabel('x pixels')
    plt.ylabel('y pixels')
    plt.legend()
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(i).zfill(int(np.floor(np.log10(meta.n_int))+1))
    plt.tight_layout()
    fname = (f'figs{os.sep}fig3307_file{file_number}_int{int_number}_2D_frame' + figure_filetype)
    plt.savefig(meta.outputdir + fname, dpi=250, tight_layout=True)
    if not meta.hide_plots:
        plt.pause(0.2)


def phot_npix(meta, data):
    """
    Plots the (x, y) centroids and (sx, sy) the Gaussian 1-sigma half-widths as a function of time.
    """
    plt.figure(3308)
    plt.clf()
    plt.suptitle('Aperture sizes over time')
    plt.subplot(211)
    plt.plot(range(len(data.nappix)), data.nappix)
    plt.xlabel('nappix')
    plt.subplot(212)
    plt.plot(range(len(data.nskypix)), data.nskypix)
    plt.xlabel('nappix')
    plt.legend()
    plt.xlabel('Time')
    plt.tight_layout()
    fname = (f'figs{os.sep}fig3308_aperture_size' + figure_filetype)
    plt.savefig(meta.outputdir + fname, dpi=250)
    if not meta.hide_plots:
        plt.pause(0.2)
