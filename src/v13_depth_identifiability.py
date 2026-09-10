"""Audit what the supplied monocular data can actually identify in 3D.

Exports conditional ray intervals with radius provenance, and constructs two
complete trajectories which have identical image projections and satisfy the
current anatomical-radius and speed proxies, but differ by 6 mm in space.
This is a reproducible counterexample to a uniquely recovered 3D trajectory.
No radius is enlarged and no supplied XYZ point is moved to make it fit.
"""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from config import DATA, OUT, FPS, imread_u
from v11_realtime_tube import TubeGeometry


def remeasure_width(g):
    """Independent image-width audit at the CURRENT camera projection.

    Only accept a dye segment enclosing the projected centreline with both
    boundaries visible. Off-centre axes, large junctions and transparent lumen
    remain unknown; no radius is inferred from where the robot happens to be.
    This remains a projected-width proxy, not a 3D surface measurement.
    """
    bg=imread_u(OUT/'background_median.png')
    import cv2
    hsv=cv2.cvtColor(bg,cv2.COLOR_BGR2HSV).astype(float)
    hue=np.minimum(hsv[:,:,0],180-hsv[:,:,0])
    orange=np.clip(1-hue/25,0,1)*np.clip(hsv[:,:,1]/60,0,1)*np.clip(hsv[:,:,2]/60,0,1)
    orange=cv2.GaussianBlur(orange,(3,3),0)
    offsets=np.arange(-100,100.01,.25)
    width=np.full(len(g.s),np.nan)
    bias=np.full(len(g.s),np.nan)
    for i in range(len(g.s)):
        p=g.P[i]+offsets[:,None]*g.normal[i]
        profile=map_coordinates(orange,[p[:,1],p[:,0]],order=1,mode='constant')
        centre=len(offsets)//2
        if profile[centre]<.5: continue
        a=b=centre
        while a>0 and profile[a-1]>=.5: a-=1
        while b<len(offsets)-1 and profile[b+1]>=.5: b+=1
        if a==0 or b==len(offsets)-1: continue
        lo=offsets[a-1]+.25*(.5-profile[a-1])/(profile[a]-profile[a-1])
        hi=offsets[b]+.25*(.5-profile[b])/(profile[b+1]-profile[b])
        if hi-lo<5 or hi-lo>100: continue
        width[i]=hi-lo
        bias[i]=(hi+lo)/2
    return width,bias


def ray_sphere_interval(origin, ray, centre, radius):
    """Positive ray depth interval through a local radius sphere, or None."""
    distance=centre-origin
    midpoint=float(np.dot(distance,ray))
    radial2=float(np.dot(distance,distance)-midpoint**2)
    disc=radius**2-radial2
    if disc < -1e-9: return None
    half=np.sqrt(max(disc,0.))
    if midpoint+half<0: return None
    return max(0.,midpoint-half),midpoint+half


def main():
    g=TubeGeometry()
    z=np.load(DATA/'v12_aligned.npz')
    X=z['X'];s=z['s'];uv=z['UV']
    valid=np.isfinite(X).all(axis=1)
    centre=np.stack([np.interp(s,g.s,g.C[:,j]) for j in range(3)],axis=1)
    # Provenance now follows the radius the pipeline actually uses (V11's
    # TubeGeometry, policy C): measured on the current projection, or the 3 mm
    # floor. The legacy L1 widths are kept only for the conflict bookkeeping.
    legacy_width=np.load(DATA/'l1_size_cue.npz')['width_px']
    indices=np.clip(np.rint(np.nan_to_num(s)/.25).astype(int),0,len(g.s)-1)
    width_known=g.radius_measured[indices] & valid
    conflicts=z['status']=='geometry_conflict'
    new_width,new_bias=remeasure_width(g)

    origin=-g.R.T@g.t
    rays=X-origin
    rays/=np.linalg.norm(rays,axis=1)[:,None]
    amplitude=np.zeros(len(X))
    amplitude[200:301]=3.*np.sin(np.linspace(0,np.pi,101))**2
    plus=X+amplitude[:,None]*rays
    minus=X-amplitude[:,None]*rays
    changed=amplitude>1e-8
    pixel_difference=np.linalg.norm(g.project(plus)-g.project(minus),axis=1)
    separation=np.linalg.norm(plus-minus,axis=1)
    speed_plus=np.linalg.norm(np.diff(plus,axis=0),axis=1)*FPS
    speed_minus=np.linalg.norm(np.diff(minus,axis=0),axis=1)*FPS
    excess_plus=np.linalg.norm(plus-centre,axis=1)-z['radius']
    excess_minus=np.linalg.norm(minus-centre,axis=1)-z['radius']
    if not width_known[changed].all():
        # Under policy C nothing is imputed; an unmeasured frame carries the
        # stated 3 mm floor. Report it rather than abort - the JSON records it.
        print(f'[V13] note: {int((~width_known[changed]).sum())} of the perturbed frames sit on the 3 mm floor radius, not a measured width')
    assert np.nanmax(pixel_difference)<1e-8
    assert np.nanmax(excess_plus)<1e-7 and np.nanmax(excess_minus)<1e-7
    assert np.nanmax(speed_plus)<60+1e-6 and np.nanmax(speed_minus)<60+1e-6

    # Export the feasible interval of the observed ray at the selected s under
    # the inherited spherical radius proxy, only when its source was measured.
    # Absence of an interval at this s does not exclude all other branches.
    lower=np.full(len(X),np.nan);upper=lower.copy()
    low_xyz=np.full_like(X,np.nan);high_xyz=low_xyz.copy()
    provenance=np.full(len(X),'missing_detection',dtype='<U40')
    for k in np.where(valid)[0]:
        if not width_known[k]:
            provenance[k]='radius_unmeasured_depth_unbounded'
            continue
        ray=np.r_[(uv[k]-g.cxy)/g.f,1.]@g.R
        ray/=np.linalg.norm(ray)
        interval=ray_sphere_interval(origin,ray,centre[k],z['radius'][k])
        if interval is None:
            provenance[k]='no_interval_at_selected_s'
            continue
        lower[k],upper[k]=interval
        low_xyz[k]=origin+lower[k]*ray
        high_xyz[k]=origin+upper[k]*ray
        provenance[k]='interval_conditional_on_radius_proxy'
    has_interval=np.isfinite(lower)
    report={
        'input_geometry':'71 XYZ centreline points; no lumen surface or independently measured camera calibration',
        'legacy_geometry_conflict_count':int(conflicts.sum()),
        'conflicts_with_imputed_not_measured_width':int(np.sum(conflicts & ~width_known)),
        'conflicts_with_legacy_width_measurement':int(np.sum(conflicts & width_known)),
        'frames_with_no_legacy_width_measurement':int(np.sum(valid & ~width_known)),
        'current_camera_width_measurement_count':int(np.isfinite(new_width).sum()),
        'current_camera_junction_width_measurement_count':int(np.sum(np.isfinite(new_width)&(g.s>109)&(g.s<119))),
        'counterexample':{
            'modified_frames':[200,300],
            'uses_only_nonimputed_radius_frames':bool(width_known[changed].all()),
            'maximum_3d_separation_mm':float(np.nanmax(separation)),
            'maximum_projection_difference_px':float(np.nanmax(pixel_difference)),
            'maximum_speed_plus_mm_s':float(np.nanmax(speed_plus)),
            'maximum_speed_minus_mm_s':float(np.nanmax(speed_minus)),
            'maximum_radius_excess_plus_mm':float(np.nanmax(excess_plus)),
            'maximum_radius_excess_minus_mm':float(np.nanmax(excess_minus))},
        'conditional_ray_intervals':{
            'frames':int(has_interval.sum()),
            'median_length_mm':float(np.median((upper-lower)[has_interval])),
            'p90_length_mm':float(np.percentile((upper-lower)[has_interval],90)),
            'limitations':'Conditional on selected s, fitted camera and spherical radius proxy. Not calibrated confidence intervals or true vessel bounds.'},
        'conclusion':'These data and current constraints do not uniquely determine the 3D trajectory. Motion regularity selects a representative; it does not supply measured depth.'}
    np.savez(DATA/'v13_depth_audit.npz',X_reference=X,X_plus=plus,X_minus=minus,
             separation=separation,projection_difference=pixel_difference,
             interval_low_xyz=low_xyz,interval_high_xyz=high_xyz,
             radius_known=width_known,provenance=provenance,
             current_width_px=new_width,current_width_axis_bias_px=new_bias)
    (OUT/'v13_depth_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (OUT/'v13_3d_positions_with_provenance.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['frame','time_s','x_estimate','y_estimate','z_estimate',
                         'x_near','y_near','z_near','x_far','y_far','z_far',
                         'legacy_radius_mm','radius_measured','depth_status'])
        for k in range(len(X)):
            writer.writerow([k,k/FPS,*X[k],*low_xyz[k],*high_xyz[k],z['radius'][k],bool(width_known[k]),provenance[k]])
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    window=np.arange(200,301)
    a=axes[0];qp=g.project(plus);qm=g.project(minus)
    a.plot(qp[window,0],qp[window,1],lw=3,label='3D solution A')
    a.plot(qm[window,0],qm[window,1],'--',lw=1.5,label='3D solution B')
    a.invert_yaxis();a.set_aspect('equal');a.legend();a.set_title('Identical image projections')
    a.set_xlabel('u (px)');a.set_ylabel('v (px)')
    a=axes[1];a.plot(window/FPS,separation[window],color='#b33b29',lw=2)
    a.set_xlabel('time (s)');a.set_ylabel('3D distance between A and B (mm)')
    a.set_title('Up to 6 mm apart in 3D');a.grid(alpha=.2)
    a=axes[2];a.plot(window[1:]/FPS,speed_plus[window[:-1]],label='A')
    a.plot(window[1:]/FPS,speed_minus[window[:-1]],'--',label='B')
    a.axhline(60,color='r',ls=':');a.set_ylim(0,65);a.legend()
    a.set_xlabel('time (s)');a.set_ylabel('speed (mm/s)')
    a.set_title('Both obey radius and speed constraints')
    fig.suptitle('Depth ambiguity remains even with a fixed camera and the current tube model')
    fig.tight_layout();fig.savefig(OUT/'figs'/'v13_depth_counterexample.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
