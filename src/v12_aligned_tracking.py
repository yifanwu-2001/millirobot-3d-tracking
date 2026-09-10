"""Stabilised, time-aligned robot overlay with explicit 3D model conflicts.

Yellow = measured robot track, cyan dashed = supplied anatomical centreline.
The visible 2D motion is retained. Missing detections are shown as missing;
points incompatible with the assumed tube do not get a fabricated 3D fix.
V11 supplies a delayed branch prior. Within that branch, continuous arc length
is refined using distance from the observed viewing ray to the centreline.

Run: python src/v12_aligned_tracking.py [--render]
Requires V11 setup/branch results and the original video; writes new V12 files.
"""
import argparse
import csv
import json
import time
from collections import deque

import cv2
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.transform import Rotation

from config import DATA, OUT, VIDEO, FPS, W, H, imwrite_u
from v11_realtime_tube import TubeGeometry


class BackgroundStabilizer:
    """Register each incoming frame to a fixed background without drift."""
    def __init__(self, frame, projected_route):
        self.reference = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.full_like(self.reference, 255)
        mask[:30] = 0; mask[-25:] = 0
        mask[:, :25] = 0; mask[:, -25:] = 0
        # Setup anatomy, not future observations, excludes moving robot pixels.
        cv2.polylines(mask, [np.rint(projected_route).astype(np.int32)],
                      False, 0, 130)
        self.points = cv2.goodFeaturesToTrack(self.reference, 400, .02, 12, mask=mask)
        if self.points is None or len(self.points) < 20:
            raise ValueError('Insufficient fixed background landmarks')
        self.last = np.eye(2, 3)
        self.heldout_before = np.nan
        self.heldout_after = np.nan

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        q, status, _ = cv2.calcOpticalFlowPyrLK(self.reference, gray, self.points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 25, .01))
        p = self.points[status.ravel() == 1, 0]
        q = q[status.ravel() == 1, 0]
        if len(p) < 20:
            return cv2.invertAffineTransform(self.last), np.nan, False
        train = np.arange(len(p)) % 4 != 0
        A, inliers = cv2.estimateAffinePartial2D(p[train], q[train], method=cv2.RANSAC,
                                              ransacReprojThreshold=1.5)
        valid = A is not None and np.mean(inliers) > .65
        if valid:
            self.last = A
            residual = np.median(np.linalg.norm(p[train] @ A[:, :2].T + A[:, 2]-q[train],
                                                axis=1)[inliers.ravel() == 1])
        else:
            residual = np.nan
        inverse = cv2.invertAffineTransform(self.last)
        self.heldout_before = np.median(np.linalg.norm(q[~train]-p[~train],axis=1))
        self.heldout_after = np.median(np.linalg.norm(transform_uv(q[~train],inverse)-p[~train],axis=1))
        return inverse, float(residual), bool(valid)


def transform_uv(uv, affine):
    return np.asarray(uv) @ affine[:, :2].T + affine[:, 2]


class VisibleRobotFilter:
    """A causal adaptive smoother which retains fast visible robot movement."""
    def __init__(self):
        self.position = None
        self.previous_raw = None
        self.velocity = np.zeros(2)
        self.gap = 0

    def update(self, uv):
        valid = np.isfinite(uv).all()
        if valid and self.position is not None:
            prediction = self.position + self.velocity / FPS
            valid = np.linalg.norm(uv-prediction) <= min(150., 65.+8*self.gap)
        if not valid:
            self.gap += 1
            self.velocity *= .75
            # Do not turn a prediction through an occlusion into a measurement.
            return np.full(2, np.nan), False
        if self.position is None or self.gap:
            self.position = uv.copy()
            self.velocity[:] = 0
        else:
            derivative = (uv-self.previous_raw)*FPS
            self.velocity = .6*self.velocity + .4*derivative
            cutoff_hz = 3. + .06*np.linalg.norm(self.velocity)
            alpha = 1. / (1. + FPS/(2*np.pi*cutoff_hz))
            self.position += alpha*(uv-self.position)
        self.previous_raw = uv.copy()
        self.gap = 0
        return self.position.copy(), True


class RayTubeRefiner:
    """Refine inside the selected branch; preserve XYZ anatomy and radius.

    A viewing ray supplies exact reprojection. The closest point on this ray
    to C(s) is selected as a representative, NOT as observable depth truth.
    When its radial distance exceeds the supplied radius proxy, the constrained
    estimate stops at the bound and the frame is explicitly marked conflict.
    """
    def __init__(self, geometry):
        self.g = geometry
        self.origin = -geometry.R.T @ geometry.t
        self.previous_s = None
        self.previous_x = None

    def centre(self, s):
        return np.array([np.interp(s, self.g.s, self.g.C[:,j]) for j in range(3)])

    def update(self, uv, branch_s):
        if not np.isfinite(uv).all():
            self.previous_s = None
            self.previous_x = None
            return dict(s=np.nan, X=np.full(3,np.nan), Q=np.full(2,np.nan),
                        radial=np.nan, radius=np.nan, status='missing_detection')
        ray = np.r_[(uv-self.g.cxy)/self.g.f, 1.] @ self.g.R
        ray /= np.linalg.norm(ray)
        lower, upper = max(0.,branch_s-10.), min(self.g.s[-1],branch_s+10.)
        if self.previous_s is not None:
            lower=max(lower,self.previous_s-60./FPS)
            upper=min(upper,self.previous_s+60./FPS)
        if upper <= lower:
            # Contradictory branch prior: explicitly reacquire, no hidden jump.
            return dict(s=np.nan, X=np.full(3,np.nan), Q=np.full(2,np.nan),
                        radial=np.nan, radius=np.nan, status='branch_conflict')

        def cost(s):
            c=self.centre(s)
            x=self.origin+np.dot(c-self.origin,ray)*ray
            return np.dot(x-c,x-c)+.015*(s-branch_s)**2

        # The local projected curve can bend back. Bracket its best grid
        # interval first, rather than assume a unimodal 20 mm interval.
        grid=np.linspace(lower,upper,max(5,int((upper-lower)/.25)+1))
        j=int(np.argmin([cost(s) for s in grid]))
        a,b=grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]
        sol=minimize_scalar(cost,bounds=(a,b),method='bounded',options={'xatol':1e-5})
        candidates=[lower,upper,float(sol.x)]
        s=min(candidates,key=cost)
        self.previous_s=s
        c=self.centre(s)
        x=self.origin+np.dot(c-self.origin,ray)*ray
        d=np.linalg.norm(x-c)
        radius=float(np.interp(s,self.g.s,self.g.radius))
        status='within_radius_proxy' if d <= radius else 'geometry_conflict'
        constrained=c+(x-c)*min(1.,radius/max(d,1e-12))
        if self.previous_x is not None:
            previous = self.previous_x
            step=60./FPS
            jump=np.linalg.norm(constrained-previous)
            if jump > step:
                limited=previous+(constrained-previous)*step/jump
                if np.linalg.norm(limited-c) > radius+1e-8:
                    # Closest feasible point in the intersection of the motion
                    # ball and anatomical radius ball; never move only the 2D
                    # marker and leave its 3D counterpart inconsistent.
                    target=constrained.copy()
                    sol=minimize(lambda y: np.dot(y-target,y-target),limited,
                        method='SLSQP',constraints=[
                            {'type':'ineq','fun':lambda y: step**2-np.dot(y-previous,y-previous)},
                            {'type':'ineq','fun':lambda y: radius**2-np.dot(y-c,y-c)}],
                        options={'ftol':1e-10,'maxiter':80})
                    if (not sol.success or np.linalg.norm(sol.x-previous)>step+1e-5
                            or np.linalg.norm(sol.x-c)>radius+1e-5):
                        self.previous_s=None;self.previous_x=None
                        return dict(s=np.nan,X=np.full(3,np.nan),Q=np.full(2,np.nan),
                                    radial=d,radius=radius,status='motion_geometry_conflict')
                    limited=sol.x
                constrained=limited
                if status=='within_radius_proxy': status='motion_limited'
        self.previous_x=constrained.copy()
        return dict(s=s,X=constrained,Q=self.g.project(constrained),
                    radial=float(d),radius=radius,status=status)


def analyse():
    cv2.setNumThreads(1)
    cv2.setRNGSeed(12)
    g=TubeGeometry()
    raw=np.load(DATA/'track2d.npz')['uv']
    branch=np.load(DATA/'v11_realtime.npz')
    cap=cv2.VideoCapture(str(VIDEO))
    ok,first=cap.read()
    if not ok: raise RuntimeError('Source video cannot be decoded')
    stabilizer=BackgroundStabilizer(first,g.P)
    detector=VisibleRobotFilter()
    refiner=RayTubeRefiner(g)
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    transforms=[]; observations=[]; tracks=[]; accepted=[]; results=[]
    landmarks=[]; registration_ok=[]; times=[]; heldout_before=[];heldout_after=[]
    # The V11 prior was calibrated in the source coordinate system; using
    # frame-zero coordinates changes it by only the measured background warp.
    for k,u in enumerate(raw):
        ok,frame=cap.read()
        if not ok: raise RuntimeError(f'Cannot decode frame {k}')
        t=time.perf_counter()
        A,residual,reg_ok=stabilizer.update(frame)
        stable_raw=transform_uv(u,A)
        tracked,valid=detector.update(stable_raw)
        # Convert the delayed branch's estimated image point into reference
        # coordinates too; the 3D setup remains the fixed calibrated model.
        result=refiner.update(tracked,branch['s'][k])
        times.append((time.perf_counter()-t)*1000)
        transforms.append(A); observations.append(stable_raw); tracks.append(tracked)
        accepted.append(valid); results.append(result)
        landmarks.append(residual);registration_ok.append(reg_ok)
        heldout_before.append(stabilizer.heldout_before);heldout_after.append(stabilizer.heldout_after)
    cap.release()
    A=np.asarray(transforms); UV=np.asarray(tracks); raw_stable=np.asarray(observations)
    X=np.array([r['X'] for r in results]);Q=np.array([r['Q'] for r in results])
    status=np.array([r['status'] for r in results])
    accepted=np.asarray(accepted)
    oldq=np.einsum('kij,kj->ki',A[:,:,:2],branch['Q'])+A[:,:,2]
    old_error=np.linalg.norm(oldq-raw_stable,axis=1)
    error=np.linalg.norm(UV-raw_stable,axis=1)
    spatial_error=np.linalg.norm(Q-raw_stable,axis=1)
    s=np.array([r['s'] for r in results]);speeds=np.abs(np.diff(s))*FPS
    speed3d=np.linalg.norm(np.diff(X,axis=0),axis=1)*FPS
    valid3d=np.isfinite(Q).all(axis=1)
    def summary(e,ok):
        return dict(n=int(ok.sum()),median=float(np.median(e[ok])),
                    p90=float(np.percentile(e[ok],90)),max=float(np.max(e[ok])),
                    over30_fraction=float(np.mean(e[ok]>30)))
    shared=accepted & branch['accepted']
    report=dict(
        metric_note='Image residuals are fit diagnostics against the same detector, not independent accuracy.',
        old_overlay_vs_same_detection_px=summary(old_error,shared),
        new_yellow_track_vs_same_detection_px=summary(error,shared),
        constrained_3d_vs_same_detection_px=summary(spatial_error,shared & valid3d),
        accepted_detections=int(accepted.sum()),
        geometry_conflict_frames=int(np.sum(status=='geometry_conflict')),
        branch_conflict_frames=int(np.sum(status=='branch_conflict')),
        motion_limited_frames=int(np.sum(status=='motion_limited')),
        motion_geometry_conflict_frames=int(np.sum(status=='motion_geometry_conflict')),
        missing_detection_frames=int(np.sum(status=='missing_detection')),
        background_registration_failure_frames=int(np.sum(~np.array(registration_ok))),
        background_landmark_residual_px_median=float(np.nanmedian(landmarks)),
        heldout_landmark_motion_before_px_median=float(np.nanmedian(heldout_before)),
        heldout_landmark_motion_after_px_median=float(np.nanmedian(heldout_after)),
        added_processing_ms_median=float(np.median(times)),
        added_processing_ms_p90=float(np.percentile(times,90)),
        axial_speed_max_mm_s=float(np.nanmax(speeds)),
        axial_speed_violation_count=int(np.sum(speeds>60+1e-4)),
        full_3d_speed_max_mm_s=float(np.nanmax(speed3d)),
        full_3d_speed_violation_count=int(np.sum(speed3d>60+1e-3)),
        latency_note='2D overlay causal; 3D replays V11 branch prior available 12 frames later. Video and estimates are aligned to the acquisition timestamp.',
        calibration_note='Camera, radii, V11 branch prior and cached detector are inherited setup/replay artifacts; not a newly calibrated full live fluoroscopy pipeline.')
    assert np.nanmax(speeds)<=60+1e-4 and np.nanmax(speed3d)<=60+1e-3
    data=dict(A=A,UV=UV,raw_stable=raw_stable,accepted=accepted,X=X,Q=Q,s=s,
              status=status,radial=np.array([r['radial'] for r in results]),
              radius=np.array([r['radius'] for r in results]),
              old_Q=oldq,landmark_residual=landmarks,registration_ok=registration_ok)
    np.savez(DATA/'v12_aligned.npz',**data)
    (OUT/'v12_alignment_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (OUT/'v12_aligned_tracking.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['frame','acquisition_time_s','3d_available_time_s','u_stable','v_stable',
                         'x_mm','y_mm','z_mm','s_mm','radial_required_mm','radius_proxy_mm','status'])
        for k in range(len(UV)):
            writer.writerow([k,k/FPS,min(k+12,len(UV)-1)/FPS,*UV[k],*X[k],s[k],data['radial'][k],data['radius'][k],status[k]])
    print(json.dumps(report,indent=2),flush=True)
    return g,data


def dashed_route(frame, route, color=(160,135,85)):
    pts=np.rint(route).astype(np.int32)
    for j in range(0,len(pts)-3,12):
        cv2.polylines(frame,[pts[j:j+7]],False,color,1,cv2.LINE_AA)


def text(frame,label,xy,scale=.55,color=(230,230,230)):
    cv2.putText(frame,label,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def render(g,data):
    cap=cv2.VideoCapture(str(VIDEO))
    path=OUT/'figs'/'V12_stabilized_aligned.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(1520,800))
    if not writer.isOpened(): raise RuntimeError('Cannot open video writer')
    view=Rotation.from_euler('xyz',[25,0,-55],degrees=True).as_matrix()
    c=g.C.mean(axis=0)
    base=(g.C-c)@view.T
    scale=min(475/max(np.ptp(base[:,0]),1),355/max(np.ptp(base[:,1]),1))
    offset=np.array([280,550])-scale*(base[:,:2].max(axis=0)+base[:,:2].min(axis=0))/2
    route3d=base[:,:2]*scale+offset
    trail=deque(maxlen=60)
    crops=[]; selected={100,480,540,600,897,1000}
    for k in range(len(data['UV'])):
        ok,original=cap.read()
        if not ok: raise RuntimeError(f'Cannot render frame {k}')
        stable=cv2.warpAffine(original,data['A'][k],(W,H),flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
        # Keep natural video colours and fixed display coordinates.
        display=stable.copy()
        dashed_route(display,g.P)
        uv=data['UV'][k];q=data['Q'][k]; status=data['status'][k]
        if np.isfinite(uv).all():
            trail.append(uv.copy())
            if len(trail)>1:
                cv2.polylines(display,[np.rint(trail).astype(np.int32)],False,(0,230,255),2,cv2.LINE_AA)
            point=tuple(np.rint(uv).astype(int))
            cv2.circle(display,point,10,(20,20,20),3,cv2.LINE_AA)
            cv2.circle(display,point,10,(0,230,255),1,cv2.LINE_AA)
            cv2.circle(display,point,2,(0,230,255),-1,cv2.LINE_AA)
        else:
            trail.clear()
        if status in ('geometry_conflict','motion_limited'):
            point3d=tuple(np.rint(q).astype(int))
            cv2.drawMarker(display,point3d,(30,80,255),cv2.MARKER_TILTED_CROSS,12,2)
            cv2.line(display,point3d,tuple(np.rint(uv).astype(int)),(30,80,255),1,cv2.LINE_AA)
        panel=np.full((800,560,3),(32,29,25),np.uint8)
        text(panel,'ROBOT TRACK / FIXED 3D VIEW',(20,30),.66)
        text(panel,'Yellow: observed robot position + recent path',(20,65),.5,(0,230,255))
        text(panel,'Dashed: supplied vessel centreline',(20,90),.5,(185,160,110))
        # Fixed, stable zoom at the junction makes registration inspectable.
        crop=display[395:680,70:420]
        panel[108:365,20:335]=cv2.resize(crop,(315,257),interpolation=cv2.INTER_LINEAR)
        text(panel,'JUNCTION',(350,145),.53)
        if status=='geometry_conflict':
            text(panel,'3D MODEL',(350,185),.5,(50,120,255))
            text(panel,'CONFLICT',(350,210),.5,(50,120,255))
        elif status in ('missing_detection','motion_geometry_conflict','branch_conflict'):
            text(panel,'DETECTION',(350,185),.5,(50,120,255))
            text(panel,'MISSING',(350,210),.5,(50,120,255))
        elif status=='motion_limited':
            text(panel,'MOTION',(350,185),.5,(50,120,255))
            text(panel,'LIMITED',(350,210),.5,(50,120,255))
        else:
            text(panel,'3D MODEL',(350,185),.5)
            text(panel,'COMPATIBLE',(350,210),.45)
        text(panel,'3D output delay: 0.40 s',(20,395),.5)
        cv2.polylines(panel,[np.rint(route3d).astype(np.int32)],False,(140,130,110),2,cv2.LINE_AA)
        lo=max(0,k-90)
        segment=data['X'][lo:k+1]
        good=np.isfinite(segment).all(axis=1)
        for j in range(len(segment)-1):
            if good[j] and good[j+1]:
                points=(segment[j:j+2]-c)@view.T
                cv2.polylines(panel,[np.rint(points[:,:2]*scale+offset).astype(np.int32)],False,(0,200,230),2,cv2.LINE_AA)
        if np.isfinite(data['X'][k]).all():
            point=(data['X'][k]-c)@view.T
            color=(30,80,255) if status=='geometry_conflict' else (0,230,255)
            cv2.circle(panel,tuple(np.rint(point[:2]*scale+offset).astype(int)),6,color,-1,cv2.LINE_AA)
            text(panel,f"s = {data['s'][k]:.1f} mm; radial need = {data['radial'][k]:.1f} mm",(20,750),.5)
        text(panel,'Depth is model-dependent, not 3D ground truth.',(20,778),.5)
        canvas=np.full((800,1520,3),(32,29,25),np.uint8)
        canvas[:720,:960]=display
        canvas[:,960:]=panel
        text(canvas,f'Acquired {k/FPS:05.2f} s | frame {k:04d} | background stabilised',(15,748),.6)
        text(canvas,'Synchronous replay: video and overlays share the same acquisition timestamp',(15,778),.5)
        writer.write(canvas)
        if k in selected:
            raw=data['raw_stable'][k]
            centre=raw if np.isfinite(raw).all() and data['accepted'][k] else np.array([235.,530.])
            x0,y0=np.clip(np.rint(centre-[155,105]).astype(int),[0,0],[650,510])
            before=stable[y0:y0+210,x0:x0+310].copy()
            cv2.polylines(before,[np.rint(g.P-[x0,y0]).astype(np.int32)],False,(0,230,255),1)
            old=tuple(np.rint(data['old_Q'][k]-[x0,y0]).astype(int))
            cv2.drawMarker(before,old,(0,255,0),cv2.MARKER_CROSS,14,1)
            after=display[y0:y0+210,x0:x0+310].copy()
            strip=np.full((25,620,3),30,np.uint8)
            text(strip,f'Frame {k} | old overlay                         corrected overlay',(5,18),.45)
            crops.append(np.vstack([strip,np.hstack([before,after])]))
    cap.release();writer.release()
    imwrite_u(OUT/'figs'/'v12_before_after.jpg',np.vstack(crops))
    print(f'Saved {path}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    geometry,data=analyse()
    if args.render: render(geometry,data)
