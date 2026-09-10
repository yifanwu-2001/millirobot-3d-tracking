"""Measure background motion independently of the robot and audit V11 geometry."""
import json
import cv2
import numpy as np
from config import VIDEO, DATA, OUT, imwrite_u
from v11_realtime_tube import TubeGeometry


def main():
    cv2.setNumThreads(1)
    g = TubeGeometry()
    raw = np.load(DATA / 'track2d.npz')['uv']
    old = np.load(DATA / 'v11_realtime.npz')
    cap = cv2.VideoCapture(str(VIDEO))
    ok, first = cap.read()
    if not ok:
        raise RuntimeError('Cannot read source video')
    reference = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    mask = np.full_like(reference, 255)
    mask[:35] = 0
    mask[-25:] = 0
    mask[:, :25] = 0
    mask[:, -25:] = 0
    # Exclude the entire robot route, not just the first-frame silhouette.
    for u in raw[np.isfinite(raw).all(axis=1)]:
        cv2.circle(mask, tuple(np.rint(u).astype(int)), 30, 0, -1)
    pts = cv2.goodFeaturesToTrack(reference, 500, .02, 12, mask=mask)
    transforms, errors, inliers = [], [], []
    selections = [100, 480, 540, 600, 897, 1000]
    crops = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for k in range(len(raw)):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot read frame {k}')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tracked, valid, _ = cv2.calcOpticalFlowPyrLK(reference, gray, pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 25, .01))
        src = pts[valid.ravel() == 1, 0]
        dst = tracked[valid.ravel() == 1, 0]
        affine, keep = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=1.5)
        if affine is None:
            raise RuntimeError(f'Cannot register frame {k}')
        error = np.linalg.norm(src @ affine[:, :2].T + affine[:, 2] - dst, axis=1)
        transforms.append(affine)
        errors.append(np.median(error[keep.ravel() == 1]))
        inliers.append(np.mean(keep))
        if k in selections:
            u = raw[k]
            p = g.P[old['s_index'][k]]
            q = old['Q'][k]
            centre = u if np.isfinite(u).all() else q
            x0, y0 = np.clip(np.rint(centre - [155, 110]).astype(int), [0, 0], [650, 500])
            crop = frame[y0:y0+220, x0:x0+310].copy()
            cv2.polylines(crop, [np.rint(g.P-[x0,y0]).astype(np.int32)], False, (0,255,255),1,cv2.LINE_AA)
            cv2.drawMarker(crop, tuple(np.rint(q-[x0,y0]).astype(int)), (0,255,0), cv2.MARKER_CROSS,16,1)
            if np.isfinite(u).all():
                cv2.circle(crop, tuple(np.rint(u-[x0,y0]).astype(int)), 9,(255,0,255),1)
            cv2.putText(crop, f'frame {k} | green=V11 | pink=det', (5,18), 0,.42,(0,0,0),1,cv2.LINE_AA)
            crops.append(crop)
    cap.release()
    transforms = np.asarray(transforms)
    angles = np.degrees(np.arctan2(transforms[:,1,0],transforms[:,0,0]))
    centre = np.array([480.,360.])
    motion = np.linalg.norm(transforms[:,:,:2] @ centre + transforms[:,:,2] - centre,axis=1)
    # Independent finite-difference check of the lift Jacobian.
    i=450
    X=g.C[i]
    numerical=np.stack([(g.project(X+np.eye(3)[j]*1e-4)-g.project(X))/1e-4 for j in range(3)],1)
    z=g.z[i]; x,y=g.xc[i,:2]
    B=g.f*np.array([[1/z,0,-x/z**2],[0,1/z,-y/z**2]])
    offsets=old['d_mm']*g.scale[old['s_index']]
    intended=g.P[old['s_index']]+offsets[:,None]*g.normal[old['s_index']]
    nonzero=np.abs(offsets)>1
    lift_error=np.linalg.norm(old['Q']-intended,axis=1)
    accepted=old['accepted'] & np.isfinite(raw).all(axis=1)
    raw_error=np.linalg.norm(old['Q']-raw,axis=1)
    report=dict(background_motion_px=dict(median=float(np.median(motion)),p90=float(np.percentile(motion,90)),max=float(motion.max())),
        background_rotation_deg_max=float(np.max(np.abs(angles))),
        landmark_residual_px_median=float(np.median(errors)),
        inlier_fraction_median=float(np.median(inliers)),
        jacobian_error_using_RT=float(np.max(np.abs(numerical-B@g.R.T))),
        jacobian_error_using_R=float(np.max(np.abs(numerical-B@g.R))),
        lift_error_active_px=dict(median=float(np.median(lift_error[nonzero])),p90=float(np.percentile(lift_error[nonzero],90)),max=float(lift_error.max())),
        v11_vs_accepted_raw_px=dict(median=float(np.median(raw_error[accepted])),p90=float(np.percentile(raw_error[accepted],90)),over30_fraction=float(np.mean(raw_error[accepted]>30))))
    np.savez(DATA/'v12_motion_audit.npz', transforms=transforms,motion_px=motion,landmark_error=errors,inliers=inliers)
    (OUT/'v12_motion_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    imwrite_u(OUT/'figs'/'v12_before_crops.jpg', np.vstack([np.hstack(crops[:3]),np.hstack(crops[3:])]))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
