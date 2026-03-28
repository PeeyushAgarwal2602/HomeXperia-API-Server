import cv2
import numpy as np
import math

def get_lighting_map(img, blur_k=51):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur_k % 2 == 0: blur_k += 1
    gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    return gray.astype(np.float32) / 255.0

def blend_hard_replace(original, texture, mask_gray, shadow_strength=0.1):
    orig_f = original.astype(np.float32) / 255.0
    tex_f = texture.astype(np.float32) / 255.0
    lighting_map = get_lighting_map(original, blur_k=51)
    lighting_3ch = cv2.merge([lighting_map, lighting_map, lighting_map])
    shaded_texture = tex_f * (lighting_3ch ** shadow_strength)
    mask_f = mask_gray.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (3, 3), 0) 
    mask_3ch = cv2.merge([mask_f, mask_f, mask_f])
    result = (orig_f * (1.0 - mask_3ch)) + (shaded_texture * mask_3ch)
    return np.clip(result * 255, 0, 255).astype(np.uint8)

def order_points_robust(pts):
    if len(pts) != 4: return pts
    sorted_y = pts[np.argsort(pts[:, 1])]
    top_pts = sorted_y[:2]
    bottom_pts = sorted_y[2:]
    top_pts = top_pts[np.argsort(top_pts[:, 0])]
    bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]
    tl, tr = top_pts[0], top_pts[1]
    bl, br = bottom_pts[0], bottom_pts[1]
    return np.array([tl, tr, br, bl], dtype="float32")

def get_global_corners(contours):
    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    if len(approx) == 4:
        pts_dst = np.squeeze(approx).astype(np.float32)
    else:
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        pts_dst = box
    return order_points_robust(pts_dst)

def apply_pattern(room_img, rug_tex, mask_img, repeat=1, rotation_deg=0):
    print("[INFO] Processing Rug (Accepting HD Input)...")
    H, W = room_img.shape[:2]
    
    if len(mask_img.shape) == 3: mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else: mask_gray = mask_img
    
    mask_gray = cv2.resize(mask_gray, (W, H), interpolation=cv2.INTER_NEAREST)
    _, thresh = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return room_img

    try:
        pts_dst = get_global_corners(contours)
        
        tex_h, tex_w = rug_tex.shape[:2]

        if rotation_deg != 0:
            angle_rad = math.radians(rotation_deg)
            cos_a = abs(math.cos(angle_rad))
            sin_a = abs(math.sin(angle_rad))
            
            # Calculate new dimensions required if the box was expanding
            new_w = tex_w * cos_a + tex_h * sin_a
            new_h = tex_w * sin_a + tex_h * cos_a
            
            # Scale factor to zoom the image so it covers the original dimensions entirely
            scale = max(new_w / tex_w, new_h / tex_h)
            
            center = (tex_w / 2, tex_h / 2)
            M_rot = cv2.getRotationMatrix2D(center, rotation_deg, scale)
            
            # Apply rotation and zoom, cropping right back to original (tex_w, tex_h) size
            rug_tex = cv2.warpAffine(rug_tex, M_rot, (tex_w, tex_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        pad_x = int(tex_w * 0.2)
        pad_y = int(tex_h * 0.2)
        padded_rug = cv2.copyMakeBorder(rug_tex, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT)
        
        pts_src = np.array([
            [pad_x, pad_y],                   
            [pad_x + tex_w - 1, pad_y],       
            [pad_x + tex_w - 1, pad_y + tex_h - 1], 
            [pad_x, pad_y + tex_h - 1]        
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped_tex = cv2.warpPerspective(padded_rug, M, (W, H), flags=cv2.INTER_LINEAR)
        
        return blend_hard_replace(room_img, warped_tex, mask_gray, shadow_strength=0.15)

    except Exception as e:
        print(f"Error in apply_rug_pattern: {e}")
        return room_img