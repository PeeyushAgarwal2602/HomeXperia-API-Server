import cv2
import numpy as np
import math

def get_lighting_map(img, blur_k=51):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur_k % 2 == 0: blur_k += 1
    gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    return gray.astype(np.float32) / 255.0

def tile_texture(pattern, area_w, area_h, tile_size_w, grout_width=0, grout_color=(180, 180, 180)):
    if isinstance(grout_color, str):
        try:
            hex_c = grout_color.lstrip('#')
            if len(hex_c) == 6:
                r = int(hex_c[0:2], 16)
                g = int(hex_c[2:4], 16)
                b = int(hex_c[4:6], 16)
                grout_color = (b, g, r)
            else:
                grout_color = (180, 180, 180) 
        except ValueError:
            print(f"[WARNING] Invalid grout color code: {grout_color}. Using gray.")
            grout_color = (180, 180, 180)

    ph, pw = pattern.shape[:2]
    scale = tile_size_w / float(pw)
    tile_size_h = int(ph * scale)
    tile = cv2.resize(pattern, (int(tile_size_w), int(tile_size_h)), interpolation=cv2.INTER_LANCZOS4)

    if grout_width > 0:
        th_raw, tw_raw = tile.shape[:2]
        th_grout = th_raw + grout_width
        tw_grout = tw_raw + grout_width
        
        tile_with_grout = np.full((th_grout, tw_grout, 3), grout_color, dtype=np.uint8)
        
        tile_with_grout[0:th_raw, 0:tw_raw] = tile
        
        tile = tile_with_grout

    th, tw = tile.shape[:2]
    if th == 0 or tw == 0: return np.zeros((area_h, area_w, 3), dtype=np.uint8)

    grid_h, grid_w = area_h + th, area_w + tw
    grid_out = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for y in range(0, grid_h, th):
        for x in range(0, grid_w, tw):
            h_slice = min(th, grid_h - y)
            w_slice = min(tw, grid_w - x)
            grid_out[y:y+h_slice, x:x+w_slice] = tile[:h_slice, :w_slice]

    return grid_out[:area_h, :area_w]

def blend_hard_replace(original, texture, mask_gray, shadow_strength=0.15):
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

def get_global_corners_and_rotation(contours):
    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)
    
    rect_rot = cv2.minAreaRect(hull)
    (center, (w, h), angle) = rect_rot
    
    if w < h:
        auto_angle = angle
    else:
        auto_angle = angle - 90
        
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) == 4 and cv2.isContourConvex(approx):
        pts_dst = np.squeeze(approx).astype(np.float32)
    else:
        rect = cv2.minAreaRect(hull)
        pts_dst = cv2.boxPoints(rect)
        
    return order_points_robust(pts_dst), auto_angle

def apply_pattern(room_img, floor_tex, mask_img, repeat=3, rotation_deg=0, grout_width=0, grout_color=(180, 180, 180)):
    print(f"[INFO] Processing Floor (HD) with User Rotation {rotation_deg}, Grout {grout_width}px...")
    H, W = room_img.shape[:2]
    
    if len(mask_img.shape) == 3: mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else: mask_gray = mask_img
    
    mask_gray = cv2.resize(mask_gray, (W, H), interpolation=cv2.INTER_NEAREST)
    _, thresh = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return room_img

    try:
        pts_dst, auto_angle = get_global_corners_and_rotation(contours)
        
        # Combine Auto-Rotation with User Slider
        final_rotation = auto_angle + rotation_deg
        
        # Aspect Ratio & Dimensions
        width_top = np.linalg.norm(pts_dst[0] - pts_dst[1])
        width_bottom = np.linalg.norm(pts_dst[3] - pts_dst[2])
        dst_w = max(width_top, width_bottom)
        
        height_left = np.linalg.norm(pts_dst[0] - pts_dst[3])
        height_right = np.linalg.norm(pts_dst[1] - pts_dst[2])
        dst_h = max(height_left, height_right)

        tile_pixel_size = W / max(1, repeat)
        diagonal = int(math.ceil(math.sqrt(H**2 + W**2)))
        
        # Safety clamp for huge HD images
        super_diag = min(diagonal * 3, 30000)
        center = (super_diag // 2, super_diag // 2)

        huge_tiled = tile_texture(
            floor_tex, 
            super_diag, 
            super_diag, 
            tile_pixel_size, 
            grout_width=grout_width, 
            grout_color=grout_color
        )

        # huge_tiled = tile_texture(floor_tex, super_diag, super_diag, tile_pixel_size)

        M_rot = cv2.getRotationMatrix2D(center, final_rotation, 1.0)
        rotated_huge = cv2.warpAffine(huge_tiled, M_rot, (super_diag, super_diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        src_w = dst_w
        src_h = dst_h

        start_x = center[0] - (src_w // 2)
        start_y = center[1] - (src_h // 2)
        
        pts_src = np.array([
            [start_x, start_y],               
            [start_x + src_w, start_y],       
            [start_x + src_w, start_y + src_h], 
            [start_x, start_y + src_h]        
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped_tex = cv2.warpPerspective(rotated_huge, M, (W, H), flags=cv2.INTER_LINEAR)
        
        return blend_hard_replace(room_img, warped_tex, mask_gray, shadow_strength=0.16)

    except Exception as e:
        print(f"Error in apply_floor_pattern: {e}")
        import traceback
        traceback.print_exc()
        return room_img