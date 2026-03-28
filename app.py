import os
import io
import cv2
import base64
import requests
import numpy as np
import uuid
import time
import json
from flask import Flask, request, jsonify, send_from_directory, url_for, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from functools import wraps

from utils.curtain import apply_pattern as apply_curtain_pattern
from utils.rugs import apply_pattern as apply_rug_pattern
from utils.floor import apply_pattern as apply_floor_pattern
from utils.wall import apply_pattern as apply_wall_pattern
from utils.pdf_generator import generate_report_pdf

load_dotenv()
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated'
MASK_FOLDER = 'masks'
DATA_FILE = os.path.join('data', 'rooms_data.json')
IMAGE_FOLDER = os.path.join('static', 'room-images')
API_KEY = os.getenv("APP_API_KEY")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

app.config['GENERATED_FOLDER'] = GENERATED_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def require_api_key(f):
    @wraps(f)
    def protected_function(*args, **kwargs):
        if request.headers.get('x-api-key') != API_KEY:
            return jsonify({'error': 'Unauthorized access'}), 401
        return f(*args, **kwargs)
    return protected_function

def require_admin_auth(f):
    @wraps(f)
    def pass_protected_function(*args, **kwargs):
        if request.headers.get('x-api-key') != API_KEY or request.headers.get('Authorization') != AUTH_TOKEN:
            return jsonify({'error': 'Unauthorized access'}), 401
        return f(*args, **kwargs)
    return pass_protected_function

def load_room_data():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def download_image(url):
    try:
        if not url: return None
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("Could not decode image")
        return img
    except Exception as e:
        print(f"🔴 [ERROR] Failed to download image from {url}: {e}")
        return None

def find_category(category):
    if 'curtain' in category: return 'curtain'
    if 'rug' in category: return 'rug'
    if 'floor' in category: return 'floor'
    if 'wall' in category: return 'wall'
    return 'curtain'

def preprocess_image(image, room_id):
    if image is None: return None
        
    print("➡ [INFO] Applying pre-processing (Denoise + Sharpen)...")
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    gaussian_blur = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian_blur, -0.5, 0)
    cv2.imwrite(f"Debugs/debug_sharpened_{room_id}.jpg", sharpened)
    return sharpened

def upscale_image(image, room_id, target_max_dim=4000):
    if image is None: return None
    
    h, w = image.shape[:2]
    current_max = max(h, w)

    if current_max >= target_max_dim: 
        return image
    
    scale = 4500 / current_max
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    print(f"➡ [INFO] Upscaling image from {w}x{h} to {new_w}x{new_h}")
    
    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4) # INTER_LANCZOS4 is mathematically superior for upscaling clarity
    cv2.imwrite(f"Debugs/debug_upscaled_{room_id}.jpg", upscaled)
    return upscaled

def get_or_create_mask(room_id, hotspot_id, base_image, coords, mask_url=None):
    mask_filename = f"mask_{room_id}_{hotspot_id}.png"
    mask_path = os.path.join(MASK_FOLDER, mask_filename)

    if os.path.exists(mask_path):
        print(f"➡ [INFO] Loading cached mask: {mask_filename}")
        return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask_url:
        print(f"➡ [INFO] Downloading mask from URL for {hotspot_id}...")
        downloaded_mask = download_image(mask_url)
        if downloaded_mask is not None:
            if len(downloaded_mask.shape) == 3:
                downloaded_mask = cv2.cvtColor(downloaded_mask, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(mask_path, downloaded_mask)
            return downloaded_mask
        else:
            print(f"⚠ [WARN] Failed to download mask from {mask_url}. Falling back to SAM API.")

    print(f"➡ [INFO] Generating new mask for {hotspot_id}...")

    orig_h, orig_w = base_image.shape[:2]
    MAX_DIM = 2040
    scale_factor = 1.0
    
    processed_image = base_image
    processed_coords = list(coords)

    if max(orig_h, orig_w) > MAX_DIM:
        scale_factor = MAX_DIM / max(orig_h, orig_w)
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        
        processed_image = cv2.resize(base_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        processed_coords[0] = int(coords[0] * scale_factor)
        processed_coords[1] = int(coords[1] * scale_factor)
        
        print(f"➡ [INFO] Downscaling for SAM: {orig_w}x{orig_h} -> {new_w}x{new_h} (Scale: {scale_factor:.4f})")

    _, buffer = cv2.imencode('.jpg', processed_image)
    img_b64 = base64.b64encode(buffer).decode("utf-8")
    
    # print(f"\n[DEBUG] Coords sent to API: {processed_coords}\n")

    api_url = os.getenv("SAM_API_URL") 
    api_key = os.getenv("SAM_API_KEY")
    
    headers = { "x-api-key": api_key, "Content-Type": "application/json" }
    
    payload = {
        "base64": True,
        "image": img_b64,
        "overlay_mask": False,
        "refine_mask": True,
        "coordinates": str(processed_coords)
    }

    try:
        r = requests.post(api_url, headers=headers, json=payload, timeout=90)
        r.raise_for_status()
        
        response_json = r.json()
        if "image" in response_json:
            mask_b64 = response_json["image"]
        elif "masks" in response_json:
            mask_b64 = response_json["masks"][0]
        else:
            print("🔴 [ERROR] Unexpected API response keys")
            return None

        mask_bytes = base64.b64decode(mask_b64)

        received_mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        
        if scale_factor != 1.0:
            final_mask = cv2.resize(received_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            print(f"➡ [INFO] Upscaled mask back to {orig_w}x{orig_h}")
        else:
            final_mask = received_mask

        cv2.imwrite(mask_path, final_mask)
            
        return final_mask

    except Exception as e:
        print(f"🔴 [ERROR] SAM API Failed: {e}")
        return None

def process_single_layer(current_image, layer_data, room_id):
    hotspot_id = layer_data.get('hotspotId')
    product_url = layer_data.get('product', {}).get('productImageUrl')
    mask_url = layer_data.get('mask_image')
    
    coordinates = layer_data.get('coords', {})
    h, w = current_image.shape[:2]

    if isinstance(coordinates, dict):
        raw_x = float(coordinates.get('x', 0))
        raw_y = float(coordinates.get('y', 0))
    elif isinstance(coordinates, list) and len(coordinates) >= 2:
        raw_x = float(coordinates[0])
        raw_y = float(coordinates[1])

    if raw_x >= 1 or raw_y >= 1:
        abs_x = int(raw_x)
        abs_y = int(raw_y)
    else:
        abs_x = int(raw_x * w)
        abs_y = int(raw_y * h)
        
    # print(f"Absoulte X and Y coords X = {abs_x} and Y = {abs_y}")
    
    coords = [abs_x, abs_y]

    settings = layer_data.get('settings', {})

    repeat = int(settings.get('repeat', 12))
    shading = float(settings.get('shading', 0.6))
    rotation = int(settings.get('rotation', 0))
    groutWidth = int(settings.get('groutWidth', 0))
    groutColor = settings.get('groutColor', '#000000')

    mask = get_or_create_mask(room_id, hotspot_id, current_image, coords, mask_url)
    
    if mask is None:
        print(f"⚠ [WARN] Skipping layer {hotspot_id} due to missing mask.")
        return current_image

    # Get Texture
    texture = download_image(product_url)
    if texture is None:
        print(f"⚠ [WARN] Skipping layer {hotspot_id} due to missing texture.")
        return current_image

    # Apply Logic based on Category
    category = find_category(layer_data.get('category').lower())
    print(f"➡ [INFO] Applying {category} \nRepeat: {repeat}\nRot: {rotation}°\nShade: {shading}\nGroutWidth: {groutWidth}px\nGroutColor: {groutColor}\n")

    orig_h, orig_w = current_image.shape[:2]
    MAX_DIM = 4500
    scale_factor = 1.0

    if max(orig_h, orig_w) > MAX_DIM:
        scale_factor = MAX_DIM / max(orig_h, orig_w)
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        
        processed_image = cv2.resize(current_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        processed_image = preprocess_image(processed_image, room_id = room_id)
        
        print(f"➡ [INFO] Downscaling the Image for Texture Application: {orig_w}x{orig_h} -> {new_w}x{new_h} (Scale: {scale_factor:.4f})")

    current_image = processed_image if scale_factor != 1.0 else current_image

    try:
        if category == 'curtain':
            return apply_curtain_pattern(current_image, texture, mask, repeat=repeat, shading_strength=shading)
        elif category == 'rug':
            return apply_rug_pattern(current_image, texture, mask, repeat=1, rotation_deg=rotation)
        elif category == 'floor':
            return apply_floor_pattern(current_image, texture, mask, repeat=repeat, rotation_deg=rotation, grout_width=groutWidth, grout_color=groutColor)
        elif category == 'wall':
            return apply_wall_pattern(current_image, texture, mask, repeat=repeat)
        else:
            return apply_curtain_pattern(current_image, texture, mask, repeat=repeat, shading_strength=shading)
            
    except Exception as e:
        print(f"[ERROR] Failed to apply pattern for {hotspot_id}: {e}")
        return current_image

@app.route('/api/upload', methods=['POST'])
@require_api_key
def upload_image():
    data = request.json
    
    image_url = data.get('imageUrl')
    image_b64 = data.get('imageBase64')

    if not image_url and not image_b64:
        return jsonify({'success': False, 'error': 'No image provided (send imageUrl or imageBase64)'}), 400

    filename = f"upload_{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        if image_url:
            print(f"➡ [INFO] Downloading image from URL...")
            resp = requests.get(image_url, timeout=15)
            resp.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(resp.content)

        elif image_b64:
            print(f"➡ [INFO] Decoding Base64 image...")
            if ',' in image_b64:
                image_b64 = image_b64.split(',')[1]
            
            image_data = base64.b64decode(image_b64)
            with open(filepath, 'wb') as f:
                f.write(image_data)

        img_check = cv2.imread(filepath)
        if img_check is None:
            os.remove(filepath)
            return jsonify({'success': False, 'error': 'Uploaded file is not a valid image'}), 400

        local_url = f"{request.host_url}uploads/{filename}"

        return jsonify({
            'success': True,
            'localFilePath': filepath,
            'serverBaseUrl': local_url
        })

    except Exception as e:
        print(f"🔴 [ERROR] Upload failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/process-room', methods=['POST'])
@require_api_key
def process_room():
    data = request.json
    
    room_id = data.get('roomId')
    base_image_url = data.get('baseImageUrl')
    
    raw_apply_hotspot = data.get('applyHotspot')
    
    incoming_layers = []
    if isinstance(raw_apply_hotspot, list):
        incoming_layers = raw_apply_hotspot
    elif isinstance(raw_apply_hotspot, dict):
        incoming_layers = [raw_apply_hotspot]
    else:
        incoming_layers = []

    global_product = data.get('product', {})
    
    # History
    existing_hotspots = data.get('appliedHotspots', [])
    remaining_hotspots = data.get('remainingHotspots', [])

    if not room_id or not base_image_url:
        return jsonify({'success': False, 'error': 'Missing roomId or baseImageUrl'}), 400

    current_image = download_image(base_image_url)
    if current_image is None:
        return jsonify({'success': False, 'error': 'Failed to download base image'}), 400
    
    upscaled_current_image = upscale_image(current_image, room_id)
    current_image = preprocess_image(upscaled_current_image, room_id)

    full_layer_stack = list(existing_hotspots) 

    for layer_req in incoming_layers:
        new_hotspot_id = layer_req.get('hotspotId')
        if not new_hotspot_id: continue

        layer_product_data = layer_req.get('productImageUrl') or global_product
        
        active_layer_entry = {
            "category_type": layer_req.get('category_type'),
            "hotspotId": new_hotspot_id,
            "product": layer_product_data,
            "category": layer_req.get('category'),
            "coords": layer_req.get('coords'),
            "settings": layer_req.get('settings', {}),
            "mask_image": layer_req.get('mask_image')
        }

        replaced = False
        for i, existing_item in enumerate(full_layer_stack):
            if existing_item.get('hotspotId') == new_hotspot_id:
                full_layer_stack[i] = active_layer_entry
                replaced = True
                break
        
        if not replaced:
            full_layer_stack.append(active_layer_entry)

    print(f"➡ [INFO] Re-rendering stack of {len(full_layer_stack)} layers...")
    
    try:
        for layer in full_layer_stack:
            current_image = process_single_layer(current_image, layer, room_id)

        filename = f"final_{room_id}_{int(time.time())}.jpg"
        filepath = os.path.join(app.config['GENERATED_FOLDER'], filename)
        cv2.imwrite(filepath, current_image)
    
        final_image_url = f"{request.host_url}generated/{filename}"

        return jsonify({
            "success": True,
            "finalImageUrl": final_image_url,
            "appliedHotspots": full_layer_stack, 
            "remainingHotspots": remaining_hotspots
        })

    except Exception as e:
        print(f"🔴 [ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False, 
            "error": str(e),
            "finalImageUrl": base_image_url,
            "appliedHotspots": existing_hotspots,
            "remainingHotspots": remaining_hotspots
        })

@app.route('/api/reset', methods=['POST'])
@require_api_key
def reset_room():
    data = request.json
    room_id = data.get('roomId')

    if not room_id:
        return jsonify({'success': False, 'error': 'Missing roomId'}), 400

    deleted_count = 0
    
    target_folders = [app.config['GENERATED_FOLDER'], MASK_FOLDER]

    try:
        for folder in target_folders:
            if not os.path.exists(folder):
                continue
            
            for filename in os.listdir(folder):
                if f"_{room_id}_" in filename:
                    file_path = os.path.join(folder, filename)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"⚠ [WARN] Failed to delete {filename}: {e}")

        print(f"➡ [INFO] Reset complete for room {room_id}. Deleted {deleted_count} files.")
        return jsonify({'success': True})

    except Exception as e:
        print(f"🔴 [ERROR] Reset failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/mask-generation', methods=['POST'])
@require_api_key
def generate_masks_only():
    data = request.json
    room_id = data.get('roomId')
    hotspot_id = 'hotspotId'
    current_image = download_image(data.get('baseImageUrl'))
    coordinates = data.get('coords', {})

    h, w = current_image.shape[:2]

    if isinstance(coordinates, dict):
        raw_x = float(coordinates.get('x', 0))
        raw_y = float(coordinates.get('y', 0))
    elif isinstance(coordinates, list) and len(coordinates) >= 2:
        raw_x = float(coordinates[0])
        raw_y = float(coordinates[1])

    if raw_x >= 1 or raw_y >= 1:
        abs_x = int(raw_x)
        abs_y = int(raw_y)
    else:
        abs_x = int(raw_x * w)
        abs_y = int(raw_y * h)
        
    coords = [abs_x, abs_y]

    if not room_id or not coords:
        return jsonify({'success': False, 'error': 'Missing roomId or coordinates details'}), 400

    if current_image is None:
        return jsonify({'success': False, 'error': 'Could not get the base image for this URL'}), 404

    try:
        mask_filename = f"mask_{room_id}_{hotspot_id}.png"
        mask_path = os.path.join(MASK_FOLDER, mask_filename)
        
        if os.path.exists(mask_path):
            try:
                os.remove(mask_path)
                print(f"➡ [INFO] Mask Gen Call: Deleted cached mask {mask_filename} to force regeneration.")
            except Exception as del_err:
                print(f"⚠ [WARN] Mask Gen Call: Could not delete cached mask: {del_err}")

        mask_img = get_or_create_mask(room_id, hotspot_id, current_image, coords)

        if mask_img is not None:
            _, buffer = cv2.imencode('.png', mask_img)
            # mask_b64 = base64.b64encode(buffer).decode('utf-8')
            # mask_image = io.BytesIO(buffer)
            # return send_file(
            #     mask_image,
            #     mimetype='image/png',
            #     as_attachment=False, 
            #     download_name=f"mask_{room_id}.png"
            # )
            
            mask_url = f"{request.host_url}masks/mask_{room_id}_{hotspot_id}.png"
            return jsonify ({
                "success": True,
                "roomId": room_id,
                "maskImageUrl": mask_url
            })
        else:
            return jsonify ({
                "success": False,
                "roomId": room_id,
                "error": "SAM API returned no mask"
            }), 500

    except Exception as e:
        print(f"🔴 [ERROR] Mask generation failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate-pdf', methods=['POST'])
@require_api_key
def generate_pdf_report():
    try:
        data = request.json
        room_id = data.get('roomID', 'Unknown')
        
        pdf_output = generate_report_pdf(data)
        
        return send_file(
            pdf_output,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"HomeXperia_Design_{room_id}.pdf"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/uploads/<filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/generated/<filename>')
def serve_generated(filename):
    return send_from_directory(app.config['GENERATED_FOLDER'], filename)

@app.route('/masks/<filename>')
def serve_masks(filename):
    return send_from_directory(MASK_FOLDER, filename)

@app.route('/api/admin/cleanup/<target_folder>', methods=['DELETE'])
@require_admin_auth
def admin_cleanup(target_folder):
    ALLOWED_FOLDERS = {
        'uploads': app.config['UPLOAD_FOLDER'],
        'generated': app.config['GENERATED_FOLDER'],
        'masks': MASK_FOLDER
    }

    if target_folder not in ALLOWED_FOLDERS:
        return jsonify({
            'success': False, 
            'error': f'Invalid folder. Allowed: {list(ALLOWED_FOLDERS.keys())}'
        }), 400

    folder_path = ALLOWED_FOLDERS[target_folder]
    
    if not os.path.exists(folder_path):
        return jsonify({'success': False, 'error': 'Folder does not exist'}), 404

    deleted_count = 0
    errors = []

    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if os.path.isfile(file_path) or os.path.islink(file_path):
                try:
                    os.unlink(file_path)
                    deleted_count += 1
                except Exception as e:
                    errors.append(f"Failed to delete {filename}: {str(e)}")
        
        return jsonify({
            'success': True,
            'folder': target_folder,
            'deleted_files': deleted_count,
            'errors': errors if errors else None
        })

    except Exception as e:
        print(f"🔴 [CRITICAL] Cleanup failed for {target_folder}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rooms', methods=['GET'])
@require_api_key
def get_all_rooms():
    data = load_room_data()
    summary = []
    
    for room_id, details in data.items():
        image_url = url_for('static', filename=f'room-images/{details["filename"]}', _external=True)
        
        summary.append({
            "roomId": room_id,
            "imageUrl": image_url
        })
    
    return jsonify({"rooms": summary})

@app.route('/api/room/<room_id>', methods=['GET'])
@require_api_key
def get_room_details(room_id):
    data = load_room_data()
    room = data.get(room_id)
    
    if not room:
        return jsonify({"error": "Room not found"}), 404
    
    room['imageUrl'] = url_for('static', filename=f'room-images/{room["filename"]}', _external=True)
    
    return jsonify(room)

if __name__ == '__main__':
    app.run(debug=True, port=3000)