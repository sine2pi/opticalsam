import cv2, torch, subprocess, numpy as np, json, logging
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans
import torchvision.transforms.functional as TVF
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from model_builder import build_sam3_video_predictor

try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    HAS_RAFT = True
except ImportError:
    HAS_RAFT = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float16
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def metadata(path):
    cmd_key = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=pict_type',
        '-of', 'csv=p=0', '-skip_frame', 'nokey', path
    ]
    res_key = subprocess.run(cmd_key, capture_output=True, text=True)
    lines = res_key.stdout.strip().split('\n')
    num_keyframes = len(lines)

    cmd_stream = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json', 
        '-show_streams', '-select_streams', 'v:0', path
    ]
    res_stream = subprocess.run(cmd_stream, capture_output=True, text=True)
    data = json.loads(res_stream.stdout)
    
    if not data.get('streams'):
        return None, None, None, None, None, None
        
    stream = data['streams'][0]
    width = int(stream['width'])
    height = int(stream['height'])
    duration = float(stream.get('duration', 0))
        
    fps_str = stream.get('r_frame_rate', '30/1')
    try:
        num, denom = map(int, fps_str.split('/'))
        fps = num / denom if denom != 0 else 30.0
    except:
        fps = 30.0

    f_tot = stream.get('nb_frames')
    if f_tot:
        nb_frames = int(f_tot)
    else:
        nb_frames = int(duration * fps) if duration > 0 else 0
    return nb_frames, num_keyframes, width, height, duration, fps

def keyframe_indices(video_path):
    cmd = ["ffprobe", "-loglevel", "error", "-select_streams", "v:0", "-show_entries", "packet=flags", "-of", "csv=p=0", video_path]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        return [i for i, line in enumerate(output.strip().split('\n')) if 'K' in line]
    except Exception as e:
        print(f"Warning: ffprobe failed ({e}), falling back to rigid chunking.")
        return []

def eye_frames(video_path, start_frame, num_frames):
    cap_chunk = cv2.VideoCapture(video_path)
    cap_chunk.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_l, frames_r = [], []
    for _ in range(num_frames):
        ret, frame = cap_chunk.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mid = frame.shape[1] // 2
        frames_l.append(Image.fromarray(frame[:, :mid]))
        frames_r.append(Image.fromarray(frame[:, mid:]))
    cap_chunk.release()
    return frames_l, frames_r

def ffmpeg_pipe(out_path, width, height, fps):
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-c:v', 'hevc_nvenc', '-preset', 'fast', '-cq', '20',
        '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709',
        '-color_trc', 'bt709', '-color_range', 'tv', out_path
    ]
    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

def generate_colors(n_colors=256, n_samples=5000):
    np.random.seed(42)
    rgb = np.random.rand(n_samples, 3)
    lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(lab)
    centers_lab = kmeans.cluster_centers_
    colors_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    return np.clip(colors_rgb, 0, 1)

COLORS = generate_colors(n_colors=128, n_samples=5000)

def morph3x3(mask: torch.Tensor, dilation: int) -> torch.Tensor:
    if dilation == 0: return mask
    x = mask.float().view(1, 1, *mask.shape) if mask.ndim == 2 else mask
    k_size = 2 * abs(dilation) + 1
    padding = abs(dilation)
    if dilation > 0:
        x = F.max_pool2d(x, kernel_size=k_size, stride=1, padding=padding)
    else:
        x = -F.max_pool2d(-x, kernel_size=k_size, stride=1, padding=padding)
    return (x > 0.5).to(mask.dtype).view(mask.shape)

def feather_mask_tensor(mask: torch.Tensor, blur_radius: int = 3, iterations: int = 1) -> torch.Tensor:
    if blur_radius <= 0: return mask
    orig_shape = mask.shape
    if mask.ndim == 2: x = mask[None, None, ...]
    elif mask.ndim == 3: x = mask[None, ...]
    else: x = mask
    x = x.float()
    k_size = blur_radius * 2 + 1
    for _ in range(iterations):
        x = TVF.gaussian_blur(x, kernel_size=k_size, sigma=float(blur_radius))
    return x.view(orig_shape) 

def straighten_mask_edges(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size <= 0: return mask
    orig_shape = mask.shape
    m = (mask > 0.5).byte().cpu().numpy()
    if m.ndim > 2: m = m.squeeze()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    return torch.from_numpy(m).to(mask.device, dtype=mask.dtype).view(orig_shape)

def soft_matte_tensor(outputs, frames=None, width=None, height=None, fg_color=(255, 0, 0), bg_color=(0, 0, 0), dilation=0, feather_radius=0, smooth_edges=0):
    if frames is not None:
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames).to(device)
        if frames.ndim == 3 and frames.shape[0] in [1, 3]:
            frames = frames.permute(1, 2, 0)
            if frames.min() < -0.1:
                frames = (frames * 0.5) + 0.5
        if frames.dtype in [torch.bfloat16, torch.float16, torch.float32] or frames.max() <= 1.0:
            frames = (frames * 255)
        frames = frames[..., :3].to(device, dtype=dtype)
        height, width = frames.shape[:2]

    combined_mask = torch.zeros((height, width), dtype=dtype).to(device)
    for i in range(len(outputs)):
        mask = outputs[i]
        if isinstance(mask, np.ndarray): mask = torch.from_numpy(mask).to(device)
        if dilation != 0: mask = morph3x3(mask, dilation)
        if mask.shape != (height, width):
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(height, width), mode='bicubic', align_corners=True, antialias=True).squeeze(0).squeeze(0)
        if smooth_edges > 0: mask = straighten_mask_edges(mask, kernel_size=smooth_edges)
        combined_mask = torch.maximum(combined_mask, mask)
    if feather_radius > 0:
        combined_mask = feather_mask_tensor(combined_mask, blur_radius=feather_radius)

    mask_3d = combined_mask[:, :, None]
    fg_array = torch.tensor(fg_color, dtype=dtype).to(device)
    bg_array = torch.tensor(bg_color, dtype=dtype).to(device)
    return (fg_array * mask_3d + bg_array * (1.0 - mask_3d)).to(torch.uint8)

def tensor_masked_frame(outputs, frames=None, mask_type="foreground", alpha=1.0, dilation=0, matte_color=(0, 0, 0), fg_color=None, feather_radius=0, smooth_edges=0):
    if frames is not None:
        if isinstance(frames, np.ndarray): frames = torch.from_numpy(frames).to(device)
        if frames.ndim == 3 and frames.shape[0] in [1, 3]: frames = frames.permute(1, 2, 0)
        if frames.min() < -0.1: frames = (frames * 0.5) + 0.5
    if frames.dtype in [torch.bfloat16, torch.float16, torch.float32] or frames.max() <= 1.0:
        frames = (frames * 255)
    frames = frames[..., :3].to(device, dtype=dtype)
    height, width = frames.shape[:2]

    if fg_color is not None:
        fg_tensor = torch.tensor(fg_color, dtype=dtype, device=device)
        frames = fg_tensor.view(1, 1, 3).expand((height, width, 3))

    combined_mask = torch.zeros((height, width), dtype=dtype).to(device)
    if "out_binary_masks" in outputs:
        for i in range(len(outputs["out_obj_ids"])):
            mask = outputs["out_binary_masks"][i]
            if isinstance(mask, np.ndarray): mask = torch.from_numpy(mask).to(device)
            if dilation != 0: mask = morph3x3(mask, dilation)
            if mask.shape != (height, width):
                mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(height, width), mode="bicubic", align_corners=True, antialias=True).squeeze(0).squeeze(0)
            if smooth_edges > 0: mask = straighten_mask_edges(mask, kernel_size=smooth_edges)
            combined_mask = torch.maximum(combined_mask, mask)
        if feather_radius > 0: combined_mask = feather_mask_tensor(combined_mask, blur_radius=feather_radius)

    mask_3d = combined_mask[:, :, None]
    matte_tensor = torch.tensor(matte_color, dtype=dtype, device=device)
    if mask_type == "foreground":
        result = (frames * mask_3d + matte_tensor * (1.0 - mask_3d)).to(torch.uint8)
    elif mask_type == "background":
        result = (frames * (1.0 - mask_3d) + matte_tensor * mask_3d).to(torch.uint8)
    return result

def combine_masks_sbs(out_l, out_r):
    ids_l, ids_r = out_l["out_obj_ids"], out_r["out_obj_ids"]
    all_ids = sorted(list(set(ids_l.tolist()) | set(ids_r.tolist())))
    merged = {"out_obj_ids": [], "out_probs": [], "out_boxes_xywh": [], "out_binary_masks": []}
    for obj_id in all_ids:
        idx_l = np.where(ids_l == obj_id)[0]
        if len(idx_l) > 0:
            mask_l, prob_l, box_l = out_l["out_binary_masks"][idx_l[0]], out_l["out_probs"][idx_l[0]], out_l["out_boxes_xywh"][idx_l[0]]
        else:
            ref_idx_r = np.where(ids_r == obj_id)[0][0]
            mask_l, prob_l, box_l = np.zeros_like(out_r["out_binary_masks"][ref_idx_r]), 0.0, [0, 0, 0, 0]
        idx_r = np.where(ids_r == obj_id)[0]
        if len(idx_r) > 0:
            mask_r, prob_r, box_r = out_r["out_binary_masks"][idx_r[0]], out_r["out_probs"][idx_r[0]], out_r["out_boxes_xywh"][idx_r[0]]
        else:
            mask_r, prob_r, box_r = np.zeros_like(mask_l), 0.0, [0, 0, 0, 0]
        merged["out_obj_ids"].append(obj_id)
        merged["out_probs"].append((prob_l + prob_r) / 2 if (prob_l > 0 and prob_r > 0) else max(prob_l, prob_r))
        merged["out_binary_masks"].append(np.concatenate([mask_l, mask_r], axis=1))
        merged["out_boxes_xywh"].append([box_l[0]/2, box_l[1], box_l[2]/2, box_l[3]])
    for k in merged: merged[k] = np.array(merged[k])
    return merged

class RaftMotionCompensator:
    def __init__(self, device=None, max_size=256, flow_scale=1.0, interp_mode="bicubic"):
        self.device = torch.device(device) if isinstance(device, str) else (device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.max_size = max_size
        self.flow_scale = flow_scale
        self.interp_mode = interp_mode
        self.model = None
        self.transforms = None

    def _load_model(self):
        if self.model is None:
            if not HAS_RAFT:
                raise ImportError("torchvision.models.optical_flow is required for RAFT.")
            weights = Raft_Small_Weights.DEFAULT
            self.transforms = weights.transforms()
            self.model = raft_small(weights=weights, progress=False).to(self.device).eval()

    def _compute_raft_flow(self, img1, img2):

        orig_H, orig_W = img1.shape[2], img1.shape[3]
        scale_factor = self.flow_scale
        current_H, current_W = orig_H * scale_factor, orig_W * scale_factor

        if max(current_H, current_W) > self.max_size:
            scale_factor = scale_factor * (self.max_size / float(max(current_H, current_W)))

        if scale_factor != 1.0:
            new_H, new_W = int(orig_H * scale_factor), int(orig_W * scale_factor)
            img1_s = F.interpolate(img1, size=(new_H, new_W), mode=self.interp_mode, antialias=True)
            img2_s = F.interpolate(img2, size=(new_H, new_W), mode=self.interp_mode, antialias=True)
        else:
            img1_s, img2_s = img1, img2
            
        img1_t, img2_t = self.transforms(img1_s, img2_s)
        _, _, H_s, W_s = img1_t.shape
        pad_h, pad_w = (8 - H_s % 8) % 8, (8 - W_s % 8) % 8

        if pad_h > 0 or pad_w > 0:
            img1_t = F.pad(img1_t, (0, pad_w, 0, pad_h))
            img2_t = F.pad(img2_t, (0, pad_w, 0, pad_h))
            
        with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.device.type == 'cuda' else torch.float32):
            flow = self.model(img1_t, img2_t)[-1].float()
            
        flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
        if pad_h > 0 or pad_w > 0:
            flow = flow[:, :, :H_s, :W_s]
        if scale_factor != 1.0:
            flow = F.interpolate(flow, size=(orig_H, orig_W), mode=self.interp_mode)
            flow = flow / scale_factor
        return flow

    def _warp_frame(self, pt_frame, flow, t=1.0):
        
        if pt_frame.ndim == 3:
            C, H, W = pt_frame.shape
            flow_scaled = flow * t
            y, x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
            x_norm = 2.0 * (x + flow_scaled[0]) / max(W - 1, 1) - 1.0
            y_norm = 2.0 * (y + flow_scaled[1]) / max(H - 1, 1) - 1.0
            grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)
            align_corners = True if self.interp_mode != 'nearest' else None
            
            if align_corners is None:
                return F.grid_sample(pt_frame.unsqueeze(0), grid, mode=self.interp_mode, padding_mode='border').squeeze(0)
            else:
                return F.grid_sample(pt_frame.unsqueeze(0), grid, mode=self.interp_mode, padding_mode='border', align_corners=align_corners).squeeze(0)
        
        elif pt_frame.ndim == 4:
            N, C, H, W = pt_frame.shape
            flow_scaled = flow * t
            y, x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
            x_norm = 2.0 * (x + flow_scaled[0]) / max(W - 1, 1) - 1.0
            y_norm = 2.0 * (y + flow_scaled[1]) / max(H - 1, 1) - 1.0
            grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)
            grid = grid.expand(N, -1, -1, -1)
            align_corners = True if self.interp_mode != 'nearest' else None

            if align_corners is None:
                return F.grid_sample(pt_frame, grid, mode=self.interp_mode, padding_mode='border')
            else:
                return F.grid_sample(pt_frame, grid, mode=self.interp_mode, padding_mode='border', align_corners=align_corners)
        else:
            raise ValueError(f"Unexpected pt_frame dimensions: {pt_frame.ndim}")

    def stabilize_alpha_sequence(self, rgb_frames, alpha_masks, blend_weights=(0.2, 0.6, 0.2)):

        self._load_model()
        is_numpy = isinstance(rgb_frames, np.ndarray)
        
        if is_numpy:
            t_rgb = torch.from_numpy(rgb_frames).permute(0, 3, 1, 2).float().div(255.0).to(self.device)
            if alpha_masks.ndim == 3:
                t_alpha = torch.from_numpy(alpha_masks).unsqueeze(1).float().div(255.0).to(self.device)
            else:
                t_alpha = torch.from_numpy(alpha_masks).permute(0, 3, 1, 2).float().div(255.0).to(self.device)
        else:
            t_rgb = rgb_frames.to(self.device).float()
            t_alpha = alpha_masks.to(self.device).float()

        num_frames = t_rgb.shape[0]
        if num_frames < 3: return alpha_masks

        stabilized_alphas = torch.zeros_like(t_alpha)
        stabilized_alphas[0] = t_alpha[0]
        stabilized_alphas[-1] = t_alpha[-1]
        w_prev, w_curr, w_next = blend_weights

        with torch.no_grad():
            for i in tqdm(range(1, num_frames - 1)):
                prev_rgb, curr_rgb, next_rgb = t_rgb[i-1:i+2]
                prev_alpha, curr_alpha, next_alpha = t_alpha[i-1:i+2]
                flow_forward = self._compute_raft_flow(prev_rgb.unsqueeze(0), curr_rgb.unsqueeze(0)).squeeze(0)
                flow_backward = self._compute_raft_flow(next_rgb.unsqueeze(0), curr_rgb.unsqueeze(0)).squeeze(0)
                alpha_prev_warped = self._warp_frame(prev_alpha, flow_forward, t=1.0)
                alpha_next_warped = self._warp_frame(next_alpha, flow_backward, t=1.0)
                merged_alpha = (w_prev * alpha_prev_warped) + (w_curr * curr_alpha) + (w_next * alpha_next_warped)
                stabilized_alphas[i] = torch.clamp(merged_alpha, 0.0, 1.0)

        if is_numpy:
            if alpha_masks.ndim == 3:
                return (stabilized_alphas.squeeze(1).cpu().numpy() * 255.0).astype(np.uint8)
            return (stabilized_alphas.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        return stabilized_alphas

    def interpolate_video(self, input_video, output_video, src_fps, dst_fps, width, height, encoder_opts, read_cmd=None):

        self._load_model()
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video]
        duration = float(subprocess.check_output(cmd).decode().strip())
        num_src_frames = int(duration * src_fps)
        num_dst_frames = int(duration * dst_fps)
        step = src_fps / dst_fps

        if read_cmd is None:
            read_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", input_video, "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"]
        write_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(dst_fps), "-i", "-"] + encoder_opts + [output_video]

        reader = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=width*height*3*2)
        writer = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, bufsize=width*height*3*2)
        
        def read_frame():
            raw = reader.stdout.read(width * height * 3)
            if not raw: return None
            frame = np.frombuffer(raw, dtype=np.uint8).copy().reshape((height, width, 3))
            return torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).to(self.device)
        
        frame_a = read_frame()
        if frame_a is None:
            raise RuntimeError("Failed to read any frames from FFmpeg!")
        frame_b = read_frame()
        current_idx, out_idx = 0, 0
        
        with torch.no_grad():
            with tqdm(total=num_dst_frames, unit="frame") as pbar:
                while True:
                    t_idx = out_idx * step
                    while current_idx < int(t_idx) and frame_b is not None:
                        frame_a = frame_b
                        frame_b = read_frame()
                        current_idx += 1
                    if frame_b is None:
                        if int(t_idx) > current_idx: break
                        out_frame = frame_a
                    else:
                        t = float(t_idx - current_idx)
                        if t <= 1e-6: out_frame = frame_a
                        else:
                            flow_fwd = self._compute_raft_flow(frame_a.unsqueeze(0), frame_b.unsqueeze(0)).squeeze(0)
                            flow_bwd = self._compute_raft_flow(frame_b.unsqueeze(0), frame_a.unsqueeze(0)).squeeze(0)
                            
                            warp_a = self._warp_frame(frame_a, flow_fwd, t)
                            warp_b = self._warp_frame(frame_b, flow_bwd, 1.0 - t)
                            mag_fwd = torch.norm(flow_fwd, dim=0, keepdim=True)
                            mag_bwd = torch.norm(flow_bwd, dim=0, keepdim=True)
                            weight_a = torch.exp(-mag_fwd * 0.1) * (1.0 - t)
                            weight_b = torch.exp(-mag_bwd * 0.1) * t
                            Z = weight_a + weight_b
                            mask = (Z > 1e-4).float()
                            norm_a = torch.where(mask > 0, weight_a / (Z + 1e-8), 1.0 - t)
                            norm_b = torch.where(mask > 0, weight_b / (Z + 1e-8), t)
                            out_frame = torch.clamp(warp_a * norm_a + warp_b * norm_b, 0.0, 1.0)
                            
                    writer.stdin.write((out_frame.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8).tobytes())
                    out_idx += 1
                    pbar.update(1)

        reader.stdout.close()
        writer.stdin.close()
        writer.wait()
        reader.wait()

class HybridSam3MotionLoop:
    def __init__(self, video_predictor=None, raft_compensator=None, target_res=(256, 256)):

        self.predictor = video_predictor
        self.raft = raft_compensator 
        self.device = self.raft.device
        self.target_res = target_res

    def process_batch(self, frames_pil, frames_bgr, prompt_text=None, bbox=None, prior_mask=None):
   
        self.raft._load_model()
        height, width = frames_bgr[0].shape[:2]
        chunk_length = len(frames_pil)

        res_vanilla = self.predictor.handle_request(dict(
            type="start_session",
            resource_path=frames_pil
        ))
        sid_vanilla = res_vanilla["session_id"]

        if prior_mask is not None:
            self.predictor.handle_request(dict(
                type="add_new_mask",
                session_id=sid_vanilla,
                frame_index=0,
                obj_id=0,
                mask=prior_mask
            ))
            
        prompt_req_vanilla = dict(type="add_prompt", session_id=sid_vanilla, frame_index=0, obj_id=0)
        if prompt_text is not None:
            prompt_req_vanilla["text"] = prompt_text
        if bbox is not None:
            prompt_req_vanilla["bounding_boxes"] = [bbox]
            prompt_req_vanilla["bounding_box_labels"] = [1]
            
        if prompt_text is not None or bbox is not None or prior_mask is None:
            self.predictor.handle_request(prompt_req_vanilla)

        out_buffer_vanilla = []
        for st in self.predictor.handle_stream_request(dict(
            type="propagate_in_video",
            session_id=sid_vanilla,
            propagation_direction="forward",
            start_frame_index=0,
            max_frame_num_to_track=chunk_length,
        )):
            out_buffer_vanilla.append(st["outputs"])

        self.predictor.handle_request(dict(type="close_session", session_id=sid_vanilla))

        vanilla_masks = []
        sam_soft = []
        for i in range(chunk_length):
            if i < len(out_buffer_vanilla):
                outputs = out_buffer_vanilla[i]
                mask_bin = np.zeros((height, width), dtype=np.uint8)
                mask_prob = np.zeros((height, width), dtype=np.float32)

                if "out_mask_logits" in outputs:
                    for logit_tensor in outputs["out_mask_logits"]:
                        if isinstance(logit_tensor, torch.Tensor):
                            logit_np = logit_tensor.cpu().numpy()
                        else:
                            logit_np = np.array(logit_tensor)
                        if logit_np.shape != (height, width):
                            logit_np = cv2.resize(logit_np, (width, height), interpolation=cv2.INTER_LINEAR)
                        prob = 1.0 / (1.0 + np.exp(-logit_np))
                        mask_prob = np.maximum(mask_prob, prob)
                        mask_bin = np.maximum(mask_bin, (prob > 0.5).astype(np.uint8) * 255)

                elif "out_binary_masks" in outputs:
                    for m in outputs["out_binary_masks"]:
                        if isinstance(m, torch.Tensor):
                            m = m.cpu().numpy()
                        if m.shape != (height, width):
                            m = cv2.resize(m.astype(np.float32), (width, height),
                                           interpolation=cv2.INTER_NEAREST)
                        mask_prob = np.maximum(mask_prob, m.astype(np.float32))
                        mask_bin = np.maximum(mask_bin, (m > 0.5).astype(np.uint8) * 255)
                
                sam_soft.append(mask_prob)
                vanilla_masks.append(mask_bin)
            else:
                sam_soft.append(np.zeros((height, width), dtype=np.float32))
                vanilla_masks.append(np.zeros((height, width), dtype=np.uint8))

        res_inline = self.predictor.handle_request(dict(
            type="start_session",
            resource_path=frames_pil
        ))
        sid_inline = res_inline["session_id"]

        if prior_mask is not None:
            self.predictor.handle_request(dict(
                type="add_new_mask",
                session_id=sid_inline,
                frame_index=0,
                obj_id=0,
                mask=prior_mask
            ))
            
        prompt_req_inline = dict(type="add_prompt", session_id=sid_inline, frame_index=0, obj_id=0)
        if prompt_text is not None:
            prompt_req_inline["text"] = prompt_text
        if bbox is not None:
            prompt_req_inline["bounding_boxes"] = [bbox]
            prompt_req_inline["bounding_box_labels"] = [1]
            
        if prompt_text is not None or bbox is not None or prior_mask is None:
            self.predictor.handle_request(prompt_req_inline)

        session_inline = self.predictor._get_session(sid_inline)
        inference_state = session_inline["state"]
        tracker_states = inference_state["tracker_inference_states"]

        if len(tracker_states) == 0:
            print(f"[WARNING] Prompt '{prompt_text}' found no objects in this chunk. Generating empty masks.")
            self.predictor.handle_request(dict(type="close_session", session_id=sid_inline))
            final_masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(chunk_length)]
            stabilized_soft = [np.zeros((height, width), dtype=np.float32) for _ in range(chunk_length)]
            return vanilla_masks, final_masks, sam_soft, stabilized_soft

        tracker_state = tracker_states[0]
        tensors_rgb = []
        for f_bgr in frames_bgr:
            f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
            t_rgb = torch.from_numpy(f_rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
            tensors_rgb.append(t_rgb)

        prev_logits = tracker_state["output_dict"]["cond_frame_outputs"][0]["pred_masks"].to(self.device).float()
        batch_size = len(tracker_state["obj_ids"])

        self.predictor.model.tracker.propagate_in_video_preflight(
            tracker_state, run_mem_encoder=True
        )

        with torch.inference_mode():
            for frame_idx in range(1, chunk_length):
                prev_tensor = tensors_rgb[frame_idx - 1]
                curr_tensor = tensors_rgb[frame_idx]

                self.predictor.model._prepare_backbone_feats(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    reverse=False
                )

                _, _, h_mask, w_mask = prev_logits.shape
                flow = self.raft._compute_raft_flow(prev_tensor.unsqueeze(0), curr_tensor.unsqueeze(0)).squeeze(0)
                flow_downscaled = torch.nn.functional.interpolate(
                    flow.unsqueeze(0), size=(h_mask, w_mask), mode="bicubic", align_corners=False
                ).squeeze(0)
                flow_downscaled[0] *= (w_mask / width)
                flow_downscaled[1] *= (h_mask / height)
                
                warped_logits = self.raft._warp_frame(prev_logits, flow_downscaled)

                dummy_point_inputs = {
                    "point_coords": torch.zeros(batch_size, 1, 2, device=self.device),
                    "point_labels": -torch.ones(batch_size, 1, dtype=torch.int32, device=self.device)
                }

                current_out, _ = self.predictor.model.tracker._run_single_frame_inference(
                    inference_state=tracker_state,
                    output_dict=tracker_state["output_dict"],
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=dummy_point_inputs,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                    prev_sam_mask_logits=warped_logits,
                )

                tracker_state["output_dict"]["non_cond_frame_outputs"][frame_idx] = current_out
                self.predictor.model.tracker._add_output_per_object(
                    tracker_state, frame_idx, current_out, "non_cond_frame_outputs"
                )
                tracker_state["frames_already_tracked"][frame_idx] = {"reverse": False}
                prev_logits = current_out["pred_masks"].to(self.device).float()

        final_masks = []
        stabilized_soft = []
        
        for i in range(chunk_length):
            storage_key = "cond_frame_outputs" if i == 0 else "non_cond_frame_outputs"
            out = tracker_state["output_dict"][storage_key][i]
            
            logits_gpu = out["pred_masks_high_res"].to(self.device) if "pred_masks_high_res" in out else out["pred_masks"].to(self.device)
            if logits_gpu.shape[0] > 0:
                logits_gpu = torch.max(logits_gpu, dim=0, keepdim=True).values
                
            logits_resized = torch.nn.functional.interpolate(
                logits_gpu,
                size=(height, width),
                mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
            
            prob = torch.sigmoid(logits_resized).cpu().numpy()
            stabilized_soft.append(prob)
            final_masks.append((prob > 0.5).astype(np.uint8) * 255)

        self.predictor.handle_request(dict(type="close_session", session_id=sid_inline))

        return vanilla_masks, final_masks, sam_soft, stabilized_soft

def test_simple_video_batch(video_path, out_path, prompt_text="One girl", batch_size=100):

    predictor = build_sam3_video_predictor(
        has_presence_token=False,
        geo_encoder_use_img_cross_attn=True,
        strict_state_dict_loading=False,
        async_loading_frames=True,
        video_loader_type="ffmpeg",
        offload_video_to_cpu = True,
        apply_temporal_disambiguation = True,
        compile = False,
    )
    hybrid_loop = HybridSam3MotionLoop(
        video_predictor=predictor,
        raft_compensator=RaftMotionCompensator(device="cuda"),
    )

    total_frames, num_keyframes, width, height, duration, fps = metadata(video_path)
    cap = cv2.VideoCapture(video_path)
    writer = ffmpeg_pipe(out_path, width * 2, height, fps)

    frame_count = 0
    pbar = tqdm(total=total_frames, desc="Processing Batches")

    all_frames = []
    all_vanilla = []
    all_hybrid = []
    all_vanilla_soft = []
    all_hybrid_soft = []

    while frame_count < total_frames:
        frames_bgr = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret: break
            frames_bgr.append(frame)

        if not frames_bgr:
            break

        chunk_length = len(frames_bgr)
        frames_pil = []
        for f in frames_bgr:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames_pil.append(Image.fromarray(rgb))

        vanilla_m, final_masks, sam_soft, stabilized_soft = hybrid_loop.process_batch(
            frames_pil=frames_pil,
            frames_bgr=frames_bgr,
            prompt_text=prompt_text,
        )
        
        torch.cuda.empty_cache()

        all_frames.extend(frames_bgr)
        all_vanilla.extend(vanilla_m)
        all_hybrid.extend(final_masks)
        all_vanilla_soft.extend(sam_soft)
        all_hybrid_soft.extend(stabilized_soft)

        for i in range(chunk_length):
            mask_vanilla = np.zeros_like(frames_bgr[i])
            mask_vanilla[:, :, 1] = vanilla_m[i]
            overlay_vanilla = cv2.addWeighted(frames_bgr[i], 0.9, mask_vanilla, 0.3, 0)
            cv2.putText(overlay_vanilla, "Vanilla SAM3", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            
            mask_hybrid = np.zeros_like(frames_bgr[i])
            mask_hybrid[:, :, 2] = final_masks[i]
            overlay_hybrid = cv2.addWeighted(frames_bgr[i], 0.9, mask_hybrid, 0.3, 0)
            cv2.putText(overlay_hybrid, "Hybrid (SAM3 + RAFT)", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            
            sbs_frame = np.hstack((overlay_vanilla, overlay_hybrid))
            writer.stdin.write(sbs_frame.tobytes())

        frame_count += chunk_length
        pbar.update(chunk_length)

    cap.release()
    writer.stdin.close()
    writer.wait()
    vanilla_results = evaluate_mask_sequence(all_frames, all_vanilla, all_vanilla_soft, hybrid_loop.raft)
    hybrid_results = evaluate_mask_sequence(all_frames, all_hybrid, all_hybrid_soft, hybrid_loop.raft)

    diff_pixel_count = 0
    total_pixels = len(all_frames) * height * width
    for m_v, m_h in zip(all_vanilla, all_hybrid):
        diff_pixel_count += np.sum(m_v != m_h)
    
    print("\n" + "="*75)
    print("           QUANTITATIVE TRACKING STABILITY REPORT")
    print("="*75)
    print(f"{'Metric':<35} | {'Vanilla SAM3':<15} | {'Hybrid SAM3+RAFT':<18}")
    print("-"*75)
    print(f"{'Mean Warping IoU (higher=best)':<35} | {vanilla_results['mean_warping_iou']:.6f}          | {hybrid_results['mean_warping_iou']:.6f}")
    print(f"{'Std Warping IoU (lower=best)':<35} | {vanilla_results['std_warping_iou']:.6f}          | {hybrid_results['std_warping_iou']:.6f}")
    print(f"{'Mean Consecutive IoU (higher=best)':<35} | {vanilla_results['mean_consec_iou']:.6f}          | {hybrid_results['mean_consec_iou']:.6f}")
    print(f"{'Std Consecutive IoU (lower=best)':<35} | {vanilla_results['std_consec_iou']:.6f}          | {hybrid_results['std_consec_iou']:.6f}")
    print(f"{'Mean Perimeter Jitter (lower=best)':<35} | {vanilla_results['mean_perimeter_diff']:.2f}           | {hybrid_results['mean_perimeter_diff']:.2f}")
    print(f"{'Std Perimeter Jitter (lower=best)':<35} | {vanilla_results['std_perimeter_diff']:.2f}           | {hybrid_results['std_perimeter_diff']:.2f}")
    print("-"*75)
    print(f"{'Mean Soft Warping IoU (higher=best)':<35} | {vanilla_results['mean_soft_warping_iou']:.6f}          | {hybrid_results['mean_soft_warping_iou']:.6f}")
    print(f"{'Std Soft Warping IoU (lower=best)':<35} | {vanilla_results['std_soft_warping_iou']:.6f}          | {hybrid_results['std_soft_warping_iou']:.6f}")
    print(f"{'Mean Soft Consecutive IoU (hi=best)':<35} | {vanilla_results['mean_soft_consec_iou']:.6f}          | {hybrid_results['mean_soft_consec_iou']:.6f}")
    print(f"{'Std Soft Consecutive IoU (lo=best)':<35} | {vanilla_results['std_soft_consec_iou']:.6f}          | {hybrid_results['std_soft_consec_iou']:.6f}")
    print("="*75)
    print(f"[DEBUG] Total binarized mask pixels: {total_pixels}")
    print(f"[DEBUG] Differing binarized mask pixels: {diff_pixel_count} ({diff_pixel_count / total_pixels * 100:.6f}%)")
    print("="*75)

def evaluate_mask_sequence(frames_bgr, hard_masks, soft_masks, raft):

    num_frames = len(frames_bgr)
    if num_frames < 2:
        return {}
    
    warping_ious = []
    soft_warping_ious = []
    consecutive_ious = []
    soft_consecutive_ious = []
    perimeters = []
    
    for m in hard_masks:
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
        perimeters.append(perimeter)
        
    perimeter_diffs = []
    for i in range(num_frames - 1):
        perimeter_diffs.append(abs(perimeters[i+1] - perimeters[i]))
        
    for i in tqdm(range(num_frames - 1), desc="Calculating Flow Alignment"):
        m1_h = hard_masks[i]
        m2_h = hard_masks[i+1]
        m1_s = soft_masks[i]
        m2_s = soft_masks[i+1]
        
        intersection_h = np.logical_and(m1_h > 127, m2_h > 127).sum()
        union_h = np.logical_or(m1_h > 127, m2_h > 127).sum()
        consec_iou_h = intersection_h / max(union_h, 1e-8)
        consecutive_ious.append(consec_iou_h)
        
        intersection_s = np.minimum(m1_s, m2_s).sum()
        union_s = np.maximum(m1_s, m2_s).sum()
        consec_iou_s = intersection_s / max(union_s, 1e-8)
        soft_consecutive_ious.append(consec_iou_s)
        
        f1_rgb = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB)
        f2_rgb = cv2.cvtColor(frames_bgr[i+1], cv2.COLOR_BGR2RGB)
        
        t_f1 = torch.from_numpy(f1_rgb).permute(2, 0, 1).float().div(255.0).to(raft.device)
        t_f2 = torch.from_numpy(f2_rgb).permute(2, 0, 1).float().div(255.0).to(raft.device)
        
        with torch.no_grad():
            flow = raft._compute_raft_flow(t_f1.unsqueeze(0), t_f2.unsqueeze(0)).squeeze(0)
            
            t_m1_h = torch.from_numpy(m1_h).unsqueeze(0).float().div(255.0).to(raft.device)
            m1_h_warped = raft._warp_frame(t_m1_h, flow, t=1.0)
            m1_h_warped_np = (m1_h_warped.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8) * 255
            
            t_m1_s = torch.from_numpy(m1_s).unsqueeze(0).to(raft.device)
            m1_s_warped = raft._warp_frame(t_m1_s, flow, t=1.0)
            m1_s_warped_np = m1_s_warped.squeeze(0).cpu().numpy()
            
        warp_inter_h = np.logical_and(m1_h_warped_np > 127, m2_h > 127).sum()
        warp_union_h = np.logical_or(m1_h_warped_np > 127, m2_h > 127).sum()
        warp_iou_h = warp_inter_h / max(warp_union_h, 1e-8)
        warping_ious.append(warp_iou_h)
        
        warp_inter_s = np.minimum(m1_s_warped_np, m2_s).sum()
        warp_union_s = np.maximum(m1_s_warped_np, m2_s).sum()
        warp_iou_s = warp_inter_s / max(warp_union_s, 1e-8)
        soft_warping_ious.append(warp_iou_s)
        
    return {
        "mean_warping_iou": float(np.mean(warping_ious)),
        "std_warping_iou": float(np.std(warping_ious)),
        "mean_consec_iou": float(np.mean(consecutive_ious)),
        "std_consec_iou": float(np.std(consecutive_ious)),
        "mean_soft_consec_iou": float(np.mean(soft_consecutive_ious)),
        "std_soft_consec_iou": float(np.std(soft_consecutive_ious)),
        "mean_soft_warping_iou": float(np.mean(soft_warping_ious)),
        "std_soft_warping_iou": float(np.std(soft_warping_ious)),
        "mean_perimeter_diff": float(np.mean(perimeter_diffs)),
        "std_perimeter_diff": float(np.std(perimeter_diffs)),
    }

test_simple_video_batch(
    video_path="video.mp4",
    out_path="out.mp4",
    prompt_text="prompt",
    batch_size=50,
)

