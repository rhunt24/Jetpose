import pyrealsense2 as rs
from estimater import *
from datareader import *
from FoundationPose.mask import *
import time

# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = True
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.7"

parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--mesh_file', type=str, default=f'/home/mikrolar_orin/Documents/JetPose/Jetpose/FoundationPose/obj_files/cube/cube.obj')
parser.add_argument('--est_refine_iter', type=int, default=3)
parser.add_argument('--track_refine_iter', type=int, default=1) #1
parser.add_argument('--debug', type=int, default=0) # 3 # 1 for cam view only
parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
args = parser.parse_args()

set_logging_format()
set_seed(0)

mesh = trimesh.load(args.mesh_file, force='mesh')

debug = args.debug
debug_dir = args.debug_dir
os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
logging.info("estimator initialization done")

#create mask
create_mask()
mask = cv2.imread("mask.png")

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb: 
    print("Requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

i = 0
# create_mask()
# 386.75469970703125 0.0 315.081298828125
# 0.0 386.2650451660156 247.45974731445312
# 0.0 0.0 1.0
cam_K = np.array([[386.75469970703125, 0., 315.081298828125],
                   [0., 386.2650451660156, 247.45974731445312],
                   [0., 0., 1.]])
# cam_K = np.array([[0.503, 0.805, 0.506],
#                    [0.506, -0.056, 0.068],
#                    [-0.001, 0., -0.022]])
Estimating = True
time.sleep(3)
# Streaming loop
try:
    # Added 06/11/2025:
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    # Set visual preset (after pipeline start)
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
    while Estimating:
        start_loop = time.time()
        # Get frameset of color and depth
        start_time = time.time()
        frames = pipeline.wait_for_frames()
        end_time = time.time()
        print("wait_for_frame took", end_time - start_time, "seconds.")

        start_time = time.time()
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        end_time = time.time()
        print("align_process took", end_time - start_time, "seconds.")

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        # added again
        # filtered_depth = spatial.process(aligned_depth_frame)
        # filtered_depth = temporal.process(filtered_depth)
        # filtered_depth = hole_filling.process(filtered_depth)
        # depth_image = np.asanyarray(filtered_depth.get_data())/1e3

        depth_image = np.asanyarray(aligned_depth_frame.get_data())/1e3
        color_image = np.asanyarray(color_frame.get_data())
    
        # Scale depth image to mm
        depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.float32)

        # cv2.imshow('color', color_image)
        # cv2.imshow('depth before filtering', depth_image_old)
        # cv2.imshow('depth', depth_image)
        
        if cv2.waitKey(1) == 13:
            Estimating = False
            break   
        
        logging.info(f'i:{i}')
        
        start_time = time.time()
        H, W = cv2.resize(color_image, (640,480)).shape[:2]
        color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth_image_scaled, (W,H), interpolation=cv2.INTER_NEAREST)
        end_time = time.time()
        print("resize took", end_time - start_time, "seconds.")

        depth[(depth<0.1) | (depth>=np.inf)] = 0
        
        if i==0:
            if len(mask.shape)==3:
                for c in range(3):
                    if mask[...,c].sum()>0:
                        mask = mask[...,c]
                        break
            mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
        
            pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            
            if debug>=3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, cam_K)
                valid = depth>=0.1
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
            
        else:
            start_time = time.time()
            pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)
            end_time = time.time()
        print("track_one took", end_time - start_time, "seconds.")

        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{i}.txt', pose.reshape(4,4))

        if debug>=1:
            center_pose = pose@np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[...,::-1])
            cv2.waitKey(1)


        if debug>=2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{i}.png', vis)
        
        i += 1
        end_loop = time.time()
        print("Loop ", i, " took", end_loop - start_loop, "seconds.")
            
            
        
finally:
    pipeline.stop()
