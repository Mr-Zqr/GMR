import pinocchio as pio
import numpy as np
import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import threading

lab_body_name_list = [
  "pelvis",
  "left_hip_pitch_link",
  "right_hip_pitch_link",
  "waist_yaw_link",
  "left_hip_roll_link",
  "right_hip_roll_link",
  "waist_roll_link",
  "left_hip_yaw_link",
  "right_hip_yaw_link",
  "torso_link",
  "left_knee_link",
  "right_knee_link",
  "left_shoulder_pitch_link",
  "right_shoulder_pitch_link",
  "left_ankle_pitch_link",
  "right_ankle_pitch_link",
  "left_shoulder_roll_link",
  "right_shoulder_roll_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_shoulder_yaw_link",
  "right_shoulder_yaw_link",
  "left_elbow_link",
  "right_elbow_link",
  "left_wrist_roll_link",
  "right_wrist_roll_link",
  "left_wrist_pitch_link",
  "right_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
]

lab_joint_name_list = [
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "waist_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "waist_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "waist_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
]

pino2lab_body_mapping = [1, 5, 21, 35, 7, 23, 37, 9, 25, 39, 11, 27, 45, 65, 13, 29, 47, 67, 15, 31, 49, 69, 51, 71, 53, 73, 55, 75, 57, 77]
pino2lab_joint_mapping = [1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 16, 23, 5, 11, 17, 24, 6, 12, 18, 25, 19, 26, 20, 27, 21, 28, 22, 29]

# Thread-safe print lock
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)

def load_robot_model(urdf_path):
    """
    Load a robot model from a URDF file.

    Parameters:
    urdf_path (str): The file path to the URDF file.

    Returns:
    pinocchio.Model: The loaded robot model.
    """
    model = pio.buildModelFromUrdf(urdf_path)
    return model

def quat_to_rotation_matrix(quat):
    """
    Convert quaternion (wxyz format) to rotation matrix.
    """
    w, x, y, z = quat
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

def process_motion_file(input_file, output_file, urdf_path, contact_threshold=0.010, verbose=True):
    """
    Process a single motion file and add contact mask.
    
    Parameters:
    input_file (str): Input npz file path
    output_file (str): Output npz file path
    urdf_path (str): Robot URDF file path
    contact_threshold (float): Contact detection threshold in meters
    verbose (bool): Whether to print detailed info
    
    Returns:
    bool: Success status
    """
    try:
        # Load robot model
        robot_model = load_robot_model(urdf_path)
        data = pio.Data(robot_model)
        
        # Load motion data
        motion_data = np.load(input_file, allow_pickle=True)
        joint_pos = motion_data['joint_pos']  # shape: (N, 29)
        root_pos = motion_data['body_pos_w'][:, 0, :]  # shape: (N, 3)
        root_quat = motion_data['body_quat_w'][:, 0, :]  # shape: (N, 4), wxyz format
        
        if verbose:
            safe_print(f"\n[{Path(input_file).name}] Motion data shape: {joint_pos.shape}")
            safe_print(f"[{Path(input_file).name}] Number of frames: {joint_pos.shape[0]}")
        
        # Build mappings
        pino_joint_name_list = [link for link in robot_model.names]
        pino_body_name_list = [frame.name for frame in robot_model.frames]
        
        # Calculate joint_mapping from lab to pinocchio
        joint_mapping = []
        for lab_joint_name in lab_joint_name_list:
            if lab_joint_name in pino_joint_name_list:
                joint_mapping.append(pino_joint_name_list.index(lab_joint_name))
        
        body_mapping = []
        for lab_body_name in lab_body_name_list:
            if lab_body_name in pino_body_name_list:
                body_mapping.append(pino_body_name_list.index(lab_body_name))
        
        # Process each frame and find contact body
        contact_bodies_per_frame = []
        contact_mask = np.zeros((joint_pos.shape[0], 30), dtype=np.int32)
        
        for frame_idx in range(joint_pos.shape[0]):
            # Get joint positions for this frame
            q = np.zeros(robot_model.nq)
            
            # Map joint positions from lab format to pinocchio format
            for lab_idx, pino_idx in enumerate(joint_mapping):
                q[pino_idx-1] = joint_pos[frame_idx, lab_idx]
            
            # Compute forward kinematics (relative to root)
            pio.forwardKinematics(robot_model, data, q)
            pio.updateFramePlacements(robot_model, data)
            
            # Get root transformation for this frame
            root_rotation = quat_to_rotation_matrix(root_quat[frame_idx])
            root_translation = root_pos[frame_idx]
            
            # First pass: find the minimum z coordinate in world frame
            min_z = float('inf')
            body_z_coords = []
            
            for lab_idx, pino_idx in enumerate(body_mapping):
                frame = robot_model.frames[pino_idx]
                frame_placement = data.oMf[pino_idx]
                
                # Transform from root frame to world frame
                local_pos = frame_placement.translation
                world_pos = root_rotation @ local_pos + root_translation
                z_coord = world_pos[2]
                
                body_z_coords.append((lab_idx, z_coord))
                
                if z_coord < min_z:
                    min_z = z_coord
            
            # Second pass: mark all bodies within threshold as contact
            contact_body_names = []
            for lab_idx, z_coord in body_z_coords:
                if z_coord - min_z <= contact_threshold:
                    contact_mask[frame_idx, lab_idx] = 1
                    contact_body_names.append(lab_body_name_list[lab_idx])
            
            contact_bodies_per_frame.append((contact_body_names, min_z))
        
        # Statistics
        if verbose:
            all_contact_bodies = []
            for body_names, _ in contact_bodies_per_frame:
                all_contact_bodies.extend(body_names)
            body_counts = Counter(all_contact_bodies)
            safe_print(f"[{Path(input_file).name}] Processed {len(contact_bodies_per_frame)} frames")
            safe_print(f"[{Path(input_file).name}] Total contacts: {np.sum(contact_mask)}")
            safe_print(f"[{Path(input_file).name}] Most frequent contact: {body_counts.most_common(3)}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save contact mask to npz file
        save_dict = {
            'fps': motion_data['fps'],
            'joint_pos': motion_data['joint_pos'],
            'joint_vel': motion_data['joint_vel'],
            'body_pos_w': motion_data['body_pos_w'],
            'body_quat_w': motion_data['body_quat_w'],
            'body_lin_vel_w': motion_data['body_lin_vel_w'],
            'body_ang_vel_w': motion_data['body_ang_vel_w'],
            'contact_mask': contact_mask
        }
        
        # Add pressure_offset if it exists
        if 'pressure_offset' in motion_data:
            save_dict['pressure_offset'] = motion_data['pressure_offset']
        
        np.savez(output_file, **save_dict)
        
        safe_print(f"[{Path(input_file).name}] ✓ Saved to: {output_file}")
        return True
        
    except Exception as e:
        safe_print(f"[{Path(input_file).name}] ✗ Error: {str(e)}")
        return False

def find_npz_files(input_dir):
    """
    Recursively find all .npz files in a directory.
    
    Parameters:
    input_dir (str): Input directory path
    
    Returns:
    list: List of npz file paths
    """
    input_path = Path(input_dir)
    npz_files = list(input_path.rglob("*.npz"))
    return npz_files

def process_directory(input_dir, output_dir, urdf_path, contact_threshold=0.010, num_threads=4, verbose=True):
    """
    Process all npz files in a directory with multi-threading.
    
    Parameters:
    input_dir (str): Input directory path
    output_dir (str): Output directory path
    urdf_path (str): Robot URDF file path
    contact_threshold (float): Contact detection threshold in meters
    num_threads (int): Number of worker threads
    verbose (bool): Whether to print detailed info
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all npz files
    npz_files = find_npz_files(input_dir)
    
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return
    
    print(f"Found {len(npz_files)} npz files")
    print(f"Using {num_threads} threads")
    print(f"Contact threshold: {contact_threshold*1000:.1f}mm")
    print(f"Output directory: {output_dir}")
    print("-" * 80)
    
    # Process files with thread pool
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        future_to_file = {}
        for npz_file in npz_files:
            # Calculate relative path and output path
            rel_path = npz_file.relative_to(input_path)
            output_file = output_path / rel_path
            
            future = executor.submit(
                process_motion_file,
                str(npz_file),
                str(output_file),
                urdf_path,
                contact_threshold,
                verbose
            )
            future_to_file[future] = npz_file
        
        # Wait for completion
        for future in as_completed(future_to_file):
            npz_file = future_to_file[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                safe_print(f"[{npz_file.name}] ✗ Exception: {str(e)}")
                failed_count += 1
    
    print("-" * 80)
    print(f"Processing complete!")
    print(f"Success: {success_count}/{len(npz_files)}")
    print(f"Failed: {failed_count}/{len(npz_files)}")

def main():
    parser = argparse.ArgumentParser(description='Add contact mask to motion files with multi-threading support')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing npz files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed files')
    parser.add_argument('--urdf_path', type=str, 
                        default='/home/amax/devel/GMR/assets/unitree_g1/main.urdf',
                        help='Path to robot URDF file')
    parser.add_argument('--contact_threshold', type=float, default=0.010,
                        help='Contact detection threshold in meters (default: 0.010)')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of worker threads (default: 4)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information for each file')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Validate URDF file
    if not os.path.exists(args.urdf_path):
        print(f"Error: URDF file does not exist: {args.urdf_path}")
        return
    
    # Process directory
    process_directory(
        args.input_dir,
        args.output_dir,
        args.urdf_path,
        args.contact_threshold,
        args.num_threads,
        args.verbose
    )

if __name__ == "__main__":
    main()

