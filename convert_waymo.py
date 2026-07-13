import os

# Suppress TF/TRT noise and disable GPU (CPU-only task)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import gc
import numpy as np
import dask
import dask.dataframe as dd
from tqdm import tqdm
import tensorflow as tf

from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.utils import lidar_utils

DASK_TMP_DIR = "/mnt/bravo/jmfleming/dask_scratchpad"
INPUT_PARQUET_DIR = "/mnt/hdd/jmfleming/waymo_raw/training"
OUTPUT_KITTI_DIR = "/mnt/bravo/jmfleming/waymo_skitti"

os.makedirs(DASK_TMP_DIR, exist_ok=True)
dask.config.set({"temporary-directory": DASK_TMP_DIR})

WAYMO_TO_SKITTI_LABEL = {
    0:  0,   1:  9,   2:  10,  3:  18,  4:  18,  5:  16,
    6:  15,  7:  13,  8:  0,   9:  17,  10: 0,   11: 0,
    12: 15,  13: 16,  14: 0,   15: 72,  16: 70,  17: 40,
    18: 40,  19: 44,  20: 48,  21: 51,  22: 52,
}

def _weather_tag(weather_str: str, tod_str: str) -> str:
    weather = weather_str.lower() if weather_str else "sunny"
    tod = tod_str.lower() if tod_str else "day"
    if "rain" in weather: return "rain"
    if "fog" in weather: return "fog"
    if "night" in tod or "dawn" in tod or "dusk" in tod: return "night"
    return "sunny"

def _read_tag(tag: str, directory: str) -> dd.DataFrame:
    return dd.read_parquet(f"{directory}/{tag}/*.parquet")

def _load_stats(parquet_dir: str) -> dict:
    stats_dict = {}
    try:
        stats_df = _read_tag("stats", parquet_dir).compute()
        for _, row in stats_df.iterrows():
            seg_name = row.get("key.segment_context_name")
            if not seg_name:
                continue
            w, t = row.get("weather", ""), row.get("time_of_day", "")
            try:
                c = v2.StatsComponent.from_dict(row)
                w = getattr(c, "weather", w)
                t = getattr(c, "time_of_day", t)
            except Exception:
                pass
            stats_dict[seg_name] = _weather_tag(str(w), str(t))
        print(f"Loaded weather stats for {len(stats_dict)} segments.")
    except Exception as e:
        print(f"Warning: could not load 'stats', defaulting to sunny. ({e})")
    return stats_dict

def _get_segment_names(parquet_dir: str) -> list:
    return list(
        _read_tag("lidar", parquet_dir)[["key.segment_context_name", "key.laser_name"]]
        .query("`key.laser_name` == 1")["key.segment_context_name"]
        .unique()
        .compute()
    )

def _load_segment_df(parquet_dir: str, segment_name: str):
    """Filter in Dask before compute() — only this segment's rows hit RAM."""
    def filtered(tag, laser_filter=True):
        df = _read_tag(tag, parquet_dir)
        df = df[df["key.segment_context_name"] == segment_name]
        if laser_filter and "key.laser_name" in df.columns:
            df = df[df["key.laser_name"] == 1]
        return df

    df = v2.merge(filtered("lidar"), filtered("lidar_pose"))
    df = v2.merge(df, filtered("vehicle_pose", laser_filter=False))
    df = v2.merge(df, filtered("lidar_calibration"))
    df = v2.merge(df, filtered("lidar_segmentation"))
    return df.compute().sort_values("key.frame_timestamp_micros")

def _convert_segment(seg_df, scene_id: int, weather_tag: str, output_dir: str):
    seq_dir = os.path.join(output_dir, "sequences", f"{scene_id:04d}")
    velo_dir = os.path.join(seq_dir, "velodyne")
    label_dir = os.path.join(seq_dir, "labels")
    os.makedirs(velo_dir,  exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    first_pose_inv = None
    time_start = None
    frame_idx = 0

    with (
        open(os.path.join(seq_dir, "poses.txt"), "w") as pose_f,
        open(os.path.join(seq_dir, "times.txt"), "w") as times_f,
        open(os.path.join(seq_dir, "weather.txt"), "w") as weather_f,
    ):
        for _, row in tqdm(seg_df.iterrows(), total=len(seg_df), leave=False):
            lidar = v2.LiDARComponent.from_dict(row)
            lidar_pose = v2.LiDARPoseComponent.from_dict(row)
            lidar_calib = v2.LiDARCalibrationComponent.from_dict(row)
            vehicle_pose = v2.VehiclePoseComponent.from_dict(row)
            lidar_seg = v2.LiDARSegmentationLabelComponent.from_dict(row)

            pose_mat = np.array(vehicle_pose.world_from_vehicle.transform).reshape(4, 4)
            if first_pose_inv is None:
                first_pose_inv = np.linalg.inv(pose_mat)
            rel_pose = first_pose_inv @ pose_mat
            pose_f.write(" ".join(f"{v:.8f}" for v in rel_pose[:3].reshape(-1)) + "\n")

            ts = lidar.key.frame_timestamp_micros / 1e6
            if time_start is None:
                time_start = ts
            times_f.write(f"{ts - time_start:.6e}\n")

            points_tensor = lidar_utils.convert_range_image_to_point_cloud(
                lidar.range_image_return1,
                lidar_calib,
                lidar_pose.range_image_return1,
                frame_pose=vehicle_pose,
                keep_polar_features=True,
            )
            pts = points_tensor.numpy()
            kitti_points = np.zeros((pts.shape[0], 4), dtype=np.float32)
            kitti_points[:, :3] = pts[:, :3]
            kitti_points[:, 3] = pts[:, 4]
            kitti_points.tofile(os.path.join(velo_dir, f"{frame_idx:06d}.bin"))

            def _to_tf_tensor(ri):
                if isinstance(ri, np.ndarray):
                    return tf.constant(ri)
                shape = ri.shape
                dims = shape.dims if hasattr(shape, 'dims') else list(shape)
                return tf.reshape(tf.constant(ri.values), dims)

            ri_tensor = _to_tf_tensor(lidar.range_image_return1)
            mask = ri_tensor[..., 0] > 0
            seg_tensor = _to_tf_tensor(lidar_seg.range_image_return1)
            seg_points = tf.boolean_mask(seg_tensor[..., 1], mask).numpy()

            skitti = np.vectorize(lambda x: WAYMO_TO_SKITTI_LABEL.get(int(x), 0))(seg_points).astype(np.uint32)(skitti & 0xFFFF).astype(np.uint32).tofile(os.path.join(label_dir, f"{frame_idx:06d}.label"))

            weather_f.write(weather_tag + "\n")
            frame_idx += 1

    with open(os.path.join(seq_dir, "calib.txt"), "w") as f:
        f.write("P0: 0 0 0 0 0 0 0 0 0 0 0 0\nTr: 1 0 0 0 0 1 0 0 0 0 1 0\n")

class WaymoParquetConverter:
    def __init__(self, parquet_dir: str, output_dir: str, scene_id_offset: int = 500):
        self.parquet_dir = parquet_dir
        self.output_dir = os.path.expanduser(output_dir)
        self.offset = scene_id_offset
        self.checkpoint = os.path.join(self.output_dir, "converted.txt")
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_done(self) -> set:
        if not os.path.exists(self.checkpoint):
            return set()
        with open(self.checkpoint) as f:
            return {line.strip() for line in f if line.strip()}

    def _mark_done(self, segment_name: str):
        with open(self.checkpoint, "a") as f:
            f.write(segment_name + "\n")

    def convert_all(self):
        print(f"Reading segment names from {self.parquet_dir}...")
        stats_dict = _load_stats(self.parquet_dir)
        all_segments = _get_segment_names(self.parquet_dir)
        done_segments = self._load_done()
        pending = [s for s in all_segments if s not in done_segments]

        print(f"Total: {len(all_segments)}  |  Done: {len(done_segments)}  |  Pending: {len(pending)}")

        for i, segment_name in enumerate(pending):
            scene_id = all_segments.index(segment_name) + self.offset
            weather_tag = stats_dict.get(segment_name, "sunny")
            print(f"  [{i+1}/{len(pending)}] Scene {scene_id:04d}  [{weather_tag}]  {segment_name}")

            try:
                seg_df = _load_segment_df(self.parquet_dir, segment_name)
                _convert_segment(seg_df, scene_id, weather_tag, self.output_dir)
                self._mark_done(segment_name)
            except Exception as e:
                import traceback
                print(f"    ✗ FAILED: {e}")
                traceback.print_exc()
            finally:
                try:
                    del seg_df
                except NameError:
                    pass
                gc.collect()
                tf.keras.backend.clear_session()

if __name__ == "__main__":
    converter = WaymoParquetConverter(
        parquet_dir=INPUT_PARQUET_DIR,
        output_dir=OUTPUT_KITTI_DIR,
        scene_id_offset=500,
    )
    converter.convert_all()