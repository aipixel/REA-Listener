from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VicoLayout:
    root: Path
    metadata_path: Path
    emotion_dir: Path
    audio_stats_path: Path
    keypoint_stats_104_path: Path
    keypoint_stats_468_path: Path
    coef_mean_path: Path
    coef_std_path: Path
    reference_image_dir: Path
    listener_original_dir: Path
    speaker_keypoint_dir: Path


@dataclass(frozen=True)
class RealTalkLayout:
    root: Path
    split_path: Path
    coef_norm_path: Path
    emotion_dir: Path
    audio_stats_path: Path
    keypoint_stats_104_path: Path
    reference_image_dir: Path
    listener_track_dir: Path
    speaker_track_dir: Path
    speaker_keypoint_dir: Path


def build_vico_layout(root: str | Path) -> VicoLayout:
    root_path = Path(root).expanduser().resolve()
    return VicoLayout(
        root=root_path,
        metadata_path=root_path / "RLD_data_with_val_and_length.xlsx",
        emotion_dir=root_path / "emotion_annotations",
        audio_stats_path=root_path / "audio_feats_mean_std.bin",
        keypoint_stats_104_path=root_path / "video_keypoint_104_feats_mean_std_vico.bin",
        keypoint_stats_468_path=root_path / "video_keypoint_468_feats_mean_std_vico.bin",
        coef_mean_path=root_path / "train_mean_no_basetm_dict.pkl",
        coef_std_path=root_path / "train_std_no_basetm_dict.pkl",
        reference_image_dir=root_path / "reference_image",
        listener_original_dir=root_path / "listener_coef_original",
        speaker_keypoint_dir=root_path / "video_keypoint_features",
    )


def build_realtalk_layout(root: str | Path) -> RealTalkLayout:
    root_path = Path(root).expanduser().resolve()
    return RealTalkLayout(
        root=root_path,
        split_path=root_path / "benchmark_train_test_split_filtered3.json",
        coef_norm_path=root_path / "coef_norm_std.pkl",
        emotion_dir=root_path / "listener_emotion",
        audio_stats_path=root_path / "audio_feats_mean_std_realtalk.bin",
        keypoint_stats_104_path=root_path / "video_keypoint_104_feats_mean_std_realtalk.bin",
        reference_image_dir=root_path / "reference_image",
        listener_track_dir=root_path / "track_realtalk_listener",
        speaker_track_dir=root_path / "track_realtalk_speaker",
        speaker_keypoint_dir=root_path / "video_keypoint_features",
    )

