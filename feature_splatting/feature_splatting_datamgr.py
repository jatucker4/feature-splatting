import gc
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type
from nerfstudio.cameras.cameras import Cameras, CameraType
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import os
import numpy as np
import torch
from jaxtyping import Float
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE

from feature_splatting.feature_extractor_cfg import SAMCLIPArgs
from unimatch.main_flow import get_args_parser
# SAMCLIP
from feature_splatting.feature_extractor import batch_extract_feature, build_flow_model, inference_flow_filenames, run_flow_inference

feat_type_to_extract_fn = {
    "CLIP": None,
    "DINO": None,
    "SAMCLIP": batch_extract_feature,
}

feat_type_to_args = {
    "CLIP": None,
    "DINO": None,
    "SAMCLIP": SAMCLIPArgs,
}

feat_type_to_main_feature_name = {
    "CLIP": "clip",
    "DINO": "dino",
    "SAMCLIP": "samclip",
}

@dataclass
class FeatureSplattingDataManagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: FeatureSplattingDataManager)
    feature_type: Literal["CLIP", "DINO", "SAMCLIP"] = "SAMCLIP"
    """Feature type to extract."""
    enable_cache: bool = True
    """Whether to cache extracted features."""

class FeatureSplattingDataManager(FullImageDatamanager):
    config: FeatureSplattingDataManagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Extract features
        self.feature_dict = self.extract_features()

        # Split into train and eval features
        self.train_feature_dict = {}
        self.eval_feature_dict = {}
        feature_dim_dict = {}
        for feature_name in self.feature_dict:
            assert len(self.feature_dict[feature_name]) == len(self.train_dataset) + len(self.eval_dataset)
            self.train_feature_dict[feature_name] = self.feature_dict[feature_name][: len(self.train_dataset)]
            self.eval_feature_dict[feature_name] = self.feature_dict[feature_name][len(self.train_dataset) :]
            feature_dim_dict[feature_name] = self.feature_dict[feature_name].shape[1:]  # c, h, w
        assert len(self.eval_feature_dict[feature_name]) == len(self.eval_dataset)

        del self.feature_dict

        # Set metadata, so we can initialize model with feature dimensionality
        self.train_dataset.metadata["feature_type"] = self.config.feature_type
        self.train_dataset.metadata["feature_dim_dict"] = feature_dim_dict
        self.train_dataset.metadata["main_feature_name"] = feat_type_to_main_feature_name[self.config.feature_type]
        self.train_dataset.metadata["clip_model_name"] = feat_type_to_args[self.config.feature_type].clip_model_name

        # Garbage collect
        torch.cuda.empty_cache()
        gc.collect()
    
    def extract_features(self) -> Dict[str, Float[torch.Tensor, "n h w c"]]:
        # Extract features
        if self.config.feature_type not in feat_type_to_extract_fn:
            raise ValueError(f"Unknown feature type {self.config.feature_type}")
        extract_fn = feat_type_to_extract_fn[self.config.feature_type]
        extract_args = feat_type_to_args[self.config.feature_type]
        image_fnames = self.train_dataset.image_filenames + self.eval_dataset.image_filenames
        self.max_frames = len(image_fnames)
        # self.delta_t = 10/self.max_frames
        self.delta_t = 1
        # Cache path for all features
        cache_dir = self.config.dataparser.data
        cache_path = cache_dir / f"feature_splatting_{self.config.feature_type.lower()}_features.pt"
        flow_cache_path = cache_dir / "feature_splatting_flow_features.pt"

        # Load cached features if available
        if self.config.enable_cache and cache_path.exists() and flow_cache_path.exists():
            cache_dict = torch.load(cache_path)
            flow_cache_dict = torch.load(flow_cache_path)
            if cache_dict.get("image_fnames") != image_fnames or flow_cache_dict.get("image_fnames") != image_fnames:
                CONSOLE.print("Image filenames have changed, cache invalidated...")
            elif cache_dict.get("args") != extract_args.id_dict():
                CONSOLE.print("Feature extraction args have changed, cache invalidated...")
            else:
                return {**cache_dict["feature_dict"], "flow_features": flow_cache_dict["flow_features"]}

        # Extract SAMCLIP/other features
        CONSOLE.print(f"Extracting {self.config.feature_type} features for {len(image_fnames)} images...")
        feature_dict = extract_fn(image_fnames, extract_args)

        # Extract Flow Features using CoTracker
        CONSOLE.print(f"Extracting flow features for {len(image_fnames)} images using UniMatch...")
        # Get Unimatch model
        flow_model = build_flow_model()
        all_images = self.cached_train + self.cached_eval
        flow_features_np = run_flow_inference(flow_model, all_images)
        flow_features = torch.from_numpy(flow_features_np)

        # Cache the extracted features
        if self.config.enable_cache:
            cache_dict = {"args": extract_args.id_dict(), "image_fnames": image_fnames, "feature_dict": feature_dict}
            flow_cache_dict = {"image_fnames": image_fnames, "flow_features": flow_features}
            cache_dir.mkdir(exist_ok=True)
            torch.save(cache_dict, cache_path)
            torch.save(flow_cache_dict, flow_cache_path)
            CONSOLE.print(f"Saved {self.config.feature_type} features to cache at {cache_path}")
            CONSOLE.print(f"Saved flow features to cache at {flow_cache_path}")
        
        feature_dict["flow_features"] = flow_features
        return feature_dict

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        camera, data = super().next_train(step)
        camera_idx = camera.metadata['cam_idx']
        feature_dict = {}
        for feature_name in self.train_feature_dict:
            feature_dict[feature_name] = self.train_feature_dict[feature_name][camera_idx]
        data["feature_dict"] = feature_dict
        camera.delta_t = self.delta_t
        # Based on camera index we can find the timestep. The delta t is constant throughout (evenly spaced samples).
        if camera_idx == 0:
            camera.time = 0
        else:
            # camera.time = camera_idx*self.delta_t
            camera.time = camera_idx
        
        return camera, data
    
    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        camera, data = super().next_eval(step)
        camera_idx = camera.metadata['cam_idx']
        feature_dict = {}
        for feature_name in self.eval_feature_dict:
            feature_dict[feature_name] = self.eval_feature_dict[feature_name][camera_idx]
        data["feature_dict"] = feature_dict
        return camera, data
