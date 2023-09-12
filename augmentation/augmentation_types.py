# -*- coding=utf-8 -*-
from enum import Enum


class AugmentationType(Enum):
	OCCLUSION = "occlusion"


class OcclusionType(Enum):
	INSTANCE_MASK = "instance_mask"
	IMG_BLOCK = "img_block"
	IMG_RECT = "img_rect"
