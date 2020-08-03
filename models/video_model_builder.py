#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from slowfast.models import ResNet, SlowFast
from slowfast.models import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class SlowFastFeat(SlowFast):
    # Overwrite forward function to return features
    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
            return x
        else:
            x, feat = self.head(x)
            return x, feat


@MODEL_REGISTRY.register()
class ResNetFeat(ResNet):
    # Overwrite forward function to return features
    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
            return x
        else:
            x, feat = self.head(x)
            return x, feat
