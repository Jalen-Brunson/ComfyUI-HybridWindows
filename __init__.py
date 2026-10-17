from comfy_api.latest import ComfyExtension

from .nodes import H3HybridWindows
from .upscale import H3HybridWarmupPrior, H3HybridLatentUpscale, H3HybridCleanLatentUpscale
from .keyframes import H3HybridKeyframes
from .sampler import MMH3HybridWindowSampler
from .control import H3HybridControlNet
from .color import VideoColorStabilize
from .segmented_blur import H3SegmentedVideoBlur
from .resume import (H3HybridRunPlan, H3HybridResumeLoad, H3HybridResumeLatent,
                     H3HybridResumeOutput, H3HybridSaveRun)


class HybridWindowsExtension(ComfyExtension):
    async def get_node_list(self):
        return [H3HybridWarmupPrior, H3HybridLatentUpscale, H3HybridCleanLatentUpscale, H3HybridWindows, H3HybridKeyframes, H3HybridControlNet, VideoColorStabilize, MMH3HybridWindowSampler,
                H3SegmentedVideoBlur, H3HybridRunPlan, H3HybridResumeLoad, H3HybridResumeLatent,
                H3HybridResumeOutput, H3HybridSaveRun]


async def comfy_entrypoint():
    return HybridWindowsExtension()
