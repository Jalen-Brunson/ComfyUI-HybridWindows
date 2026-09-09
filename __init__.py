from comfy_api.latest import ComfyExtension

from .nodes import H3HybridWindows
from .control import H3HybridControlNet
from .color import VideoColorStabilize


class HybridWindowsExtension(ComfyExtension):
    async def get_node_list(self):
        return [H3HybridWindows, H3HybridControlNet, VideoColorStabilize]


async def comfy_entrypoint():
    return HybridWindowsExtension()
