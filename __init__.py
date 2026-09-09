from comfy_api.latest import ComfyExtension

from .nodes import H3HybridWindows
from .control import H3HybridControlNet


class HybridWindowsExtension(ComfyExtension):
    async def get_node_list(self):
        return [H3HybridWindows, H3HybridControlNet]


async def comfy_entrypoint():
    return HybridWindowsExtension()
