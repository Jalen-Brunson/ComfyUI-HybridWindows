"""Expose the existing segmentation-only H3 blur with controls for that path."""
import nodes
from comfy_api.latest import io


class H3SegmentedVideoBlur(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='H3SegmentedVideoBlur', display_name='Blur Segmented Face and Hair', category='image/video',
            description='Blur the segmentation region while protecting lips. Uses h3_face_tools with face detection and likeness measurement disabled.',
            inputs=[io.Image.Input('images'), io.Boolean.Input('enabled', default=True),
                    io.Float.Input('strength', default=.5, min=0., max=1., step=.01),
                    io.Int.Input('region_grow', default=2, min=0, max=100),
                    io.Int.Input('protect_grow', default=2, min=0, max=100),
                    io.Float.Input('min_component', default=.05, min=0., max=1., step=.01),
                    io.Mask.Input('region_mask', optional=True, lazy=True),
                    io.Mask.Input('protect_mask', optional=True, lazy=True)],
            outputs=[io.Image.Output('images'), io.String.Output('report')])

    @classmethod
    def check_lazy_status(cls, enabled=True, strength=.5, region_mask=None, protect_mask=None, **kwargs):
        if not enabled or strength == 0:
            return []
        return (['region_mask'] if region_mask is None else []) + (['protect_mask'] if protect_mask is None else [])

    @classmethod
    def execute(cls, images, enabled=True, strength=.5, region_grow=2, protect_grow=2, min_component=.05,
                region_mask=None, protect_mask=None):
        if not enabled or strength == 0:
            return io.NodeOutput(images, 'Segmentation blur disabled.')
        if region_mask is None:
            raise ValueError('Connect the face and hair segmentation mask, or disable blur.')
        if 'FaceAnonymizeVideo' not in nodes.NODE_CLASS_MAPPINGS:
            raise RuntimeError('Install/update h3_face_tools and restart ComfyUI for segmentation blur.')
        result = nodes.NODE_CLASS_MAPPINGS['FaceAnonymizeVideo']().run(
            images=images, enabled=True, gender='any', keep_mouth=False, mouth_line=.73, expand=0.,
            mode='blur', strength=strength, hold=0, ema=0., all_faces=True, measure_residual=False,
            provider='cpu', unload_after=False, output_mask=False, face_detector=False,
            region_mask=region_mask, region_mode='blur', region_grow=region_grow,
            protect_mask=protect_mask, protect_grow=protect_grow, region_min_component=min_component)
        return io.NodeOutput(result[0], result[2])
