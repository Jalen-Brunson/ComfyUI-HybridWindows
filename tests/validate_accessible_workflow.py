"""Validate the reduced-dependency 05 graph and its mask semantics on CPU."""

import asyncio
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_accessible_workflow import build, nodes, SOURCE
import execution
import torch

BANNED = ('vlm_video_prompt', 'wan_chunk_io', 'path_tools', 'minimax_h3_mask_tools', 'MaskVidExperiments')
ALLOWED = ('ComfyUI-MMH3Tools', 'ComfyUI-HybridWindows', 'comfyui-videohelpersuite', 'h3_face_tools', 'ComfyUI-Sapiens2', 'ComfyUI-MiniMaxH3Mod')


async def main():
    original = json.loads(SOURCE.read_text())
    saved = {n["id"]: n for n in original["nodes"]}
    paths = [saved[i]["widgets_values_named"]["path"] for i in (257,443,255)]
    graph, api = await build(paths[0], "", "")
    for mask_on, control_on, hair_on in ((False,False,False),(True,False,False),(False,True,False),
                                       (False,False,True),(True,True,True)):
        variant, prompt = await build(paths[0], paths[1] if mask_on else "",
                                      paths[2] if control_on else "",
                                      enable_mask=mask_on, enable_control2=control_on, enable_hair=hair_on)
        result = await execution.validate_prompt('optional-branches', prompt, None)
        assert result[0], json.dumps(result, indent=2, default=str)
        types = {n['class_type']: n for n in prompt.values()}
        cond = types['MMH3ReferenceMultiPrompt']['inputs']
        blur = types['FaceAnonymizeVideo']['inputs']
        composite = types['ImageCompositeMasked']['inputs']
        assert ('ref_videos.ref_video_1' in cond) == control_on
        assert ('denoise_mask' in types['MMH3HybridWindowSampler']['inputs']) == mask_on
        assert ('mask' in composite) == mask_on
        assert ('region_mask' in blur) == hair_on
        assert ('Sapiens2Loader' in types) == hair_on
        assert prompt[blur['insightface'][0]]['class_type'] == 'H3InsightFaceLoader'
        assert types['H3InsightFaceLoader']['inputs']['model_folder'] == 'models/insightface/buffalo_l'
        applied = types['H3RefModCondSetApply']['inputs']
        assert prompt[applied['cond_set'][0]]['class_type'] == 'MMH3ReferenceMultiPrompt'
        assert prompt[applied['mods'][0]]['class_type'] == 'MiniMaxH3RefModsLoader'
        assert prompt[types['MMH3HybridWindowSampler']['inputs']['cond_set'][0]]['class_type'] == 'H3RefModCondSetApply'
        assert applied['insert_position'] == 'before_controls'
        assert {n['class_type'] for n in prompt.values() if n['class_type'].startswith('MiniMaxH3RefMod')} == {'MiniMaxH3RefModsLoader'}
        assert 'protect_mask' not in blur
        source_id = types['MMH3StreamingEncode']['inputs']['images'][0]
        assert blur['images'] == [source_id, 0]
        assert prompt[source_id]['class_type'] == 'VHS_LoadVideoFFmpegPath'
        assert types['ImageAddNoise']['inputs']['strength'] == .10
        assert prompt[cond['ref_videos.ref_video_0'][0]]['class_type'] == 'ImageCompositeMasked'
        assert composite['destination'] == types['ImageAddNoise']['inputs']['image']
    loader = nodes.NODE_CLASS_MAPPINGS['MiniMaxH3RefModsLoader']()
    loader_inputs = next(n['inputs'] for n in api.values() if n['class_type'] == 'MiniMaxH3RefModsLoader')
    bundle, hint = loader.load(**loader_inputs)
    assert bundle == [] and not hint
    original_cond = {'conds': [[[torch.zeros(1), {'minimax_refs': []}]]], 'prompts': ['test']}
    apply = nodes.NODE_CLASS_MAPPINGS['H3RefModCondSetApply']
    assert apply.execute(original_cond, 1., 'before_controls', -1, bundle)[0] is original_cond
    # Exercise the native noise/composite combination on a tiny image batch.
    image = torch.full((2, 8, 8, 3), .5)
    noise_cls = nodes.NODE_CLASS_MAPPINGS['ImageAddNoise']
    composite_cls = nodes.NODE_CLASS_MAPPINGS['ImageCompositeMasked']
    noisy = noise_cls.execute(image, 123, .10)[0]
    m = torch.zeros(2,8,8); m[:,:,4:] = 1
    out = composite_cls.execute(image, noisy, 0, 0, False, m)[0]
    torch.testing.assert_close(out[:,:,:4], image[:,:,:4])
    torch.testing.assert_close(out[:,:,4:], noisy[:,:,4:])
    full = composite_cls.execute(image, noisy, 0, 0, False)[0]
    torch.testing.assert_close(full, noisy)
    assert (image == .5).all()
    # Extract Hair from the real Sapiens2 normalized class-mask convention.
    extract = nodes.NODE_CLASS_MAPPINGS['Sapiens2SegExtract']
    hair = extract.execute(torch.tensor([[[0.,4/28],[3/28,4/28]]]), 'Hair', False, 29)[0]
    torch.testing.assert_close(hair.float(), torch.tensor([[[0.,1.],[0.,1.]]]))
    result = await execution.validate_prompt('hybrid-accessible-validation', api, None)
    assert result[0], json.dumps(result, indent=2, default=str)
    assert not any(any(p in name for p in BANNED) for name in sys.modules)
    for node in graph['nodes']:
        if node['type'] == 'VHS_LoadVideoFFmpegPath':
            state = node['widgets_values']
            assert isinstance(state, dict), 'VHS onConfigure cannot restore positional widget lists'
            assert all(state[key] == value for key, value in node['widgets_values_named'].items())
            assert 'videopreview' in state
    custom = set()
    for n in api.values():
        cls = nodes.NODE_CLASS_MAPPINGS[n['class_type']]
        mod = cls.__module__
        if mod != 'nodes' and not mod.startswith('comfy_extras.'):
            assert any(p in mod for p in ALLOWED), (n['class_type'], mod)
            custom.add(n['class_type'])
    by_id = {n['id']: n for n in graph['nodes']}
    for lid, source, slot, target, inp, kind in graph['links']:
        assert by_id[target]['inputs'][inp]['link'] == lid
        assert lid in by_id[source]['outputs'][slot]['links']
    for i, a in enumerate(graph['nodes']):
        x, y = a['pos']; w, h = a['size']
        for b in graph['nodes'][i+1:]:
            bx, by = b['pos']; bw, bh = b['size']
            assert not (x < bx+bw and bx < x+w and y-30 < by+bh and by-30 < y+h), (a['id'], b['id'])
    sampler = next(n['inputs'] for n in api.values() if n['class_type'] == 'MMH3HybridWindowSampler')
    hybrid = nodes.NODE_CLASS_MAPPINGS["MMH3HybridWindowSampler"]
    helpers = importlib.import_module(hybrid.__module__.rsplit(".", 1)[0] + ".joint")
    from comfy.nested_tensor import NestedTensor
    source = {'samples': NestedTensor([torch.zeros(1,24,62,4,4), torch.zeros(1,32,2,348)])}
    threshold = nodes.NODE_CLASS_MAPPINGS['ThresholdMask']
    solid = nodes.NODE_CLASS_MAPPINGS['SolidMask']
    mask = torch.zeros(209,64,64); mask[:,32:] = 1
    binary = threshold.execute(mask, .5)[0]
    for value in (0.,1.):
        audio = solid.execute(value,1,1)[0]
        prepared = helpers.prepare_masks(source,binary,audio,'max')
        video_mask, audio_mask = prepared['noise_mask'].unbind()
        assert (audio_mask == value).all()
        assert (video_mask[:,:,:,:2] == 0).all()
        assert (video_mask[:,:,:,2:] == 1).all()
    assert sampler['accepted_prefix_frames'] == sampler['start_window'] == 0
    assert next(n['inputs']['value'] for n in api.values() if n['class_type']=='SolidMask') == 0
    print(json.dumps({'prompt_valid': True,'processing_nodes':len(api),'custom_types':sorted(custom),
                      'excluded_packages_imported':False,'layout_overlaps':False,'mask_semantics':'passed',
                      'gpu_render':'not run'},indent=2))


if __name__ == '__main__':
    asyncio.run(main())
