import asyncio
import importlib
import json
import math
import os
import unittest
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get('COMFYUI_PATH', ROOT.parents[1]))
EXAMPLES = ROOT / 'example_workflows'
sys.path.insert(0, str(ROOT / 'tests'))
import test_clean_upscale as clean
import test_native_chain as native
import execution
import folder_paths
from comfy_extras.nodes_math import MathExpressionNode
from comfy.ldm.minimax.model import PackedLayout


async def main():
    folder_paths.add_model_folder_path('latent_upscale_models', str(COMFY / 'models/latent_upscale_models'))
    for name in ('nodes_custom_sampler', 'nodes_audio', 'nodes_minimax_h3', 'nodes_primitive',
                 'nodes_video', 'nodes_math', 'nodes_preview_any'):
        module = importlib.import_module('comfy_extras.' + name)
        if hasattr(module, 'comfy_entrypoint'):
            extension = await module.comfy_entrypoint()
            for cls in await extension.get_node_list():
                native.native_nodes.NODE_CLASS_MAPPINGS[cls.GET_SCHEMA().node_id] = cls
        else:
            native.native_nodes.NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    extension = await native.pack.comfy_entrypoint()
    for cls in await extension.get_node_list():
        native.native_nodes.NODE_CLASS_MAPPINGS[cls.GET_SCHEMA().node_id] = cls

    results = {}
    for path in sorted(EXAMPLES.glob('*PDD + DMD latent upscale.json')):
        workflow = json.loads(path.read_text())
        ns = {n['id']: n for n in workflow['nodes']}
        links = {l[0]: l for l in workflow['links']}
        assert len(ns) == len(workflow['nodes'])
        assert len(links) == len(workflow['links'])
        for lid, source, slot, target, index, typ in links.values():
            assert ns[target]['inputs'][index]['link'] == lid
            assert lid in ns[source]['outputs'][slot]['links']
            assert ns[source]['outputs'][slot]['type'] == typ
        for n in ns.values():
            for i, socket in enumerate(n['inputs']):
                if socket.get('link') is not None:
                    assert links[socket['link']][3:5] == [n['id'], i]
            for i, socket in enumerate(n['outputs']):
                for lid in socket.get('links') or []:
                    assert links[lid][1:3] == [n['id'], i]
        prompt = json.loads((EXAMPLES / (path.stem + '.api.json')).read_text())
        # Check the UI widget/link graph serializes to the API graph being validated.
        for node_id, node in prompt.items():
            ui = ns[int(node_id)]
            expected = dict(ui['widgets_values_named'])
            for socket in ui['inputs']:
                if socket.get('link') is not None:
                    link = links[socket['link']]
                    expected[socket['name']] = [str(link[1]), link[2]]
            assert expected == node['inputs']
        valid = await execution.validate_prompt(str(uuid.uuid4()), prompt, None)
        results[path.name] = {'valid': valid[0], 'error': valid[1], 'node_errors': valid[3],
                              'nodes': len(ns), 'links': len(links)}
        print(path.name, json.dumps(results[path.name]), flush=True)
        assert valid[0] and not valid[3], valid
        # The PDD and DMD loaders fork directly from the two hybrid outputs.
        loras = [n for n in prompt.values() if n['class_type'] == 'LoraLoaderModelOnly']
        assert len(loras) == 2
        a, b = [n['inputs']['model'] for n in loras]
        assert a[0] == b[0] and [a[1], b[1]] == [0, 1]
        hybrid = prompt[a[0]]
        assert hybrid['class_type'] == 'H3HybridWindows' and 'latent' in hybrid['inputs']
        assert hybrid['inputs']['total_steps'] == 9

    case = clean.CleanUpscaleTests()
    model, joint, cond, warm, _ = case.warmup(split=7)
    sigmas = native.torch.tensor([0.631578922, 0.3158, 0.0])
    assert math.isclose(clean.up.joint_run(joint).stash.sigma_end, float(sigmas[0]), abs_tol=1e-7)
    assert len(model.model.seen) == 14
    upgraded, latent, report = case.upscale(joint, warm)
    finished = case.finish(upgraded, latent, cond, sigmas)
    assert len(model.model.seen) == 18
    assert all(native.torch.isfinite(p).all() for p in finished['samples'].unbind())
    old = clean.up.joint_run(joint).stash
    new = clean.up.joint_run(upgraded).stash
    old_audio = native.comfy.utils.unpack_latents(old.raw_state, old.signature['shapes'])[1]
    new_audio = native.comfy.utils.unpack_latents(new.raw_state, new.signature['shapes'])[1]
    native.torch.testing.assert_close(old_audio, new_audio, rtol=0, atol=0)
    results['handoff'] = {'sequential_calls': 14, 'joint_calls': 4, 'windows': 2,
                           'sigmas': sigmas.tolist(), 'audio_preserved': True, 'finite_output': True}
    print('CPU native 7-step clean upscale + exact reference sigmas: PASS', flush=True)

    fl = json.loads((EXAMPLES / 'FL2V Hybrid Native - PDD + DMD latent upscale.api.json').read_text())
    dimension_exprs = [n['inputs']['expression'] for n in fl.values()
                       if n['class_type'] == 'ComfyMathExpression' and n['_meta']['title'].startswith('Final ')]
    dimensions = []
    for width, height, mp in ((480, 832, .8), (832, 480, .8), (640, 384, 1.2), (480, 832, 2.4)):
        high_w, high_h = [MathExpressionNode.execute(e, {'a': mp, 'b': width, 'c': height})[1]
                          for e in dimension_exprs]
        video = native.torch.zeros(1, 24, 1, height // 16, width // 16)
        expected = clean.up.target_size(video, mp)
        assert (high_h // 16, high_w // 16) == expected
        guide = native.torch.zeros(1, 24, 1, high_h // 16, high_w // 16)
        layout = PackedLayout(10, 12, high_h // 16, high_w // 16, 65,
                              keyframes=[{'resolved_frame_index': 0, 'latent': guide}])
        rows = next(end - start for start, end, kind in layout.segments if kind == 'cond')
        assert rows == math.ceil(high_h / 32) * math.ceil(high_w / 32)
        dimensions.append([width, height, mp, high_w, high_h])
    results['FL2V_guide_sizes'] = dimensions
    print('FL2V final guide dimensions and native packed rows: PASS', dimensions, flush=True)
    print(json.dumps(results, indent=2), flush=True)


asyncio.run(main())

suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in
                           ('test_native_chain', 'test_upscale', 'test_clean_upscale'))
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
