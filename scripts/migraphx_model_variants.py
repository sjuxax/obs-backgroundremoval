# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx>=1.16", "numpy", "onnxruntime"]
# ///
"""Generate the MIGraphX-specific model variants in data/models/migraphx/.

The plugin loads models/migraphx/<name>.onnx instead of models/<name>.onnx when
the MIGraphX execution provider is selected and the variant exists. Each variant
is numerically equivalent to the original; it only avoids a MIGraphX problem:

  rvm_mobilenetv3_fp32.onnx
      Every input is dynamic and downsample_ratio is a runtime input, so MIGraphX
      parses AveragePool(count_include_pad) on a dynamic shape and rejects it.
      The plugin always runs RVM at 192x192 with downsample_ratio=1, so pin those.
      downsample_ratio stays a (now unused) graph input so the plugin's feeding
      code is unchanged.

  tbefn_fp32.onnx
      MIGraphX's simplify_algebra (find_conv_concat_split_fuse) rewrites
      Conv(Concat(a, b)) and emits a use-before-def, aborting the compile.
      Pre-split those convs into Conv(a) + Conv(b) so the pass never matches.

Usage: uv run scripts/migraphx_model_variants.py [--models-dir data/models]
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference

RVM_SIDE = 192
RVM_STATE_CHANNELS = (16, 20, 40, 64)


def rvm_state_shape(level: int) -> list[int]:
    scale = 2 << (level - 1)
    return [1, RVM_STATE_CHANNELS[level - 1], RVM_SIDE // scale, RVM_SIDE // scale]


def set_dims(value_info: onnx.ValueInfoProto, dims: list[int]) -> None:
    for dim, size in zip(value_info.type.tensor_type.shape.dim, dims):
        dim.Clear()
        dim.dim_value = size


def make_static_rvm(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    graph.initializer.append(
        numpy_helper.from_array(np.array([1.0], np.float32), "downsample_ratio_const")
    )
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == "downsample_ratio":
                node.input[index] = "downsample_ratio_const"

    output_channels = {"fgr": 3, "pha": 1}
    for value in list(graph.input) + list(graph.output):
        if value.name == "src":
            set_dims(value, [1, 3, RVM_SIDE, RVM_SIDE])
        elif value.name in output_channels:
            set_dims(value, [1, output_channels[value.name], RVM_SIDE, RVM_SIDE])
        elif value.name[:1] == "r" and value.name[1:2].isdigit():
            set_dims(value, rvm_state_shape(int(value.name[1])))

    # Stale symbolic value_info would otherwise survive shape inference.
    del graph.value_info[:]
    return shape_inference.infer_shapes(model, data_prop=True)


def split_concat_convs(model: onnx.ModelProto) -> onnx.ModelProto:
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    producers = {out: node for node in graph.node for out in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    channel_counts = {
        value.name: value.type.tensor_type.shape.dim[1].dim_value
        for value in list(graph.value_info) + list(graph.input)
        if len(value.type.tensor_type.shape.dim) > 1
    }

    def attribute(node: onnx.NodeProto, key: str, default=None):
        for attr in node.attribute:
            if attr.name == key:
                return helper.get_attribute_value(attr)
        return default

    new_nodes = []
    for node in graph.node:
        concat = producers.get(node.input[0]) if node.op_type == "Conv" else None
        if concat is None:
            new_nodes.append(node)
            continue
        # Same precondition as find_conv_concat_split_fuse: a concat input also
        # feeds another conv.
        matches = (
            concat.op_type == "Concat"
            and attribute(concat, "axis") in (1, -3)
            and attribute(node, "group", 1) == 1
            and node.input[1] in initializers
            and any(
                other.op_type == "Conv" and other is not node
                for tensor in concat.input
                for other in consumers.get(tensor, [])
            )
        )
        if not matches:
            new_nodes.append(node)
            continue

        weight = numpy_helper.to_array(initializers[node.input[1]])
        bias = node.input[2] if len(node.input) > 2 else None
        conv_attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        offset = 0
        partial_outputs = []
        for index, tensor in enumerate(concat.input):
            width = channel_counts[tensor]
            weight_name = f"{node.name}__split_w{index}"
            graph.initializer.append(
                numpy_helper.from_array(
                    np.ascontiguousarray(weight[:, offset : offset + width]), weight_name
                )
            )
            offset += width
            inputs = [tensor, weight_name] + ([bias] if bias and index == 0 else [])
            output = f"{node.name}__split_y{index}"
            new_nodes.append(helper.make_node("Conv", inputs, [output], name=output, **conv_attributes))
            partial_outputs.append(output)
        if offset != weight.shape[1]:
            raise ValueError(f"{node.name}: concat widths {offset} != weight in-channels {weight.shape[1]}")

        accumulated = partial_outputs[0]
        for index, partial in enumerate(partial_outputs[1:], start=1):
            last = index == len(partial_outputs) - 1
            output = node.output[0] if last else f"{node.name}__split_sum{index}"
            new_nodes.append(helper.make_node("Add", [accumulated, partial], [output], name=output))
            accumulated = output
        print(f"  split {node.name}: {len(concat.input)} concat inputs, weight {weight.shape}")

    del graph.node[:]
    graph.node.extend(new_nodes)
    used = {name for node in graph.node for name in node.input}
    kept = [init for init in graph.initializer if init.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    del graph.value_info[:]
    return model


def max_cpu_difference(original: Path, variant: Path, feeds: dict, steps: int = 1) -> float:
    """Run both models on CPU and return the worst absolute output difference.

    For RVM, recurrent states are fed back for `steps` iterations like the plugin does.
    """
    sessions = [
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for path in (original, variant)
    ]
    results = []
    for session in sessions:
        current = dict(feeds)
        outputs = []
        for _ in range(steps):
            outputs = session.run(None, current)
            for level in range(1, 5):
                if f"r{level}i" in current:
                    current[f"r{level}i"] = outputs[level + 1]
        results.append(outputs)
    return max(float(np.max(np.abs(a - b))) for a, b in zip(*results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, default=Path("data/models"))
    args = parser.parse_args()
    out_dir = args.models_dir / "migraphx"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    rvm_in = args.models_dir / "rvm_mobilenetv3_fp32.onnx"
    rvm_out = out_dir / rvm_in.name
    print(f"[1/2] {rvm_in} -> {rvm_out}")
    rvm = make_static_rvm(onnx.load(str(rvm_in)))
    onnx.checker.check_model(rvm)
    onnx.save(rvm, str(rvm_out))
    rvm_feeds = {
        "src": rng.random((1, 3, RVM_SIDE, RVM_SIDE), dtype=np.float32),
        "downsample_ratio": np.array([1.0], np.float32),
    }
    for level in range(1, 5):
        rvm_feeds[f"r{level}i"] = np.zeros(rvm_state_shape(level), np.float32)
    print(f"  CPU max diff vs original (4 recurrent steps): {max_cpu_difference(rvm_in, rvm_out, rvm_feeds, 4):.3e}")

    tbefn_in = args.models_dir / "tbefn_fp32.onnx"
    tbefn_out = out_dir / tbefn_in.name
    print(f"[2/2] {tbefn_in} -> {tbefn_out}")
    tbefn = split_concat_convs(onnx.load(str(tbefn_in)))
    onnx.checker.check_model(tbefn)
    onnx.save(tbefn, str(tbefn_out))
    tbefn_feeds = {"Placeholder": rng.random((1, 3, 180, 320), dtype=np.float32)}
    print(f"  CPU max diff vs original: {max_cpu_difference(tbefn_in, tbefn_out, tbefn_feeds):.3e}")


if __name__ == "__main__":
    main()
