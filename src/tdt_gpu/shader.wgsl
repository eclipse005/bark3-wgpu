// GPU TDT greedy decoder.
//
// No subgroup builtins: Chrome exposes them only behind the `subgroups` WGSL
// extension, which wgpu's web backend cannot request, so the horizontal sums and
// the joint-head exchange go through workgroup memory. Same add order, same
// result, one source for every backend.
//
// Per step the host records four dispatches:
//   joint  (65 WGs) : score every joint row, per-WG argmax
//   layer0 (40 WGs) : reduce partials + control, then LSTM layer 0
//   layer1 (40 WGs) : LSTM layer 1
//   project(40 WGs) : decoder projection
//
// LSTM and projection rows are split by channel across workgroups so the full
// weight matrix is streamed from many SMs, not one.
//
// Numerics: each 16-lane group follows the CPU AVX2 dot order. Lane j owns K
// residue j and walks K = j, j+16, j+32, ... in increasing order; lane 0 then
// sums the eight (a[j] + a[j+8]) pairs strictly left to right. The host repacks
// every decoder matrix into that same lane order, so a lane reads its 40 K
// values as five contiguous `vec4` words without changing any accumulation
// sequence.

struct Params {
    t_len: u32,
    max_steps: u32,
    _a: u32,
    _b: u32,
};

struct Partial {
    token_score: f32,
    token_id: u32,
    duration_score: f32,
    duration_id: u32,
};

@group(0) @binding(0) var<storage, read>       Pooler:   array<f32>;
@group(0) @binding(1) var<storage, read>       Embed:    array<f32>;
@group(0) @binding(2) var<storage, read>       Weights:  array<vec4<u32>>;
@group(0) @binding(3) var<storage, read>       Biases:   array<f32>;
@group(0) @binding(4) var<storage, read_write> State:    array<f32>;
@group(0) @binding(5) var<storage, read_write> Result:   array<u32>;
@group(0) @binding(6) var<storage, read_write> Partials: array<Partial>;
@group(0) @binding(7) var<uniform>             params:   Params;

const DH: u32 = 640u;
const GS: u32 = 2560u;
const VOCAB: u32 = 8193u;
const N_ROWS: u32 = 8198u;
const BLANK: u32 = 8192u;
const WG: u32 = 256u;
const LANES: u32 = 16u;
const ROW_GROUPS: u32 = WG / LANES;
const ROWS_PER_WG: u32 = 128u;
const JOINT_GROUPS: u32 = 65u;
const JOINT_TILES: u32 = ROWS_PER_WG / ROW_GROUPS;
const LAYER_GROUPS: u32 = 40u;
const CHANNELS_PER_GROUP: u32 = DH / LAYER_GROUPS;
const LAYER_ROWS: u32 = 4u * CHANNELS_PER_GROUP;
const LAYER_TILES: u32 = (LAYER_ROWS + ROW_GROUPS - 1u) / ROW_GROUPS;
const PROJ_TILES: u32 = (CHANNELS_PER_GROUP + ROW_GROUPS - 1u) / ROW_GROUPS;

// Weight arena, in vec4 words (eight f16 per word) of the lane-major layout.
const ROW4: u32 = DH / 8u;
const LANE4: u32 = DH / 128u;
const W_IH: u32 = 0u;
const W_HH: u32 = 409600u;
const W_HEAD: u32 = 819200u;
const W_PROJ: u32 = 1475040u;

// f32 biases.
const B_IH: u32 = 0u;
const B_HH: u32 = 5120u;
const B_HEAD: u32 = 10240u;
const B_PROJ: u32 = 18438u;

// f32 state offsets.
const S_H0A: u32 = 0u;
const S_H0B: u32 = 640u;
const S_H1A: u32 = 1280u;
const S_H1B: u32 = 1920u;
const S_C0: u32 = 2560u;
const S_C1: u32 = 3200u;
const S_DEC: u32 = 3840u;

// Result header. Tokens start at RESULT_TOKEN_BASE, TOKEN_WORDS apart.
const R_COUNT: u32 = 0u;
const R_FRAME: u32 = 1u;
const R_STEPS: u32 = 2u;
const R_EXIT: u32 = 3u;
const R_PARITY: u32 = 4u;
const R_ACTIVE: u32 = 5u;
const RESULT_TOKEN_BASE: u32 = 6u;

var<workgroup> red_score: array<f32, 256>;
var<workgroup> red_id: array<u32, 256>;
var<workgroup> red_dur_score: array<f32, 256>;
var<workgroup> red_dur_id: array<u32, 256>;
var<workgroup> x_stage: array<f32, 2u * DH>;
var<workgroup> layer_gates: array<f32, LAYER_ROWS>;
var<workgroup> chosen: array<u32, 2>;
// Per-lane staging for the horizontal sums, plus the joint head's cross-row
// exchange: a lane swaps with lane ^ 16, its partner in the 32-lane pair.
var<workgroup> lane_stage: array<f32, WG>;
var<workgroup> exch_score: array<f32, WG>;
var<workgroup> exch_id: array<u32, WG>;
var<workgroup> exch_dur_score: array<f32, WG>;
var<workgroup> exch_dur_id: array<u32, WG>;

// Tint refuses a barrier that sits in control flow derived from a storage-buffer
// read, and every predicate below is one. Passing the value through workgroup
// memory with \`workgroupUniformLoad\` makes it uniform by construction, so the
// kernels keep their early returns and their barriers.
var<workgroup> wg_flag: u32;

fn sync_flag(v: u32) -> bool {
    wg_flag = v;
    workgroupBarrier();
    return workgroupUniformLoad(&wg_flag) != 0u;
}

// Lane `j`'s K values live at `row4 + j * LANE4`, eight halves per vec4, in
// increasing K residue order.
fn dot_row(x_base: u32, row4: u32, lane: u32) -> f32 {
    var acc = 0.0;
    let base = row4 + lane * LANE4;
    for (var t = 0u; t < LANE4; t = t + 1u) {
        let w = Weights[base + t];
        let p0 = unpack2x16float(w.x);
        let p1 = unpack2x16float(w.y);
        let p2 = unpack2x16float(w.z);
        let p3 = unpack2x16float(w.w);
        let k = x_base + lane + t * 128u;
        acc = fma(p0.x, x_stage[k], acc);
        acc = fma(p0.y, x_stage[k + 16u], acc);
        acc = fma(p1.x, x_stage[k + 32u], acc);
        acc = fma(p1.y, x_stage[k + 48u], acc);
        acc = fma(p2.x, x_stage[k + 64u], acc);
        acc = fma(p2.y, x_stage[k + 80u], acc);
        acc = fma(p3.x, x_stage[k + 96u], acc);
        acc = fma(p3.y, x_stage[k + 112u], acc);
    }
    return acc;
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// One 16-lane row group holds one K residue per lane. Combine lane j with j+8,
// then add the eight pairs strictly left to right: the CPU horizontal reduction,
// unchanged by the move to workgroup memory.
fn sum_row(partial: f32, tid: u32, group: u32) -> f32 {
    lane_stage[tid] = partial;
    workgroupBarrier();
    let base = group * LANES;
    var sum = 0.0;
    for (var j = 0u; j < 8u; j = j + 1u) {
        sum = sum + (lane_stage[base + j] + lane_stage[base + j + 8u]);
    }
    workgroupBarrier();
    return sum;
}

// Cross-workgroup argmax: stage the 129 partials and collapse them in place.
// Equal scores keep the lower id, so the whole sequence is order independent.
fn reduce_joint(tid: u32) {
    var token_score = -3.4028234e38;
    var token_id = 0xffffffffu;
    var duration_score = -3.4028234e38;
    var duration_id = 0xffffffffu;
    if (tid < JOINT_GROUPS) {
        token_score = Partials[tid].token_score;
        token_id = Partials[tid].token_id;
        duration_score = Partials[tid].duration_score;
        duration_id = Partials[tid].duration_id;
    }
    red_score[tid] = token_score;
    red_id[tid] = token_id;
    red_dur_score[tid] = duration_score;
    red_dur_id[tid] = duration_id;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (tid < stride) {
            let other = tid + stride;
            if (red_score[other] > red_score[tid]
                || (red_score[other] == red_score[tid] && red_id[other] < red_id[tid])) {
                red_score[tid] = red_score[other];
                red_id[tid] = red_id[other];
            }
            if (red_dur_score[other] > red_dur_score[tid]
                || (red_dur_score[other] == red_dur_score[tid]
                    && red_dur_id[other] < red_dur_id[tid])) {
                red_dur_score[tid] = red_dur_score[other];
                red_dur_id[tid] = red_dur_id[other];
            }
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (tid == 0u) {
        chosen[0] = red_id[0];
        chosen[1] = red_dur_id[0];
    }
    workgroupBarrier();
}

fn run_layer(
    layer: u32,
    h_new: u32,
    c_base: u32,
    channel_base: u32,
    tid: u32,
    group: u32,
    lane: u32,
) {
    let w_layer = layer * GS * DH / 8u;
    let b_layer = layer * GS;

    for (var tile = 0u; tile < LAYER_TILES; tile = tile + 1u) {
        let local = tile * ROW_GROUPS + group;
        let valid_local = local < LAYER_ROWS;
        let gate = min(local / CHANNELS_PER_GROUP, 3u);
        let channel = channel_base + local % CHANNELS_PER_GROUP;
        let row4 = (gate * DH + channel) * ROW4;
        let dot_x = dot_row(0u, W_IH + w_layer + row4, lane);
        let dot_h = dot_row(DH, W_HH + w_layer + row4, lane);
        let sum_x = sum_row(dot_x, tid, group);
        let sum_h = sum_row(dot_h, tid, group);
        if (lane == 0u && valid_local) {
            let row = gate * DH + channel;
            layer_gates[local] = ((Biases[B_IH + b_layer + row] + Biases[B_HH + b_layer + row])
                + sum_x) + sum_h;
        }
    }
    workgroupBarrier();

    if (tid < CHANNELS_PER_GROUP) {
        let channel = channel_base + tid;
        let input = sigmoid(layer_gates[tid]);
        let forget = sigmoid(layer_gates[CHANNELS_PER_GROUP + tid]);
        let candidate = tanh(layer_gates[2u * CHANNELS_PER_GROUP + tid]);
        let output = sigmoid(layer_gates[3u * CHANNELS_PER_GROUP + tid]);
        let cell = forget * State[c_base + channel] + input * candidate;
        State[c_base + channel] = cell;
        State[h_new + channel] = output * tanh(cell);
    }
    workgroupBarrier();
}

@compute @workgroup_size(WG)
fn joint_main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let group = tid / LANES;
    let lane = tid % LANES;
    let frame = Result[R_FRAME];
    // A finished decoder leaves the workgroup idle instead of returning early:
    // every invocation has to reach the barriers below. Skipping the scoring
    // leaves the sentinels in place, which is the partial a finished step has to
    // publish anyway.
    let idle = sync_flag(select(0u, 1u, frame >= params.t_len || Result[R_EXIT] != 0u));

    var best_token_score = -3.4028234e38;
    var best_token_id = 0xffffffffu;
    var best_duration_score = -3.4028234e38;
    var best_duration_id = 0xffffffffu;

    if (!idle) {
        for (var c = tid; c < DH; c = c + WG) {
            x_stage[c] = max(Pooler[frame * DH + c] + State[S_DEC + c], 0.0);
        }
        workgroupBarrier();

        for (var tile = 0u; tile < JOINT_TILES; tile = tile + 1u) {
            let row = wgid.x * ROWS_PER_WG + tile * ROW_GROUPS + group;
            var partial = 0.0;
            if (row < N_ROWS) {
                partial = dot_row(0u, W_HEAD + row * ROW4, lane);
            }
            let sum = sum_row(partial, tid, group);
            if (lane == 0u && row < N_ROWS) {
                let score = Biases[B_HEAD + row] + sum;
                if (row < VOCAB) {
                    if (score > best_token_score) {
                        best_token_score = score;
                        best_token_id = row;
                    }
                } else if (score > best_duration_score) {
                    best_duration_score = score;
                    best_duration_id = row;
                }
            }
        }

        // The two rows sharing this 32-lane pair collapse into one candidate;
        // every lane ends up holding that pair's winner.
        exch_score[tid] = best_token_score;
        exch_id[tid] = best_token_id;
        exch_dur_score[tid] = best_duration_score;
        exch_dur_id[tid] = best_duration_id;
        workgroupBarrier();
        let partner = tid ^ 16u;
        let other_score = exch_score[partner];
        let other_id = exch_id[partner];
        if (other_score > best_token_score
            || (other_score == best_token_score && other_id < best_token_id)) {
            best_token_score = other_score;
            best_token_id = other_id;
        }
        let other_duration = exch_dur_score[partner];
        let other_duration_id = exch_dur_id[partner];
        if (other_duration > best_duration_score
            || (other_duration == best_duration_score && other_duration_id < best_duration_id)) {
            best_duration_score = other_duration;
            best_duration_id = other_duration_id;
        }
    }

    if (lane == 0u) {
        red_score[group] = best_token_score;
        red_id[group] = best_token_id;
        red_dur_score[group] = best_duration_score;
        red_dur_id[group] = best_duration_id;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var score = red_score[0];
        var id = red_id[0];
        var dur_score = red_dur_score[0];
        var dur_id = red_dur_id[0];
        for (var g = 1u; g < ROW_GROUPS; g = g + 1u) {
            if (red_score[g] > score || (red_score[g] == score && red_id[g] < id)) {
                score = red_score[g];
                id = red_id[g];
            }
            if (red_dur_score[g] > dur_score
                || (red_dur_score[g] == dur_score && red_dur_id[g] < dur_id)) {
                dur_score = red_dur_score[g];
                dur_id = red_dur_id[g];
            }
        }
        Partials[wgid.x] = Partial(score, id, dur_score, dur_id);
    }
}

@compute @workgroup_size(WG)
fn layer0_main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let group = tid / LANES;
    let lane = tid % LANES;
    if (sync_flag(select(0u, 1u, Result[R_EXIT] != 0u))) {
        return;
    }

    reduce_joint(tid);
    let token = chosen[0];
    let duration = chosen[1] - VOCAB;
    let frame = Result[R_FRAME];
    let steps = Result[R_STEPS];
    let count = Result[R_COUNT];

    if (sync_flag(select(0u, 1u, token == BLANK))) {
        let new_frame = frame + max(duration, 1u);
        let new_steps = steps + 1u;
        if (wgid.x == 0u && tid == 0u) {
            var exit_code = 0u;
            if (new_frame >= params.t_len) {
                exit_code = 1u;
            } else if (new_steps >= params.max_steps) {
                exit_code = 2u;
            }
            Result[R_FRAME] = new_frame;
            Result[R_STEPS] = new_steps;
            Result[R_EXIT] = exit_code;
            Result[R_ACTIVE] = 0u;
        }
        return;
    }

    let new_frame = frame + duration;
    let new_steps = steps + 1u;
    if (wgid.x == 0u && tid == 0u) {
        // Three words per token: id, emission frame, TDT duration.
        let base = RESULT_TOKEN_BASE + count * 3u;
        Result[base] = token;
        Result[base + 1u] = frame;
        Result[base + 2u] = duration;
        var exit_code = 0u;
        if (new_frame >= params.t_len) {
            exit_code = 1u;
        } else if (new_steps >= params.max_steps) {
            exit_code = 2u;
        }
        Result[R_COUNT] = count + 1u;
        Result[R_FRAME] = new_frame;
        Result[R_STEPS] = new_steps;
        Result[R_EXIT] = exit_code;
        Result[R_ACTIVE] = 1u;
    }

    let parity = Result[R_PARITY];
    let h0_old = select(S_H0B, S_H0A, parity == 0u);
    let h0_new = select(S_H0A, S_H0B, parity == 0u);

    // Stage both layer-0 operands: the token embedding and the previous h.
    for (var c = tid; c < DH; c = c + WG) {
        x_stage[c] = Embed[token * DH + c];
        x_stage[DH + c] = State[h0_old + c];
    }
    workgroupBarrier();

    run_layer(
        0u,
        h0_new,
        S_C0,
        wgid.x * CHANNELS_PER_GROUP,
        tid,
        group,
        lane,
    );
}

@compute @workgroup_size(WG)
fn layer1_main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    if (sync_flag(select(0u, 1u, Result[R_ACTIVE] == 0u || Result[R_EXIT] != 0u))) {
        return;
    }
    let tid = lid.x;
    let group = tid / LANES;
    let lane = tid % LANES;
    let parity = Result[R_PARITY];
    let h0_new = select(S_H0A, S_H0B, parity == 0u);
    let h1_old = select(S_H1B, S_H1A, parity == 0u);
    let h1_new = select(S_H1A, S_H1B, parity == 0u);

    for (var c = tid; c < DH; c = c + WG) {
        x_stage[c] = State[h0_new + c];
        x_stage[DH + c] = State[h1_old + c];
    }
    workgroupBarrier();

    run_layer(
        1u,
        h1_new,
        S_C1,
        wgid.x * CHANNELS_PER_GROUP,
        tid,
        group,
        lane,
    );
}

@compute @workgroup_size(WG)
fn project_main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    if (sync_flag(select(0u, 1u, Result[R_ACTIVE] == 0u || Result[R_EXIT] != 0u))) {
        return;
    }
    let tid = lid.x;
    let group = tid / LANES;
    let lane = tid % LANES;
    let parity = Result[R_PARITY];
    let h1_new = select(S_H1A, S_H1B, parity == 0u);

    for (var c = tid; c < DH; c = c + WG) {
        x_stage[c] = State[h1_new + c];
    }
    workgroupBarrier();

    for (var tile = 0u; tile < PROJ_TILES; tile = tile + 1u) {
        let channel = wgid.x * CHANNELS_PER_GROUP + tile * ROW_GROUPS + group;
        let valid = channel < DH;
        let safe_channel = min(channel, DH - 1u);
        let partial = dot_row(0u, W_PROJ + safe_channel * ROW4, lane);
        let sum = sum_row(partial, tid, group);
        if (lane == 0u && valid) {
            State[S_DEC + channel] = Biases[B_PROJ + channel] + sum;
        }
    }

    if (wgid.x == 0u && tid == 0u) {
        Result[R_PARITY] = 1u - Result[R_PARITY];
        Result[R_ACTIVE] = 0u;
    }
}

@compute @workgroup_size(WG)
fn init_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x == 0u) {
        Result[R_COUNT] = 0u;
        Result[R_FRAME] = 0u;
        Result[R_STEPS] = 0u;
        Result[R_EXIT] = 0u;
        Result[R_PARITY] = 0u;
        Result[R_ACTIVE] = 0u;
    }
}
