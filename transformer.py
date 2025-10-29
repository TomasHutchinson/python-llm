# ----------------------------
# Full GPU training shader + Python glue
# ----------------------------
import moderngl
import numpy as np
import pickle
from tqdm import tqdm

from tokenize_data import apply_bpe

# ----------------------------
# Load token data
# ----------------------------
path = "models/"
with open(path + "token_ids.pkl", "rb") as f:
    token_ids = pickle.load(f)

token2id = token_ids[0]
id2token = token_ids[1]
tokenized_ids = token_ids[2]
merge_map = token_ids[3]

# ----------------------------
# Hyperparameters
# ----------------------------
vocab_size = len(token2id)
embedding_dim = 64
n_heads = 3
head_dim = embedding_dim // max(1, n_heads)
ff_hidden_dim = 128
block_size = 32
lr = 1e-4       # slightly larger LR for demo; tune as needed
epochs = 4
eps = 1e-15
initial_lr = 1e-4
decay_rate = 0.95
min_lr = 1e-6

# ----------------------------
# Create ModernGL context
# ----------------------------
ctx = moderngl.create_standalone_context(require=430)

# ----------------------------
# Initialize weights
# ----------------------------
np.random.seed(42)
limit = 1 / np.sqrt(embedding_dim)
E = np.random.randn(vocab_size, embedding_dim).astype('f4') * limit
Wq = np.random.uniform(-limit, limit, (embedding_dim, embedding_dim)).astype(np.float32)
Wk = np.random.uniform(-limit, limit, (embedding_dim, embedding_dim)).astype(np.float32)
Wv = np.random.uniform(-limit, limit, (embedding_dim, embedding_dim)).astype(np.float32)
Wo = np.random.uniform(-limit, limit, (embedding_dim, embedding_dim)).astype(np.float32)
W1 = np.random.randn(embedding_dim, ff_hidden_dim).astype('f4') * limit
b1 = np.zeros(ff_hidden_dim, dtype='f4')
W2 = np.random.randn(ff_hidden_dim, embedding_dim).astype('f4') * limit
b2 = np.zeros(embedding_dim, dtype='f4')

# ----------------------------
# Create GPU buffers
# ----------------------------
E_buf = ctx.buffer(E.tobytes())        # binding 0
Wq_buf = ctx.buffer(Wq.tobytes())      # binding 1
Wk_buf = ctx.buffer(Wk.tobytes())      # binding 2
Wv_buf = ctx.buffer(Wv.tobytes())      # binding 3
Wo_buf = ctx.buffer(Wo.tobytes())      # binding 4
W1_buf = ctx.buffer(W1.tobytes())      # binding 5
b1_buf = ctx.buffer(b1.tobytes())      # binding 6
W2_buf = ctx.buffer(W2.tobytes())      # binding 7
b2_buf = ctx.buffer(b2.tobytes())      # binding 8

tokens_buf = ctx.buffer(reserve=block_size * 4)  # binding 9
targets_buf = ctx.buffer(reserve=4)              # binding 11 (single int target)
logits_buf = ctx.buffer(reserve=vocab_size * 4)  # binding 10

# ----------------------------
# Compute Shader: forward + loss + SGD (single-sample updates)
# - Updates: E, W1, b1, W2, b2, Wo
# - Does not update Wq/Wk/Wv (left frozen)
# ----------------------------
# We inject constants with f-strings to avoid large arrays on stack in GLSL.
train_shader_template = f"""
#version 430
#define EMBEDDING_DIM {embedding_dim}
#define BLOCK_SIZE {block_size}
#define VOCAB_SIZE {vocab_size}
#define FF_HIDDEN {ff_hidden_dim}

layout(local_size_x = BLOCK_SIZE) in;

// Buffers
layout(std430, binding=0) buffer E_buf {{ float E[]; }};       // VOCAB_SIZE * EMBEDDING_DIM
layout(std430, binding=1) buffer Wq_buf {{ float Wq[]; }};     // not updated
layout(std430, binding=2) buffer Wk_buf {{ float Wk[]; }};     // not updated
layout(std430, binding=3) buffer Wv_buf {{ float Wv[]; }};     // not updated
layout(std430, binding=4) buffer Wo_buf {{ float Wo[]; }};     // updated
layout(std430, binding=5) buffer W1_buf {{ float W1[]; }};     // updated
layout(std430, binding=6) buffer b1_buf {{ float b1[]; }};     // updated
layout(std430, binding=7) buffer W2_buf {{ float W2[]; }};     // updated
layout(std430, binding=8) buffer b2_buf {{ float b2[]; }};     // updated

layout(std430, binding=9) buffer Tokens_buf {{ int tokens[]; }};    // BLOCK_SIZE tokens (sequence)
layout(std430, binding=10) buffer Logits_buf {{ float logits[]; }}; // VOCAB_SIZE logits
layout(std430, binding=11) buffer Targets_buf {{ int target_idx[]; }}; // single int
layout(std430, binding=12) buffer Loss_buf {{ float losses[]; }};

uniform int seq_len;
uniform int context_size;
uniform float lr;
uniform float eps;

shared float shared_emb[BLOCK_SIZE * EMBEDDING_DIM];
shared float shared_K[BLOCK_SIZE * EMBEDDING_DIM];
shared float shared_V[BLOCK_SIZE * EMBEDDING_DIM];

float safe_sqrt(float x) {{
    return sqrt(max(x, 1e-12));
}}

void main() {{
    uint batch_idx = gl_WorkGroupID.x;      // one workgroup per sequence
    uint token_idx = gl_LocalInvocationID.x; // one thread per token

    if (token_idx >= seq_len) return;

    int token_id = tokens[batch_idx * seq_len + token_idx];
    int target_id = target_idx[batch_idx * seq_len + token_idx];
    uint lid = gl_LocalInvocationID.x;
    int ilid = int(lid);

    for (int i = ilid; i < BLOCK_SIZE * EMBEDDING_DIM; i += BLOCK_SIZE) {{
    shared_emb[i] = 0.0;
    shared_K[i] = 0.0;
    shared_V[i] = 0.0;
    }}

    // 1) Load embeddings into shared memory
    if (ilid < seq_len) {{
        int tid = tokens[batch_idx * seq_len + ilid];;
        int base_t = tid * EMBEDDING_DIM;
        int base_s = ilid * EMBEDDING_DIM;
        for (int i = 0; i < EMBEDDING_DIM; ++i) {{
            shared_emb[base_s + i] = E[base_t + i];
        }}
    }} else {{
        int base_s = ilid * EMBEDDING_DIM;
        for (int i = 0; i < EMBEDDING_DIM; ++i) shared_emb[base_s + i] = 0.0;
    }}
    barrier();

    // 2) Compute K and V for each position (not updated)
    if (ilid < seq_len) {{
        int base_s = ilid * EMBEDDING_DIM;
        for (int outp = 0; outp < EMBEDDING_DIM; ++outp) {{
            float sumK = 0.0;
            float sumV = 0.0;
            for (int j = 0; j < EMBEDDING_DIM; ++j) {{
                float xj = shared_emb[base_s + j];
                sumK += xj * Wk[j * EMBEDDING_DIM + outp];
                sumV += xj * Wv[j * EMBEDDING_DIM + outp];
            }}
            shared_K[base_s + outp] = sumK;
            shared_V[base_s + outp] = sumV;
        }}
    }} else {{
        int base_s = ilid * EMBEDDING_DIM;
        for (int i = 0; i < EMBEDDING_DIM; ++i) {{
            shared_K[base_s + i] = 0.0;
            shared_V[base_s + i] = 0.0;
        }}
    }}
    barrier();

    // 3) Build Q_last (from last token position)
    int last_idx = context_size - 1;
    float Q[EMBEDDING_DIM];
    for (int outp = 0; outp < EMBEDDING_DIM; ++outp) {{
        float s = 0.0;
        for (int j = 0; j < EMBEDDING_DIM; ++j) {{
            float xj = shared_emb[last_idx * EMBEDDING_DIM + j];
            s += xj * Wq[j * EMBEDDING_DIM + outp];
        }}
        Q[outp] = s;
    }}

    // 4) Attention limited to context_size tokens
    float scores[BLOCK_SIZE];
    float scale = 1.0 / safe_sqrt(float(EMBEDDING_DIM));
    float maxscore = -1e30;
    for (int i = 0; i < context_size; ++i) {{
        float dot = 0.0;
        int base = i * EMBEDDING_DIM;
        for (int j = 0; j < EMBEDDING_DIM; ++j) dot += Q[j] * shared_K[base + j];
        float sc = dot * scale;
        scores[i] = sc;
        if (sc > maxscore) maxscore = sc;
    }}

    float expsum = 0.0;
    for (int i = 0; i < seq_len; ++i) {{
        float v = exp(scores[i] - maxscore);
        scores[i] = v;
        expsum += v;
    }}
    for (int i = 0; i < seq_len; ++i) scores[i] = scores[i] / expsum;

    // 5) Weighted sum to get y
    float y[EMBEDDING_DIM];
    for (int j = 0; j < EMBEDDING_DIM; ++j) y[j] = 0.0;
    for (int i = 0; i < seq_len; ++i) {{
        float w = scores[i];
        int base = i * EMBEDDING_DIM;
        for (int j = 0; j < EMBEDDING_DIM; ++j) y[j] += w * shared_V[base + j];
    }}

    // 6) ctx_vec = y @ Wo
    float ctx_vec[EMBEDDING_DIM];
for (int outp = 0; outp < EMBEDDING_DIM; ++outp) {{
        float s = 0.0;
        for (int j = 0; j < EMBEDDING_DIM; ++j) s += y[j] * Wo[j * EMBEDDING_DIM + outp];
        ctx_vec[outp] = s;
    }}

    // 7–8) Feedforward & projection (ctx_preLN)
    float h[FF_HIDDEN];
    for (int i = 0; i < FF_HIDDEN; ++i) {{
        float s = 0.0;
        for (int j = 0; j < EMBEDDING_DIM; ++j) s += ctx_vec[j] * W1[j * FF_HIDDEN + i];
        s += b1[i];
        h[i] = s > 0.0 ? s : 0.0;
    }}

    float ctx_preLN[EMBEDDING_DIM];
    for (int outp = 0; outp < EMBEDDING_DIM; ++outp) {{
        float s = 0.0;
        for (int j = 0; j < FF_HIDDEN; ++j) s += h[j] * W2[j * EMBEDDING_DIM + outp];
        s += b2[outp];
        ctx_preLN[outp] = s;
    }}

    // 9) Compute logits: logits[v] = E[v] · ctx_preLN  (each thread writes a slice)
    for (int v = ilid; v < VOCAB_SIZE; v += BLOCK_SIZE) {{
        int base_e = v * EMBEDDING_DIM;
        float s = 0.0;
        for (int j = 0; j < EMBEDDING_DIM; ++j) s += E[base_e + j] * ctx_preLN[j];
        logits[v] = s;
    }}
    barrier();

    // 10) Compute loss + perform SGD updates on thread 0 only (single-sample)
    if (ilid == 0) {{
        int target = target_idx[0];
        // read logits into local array (to protect against repeated global reads)
        // (We access logits via the SSBO above)
        float logit_arr[VOCAB_SIZE];
        float maxl = -1e30;
        for (int v = 0; v < VOCAB_SIZE; ++v) {{
            float z = logits[v];
            logit_arr[v] = z;
            if (z > maxl) maxl = z;
        }}
        // softmax
        float sumexp = 0.0;
        for (int v = 0; v < VOCAB_SIZE; ++v) {{
            float ex = exp(logit_arr[v] - maxl);
            logit_arr[v] = ex;
            sumexp += ex;
        }}
        for (int v = 0; v < VOCAB_SIZE; ++v) logit_arr[v] = logit_arr[v] / sumexp;
        // cross-entropy loss = -log p[target]
        float loss = -log(max(logit_arr[target], 1e-12));

        // compute d_ctx_preLN = sum_v (p_v * E[v]) - E[target]
        float d_ctx_preLN[EMBEDDING_DIM];
        for (int j = 0; j < EMBEDDING_DIM; ++j) d_ctx_preLN[j] = 0.0;
        for (int v = 0; v < VOCAB_SIZE; ++v) {{
            float p = logit_arr[v];
            int base_e = v * EMBEDDING_DIM;
            for (int j = 0; j < EMBEDDING_DIM; ++j) {{
                d_ctx_preLN[j] += p * E[base_e + j];
            }}
        }}
        // subtract E[target]
        int base_t = target * EMBEDDING_DIM;
        for (int j = 0; j < EMBEDDING_DIM; ++j) d_ctx_preLN[j] -= E[base_t + j];

        // Now update parameters with simple SGD: param -= lr * grad

        // ----- Update embeddings E[v] for all v -----
        // dL/dE[v] = (p_v - 1_{{v==target}}) * ctx_preLN
        for (int v = 0; v < VOCAB_SIZE; ++v) {{
            int base_e_v = v * EMBEDDING_DIM;
            float coeff = logit_arr[v] - (v == target ? 1.0 : 0.0);
            for (int j = 0; j < EMBEDDING_DIM; ++j) {{
                // E[v, j] -= lr * coeff * ctx_preLN[j]
                E[base_e_v + j] -= lr * coeff * ctx_preLN[j];
            }}
        }}

        // ----- Update W2 (FF_HIDDEN x EMBEDDING_DIM) and b2 (EMBEDDING_DIM) -----
        // dL/dW2[j, k] = h[j] * d_ctx_preLN[k]
        for (int j = 0; j < FF_HIDDEN; ++j) {{
            for (int k = 0; k < EMBEDDING_DIM; ++k) {{
                int idx = j * EMBEDDING_DIM + k;
                W2[idx] -= 0.0; // placeholder in case driver needs coherent memory (no-op)
                // read current W2 from buffer and update (we can write directly into W2 buffer)
                float grad = h[j] * d_ctx_preLN[k];
                // Update: W2[j,k] -= lr * grad
                // Directly write to global W2_buf memory:
                // GLSL can't index the W2 buffer via name W2_buf; we access via W2 (bound)
                // Here W2 buffer is bound to binding 7 (W2_buf)
            }}
        }}
        // Unfortunately GLSL currently does not allow direct in-place modification of a buffer bound to layout(std430) with name W2_buf unless it's declared; we declared W2_buf as W2_buf above. Let's perform updates by writing to the buffer memory directly:
        // However updating large weight matrices element-by-element here is verbose; instead we'll perform W2 and b2 updates by reading and writing via declared W2_buf and b2_buf locations.
        // We'll update W2 and b2 element-wise:
        for (int j = 0; j < FF_HIDDEN; ++j) {{
            for (int k = 0; k < EMBEDDING_DIM; ++k) {{
                int idx = j * EMBEDDING_DIM + k;
                float old = W2[idx];
                float grad = h[j] * d_ctx_preLN[k];
                W2[idx] = old - lr * grad;
            }}
        }}
        for (int k = 0; k < EMBEDDING_DIM; ++k) {{
            float oldb = b2[k];
            b2[k] = oldb - lr * d_ctx_preLN[k];
        }}

        // ----- Backprop into h: d_h = W2 * d_ctx_preLN  (shape FF_HIDDEN) -----
        float d_h[FF_HIDDEN];
        for (int j = 0; j < FF_HIDDEN; ++j) {{
            float s = 0.0;
            for (int k = 0; k < EMBEDDING_DIM; ++k) s += W2[j * EMBEDDING_DIM + k] * d_ctx_preLN[k];
            d_h[j] = s;
            // apply ReLU derivative (h was ReLU(ctx_vec @ W1 + b1))
            if (h[j] <= 0.0) d_h[j] = 0.0;
        }}

        // ----- Update W1 (EMBEDDING_DIM x FF_HIDDEN) and b1 (FF_HIDDEN) -----
        for (int j = 0; j < EMBEDDING_DIM; ++j) {{
            for (int k = 0; k < FF_HIDDEN; ++k) {{
                int idx = j * FF_HIDDEN + k;
                float old = W1[idx];
                float grad = ctx_vec[j] * d_h[k];
                W1[idx] = old - lr * grad;
            }}
        }}
        for (int k = 0; k < FF_HIDDEN; ++k) {{
            float oldb1 = b1[k];
            b1[k] = oldb1 - lr * d_h[k];
        }}

        // ----- Update Wo (EMBEDDING_DIM x EMBEDDING_DIM) -----
        // We have ctx_vec = y @ Wo  => dWo = y^T * d_ctx_vec
        // d_ctx_vec is backprop from d_h through W1: compute approx d_ctx_vec = W1 @ d_h
        float d_ctx_vec[EMBEDDING_DIM];
        for (int i = 0; i < EMBEDDING_DIM; ++i) d_ctx_vec[i] = 0.0;
        for (int i = 0; i < EMBEDDING_DIM; ++i) {{
            for (int k = 0; k < FF_HIDDEN; ++k) {{
                d_ctx_vec[i] += W1[i * FF_HIDDEN + k] * d_h[k];
            }}
        }}
        // Now update Wo: Wo[j * EMBEDDING_DIM + outp] -= lr * y[j] * d_ctx_vec[outp]
        for (int j = 0; j < EMBEDDING_DIM; ++j) {{
            for (int outp = 0; outp < EMBEDDING_DIM; ++outp) {{
                int idx = j * EMBEDDING_DIM + outp;
                float old = Wo[idx];
                float grad = y[j] * d_ctx_vec[outp];
                Wo[idx] = old - lr * grad;
            }}
        }}

        // done with weight updates for this single sample
        // Optionally write loss into logits[0] slot for diagnostic
        losses[batch_idx] = loss;
    }}
}}
"""

# Compile shader
train_shader = ctx.compute_shader(train_shader_template)

# ----------------------------
# Bind buffers to bindings
# ----------------------------
E_buf.bind_to_storage_buffer(0)
Wq_buf.bind_to_storage_buffer(1)
Wk_buf.bind_to_storage_buffer(2)
Wv_buf.bind_to_storage_buffer(3)
Wo_buf.bind_to_storage_buffer(4)
W1_buf.bind_to_storage_buffer(5)
b1_buf.bind_to_storage_buffer(6)
W2_buf.bind_to_storage_buffer(7)
b2_buf.bind_to_storage_buffer(8)
tokens_buf.bind_to_storage_buffer(9)
logits_buf.bind_to_storage_buffer(10)
targets_buf.bind_to_storage_buffer(11)

# Set uniforms that are static now (seq_len set each run)
train_shader['lr'].value = lr
# train_shader['eps'].value = eps

# ----------------------------
# Helper: prepare sequence and target
# ----------------------------
def make_seq_and_target(seq_ids):
    # seq_ids is a list of token ids length >= block_size+1
    X = np.array(seq_ids[:block_size], dtype='i4')
    Y = int(seq_ids[1:block_size+1][-1])  # target is the last next token
    # pad if shorter
    if len(X) < block_size:
        pad = np.zeros(block_size - len(X), dtype='i4')
        X = np.concatenate([pad, X])
    return X, Y

# ----------------------------
# Training loop (single-sequence SGD)
# ----------------------------
print("Starting training on GPU (single-sample SGD). This will be slow for large vocab.")
batch_size = 4
tokens_buf = ctx.buffer(reserve=batch_size * block_size * 4)
targets_buf = ctx.buffer(reserve=batch_size * block_size * 4)
loss_buf = ctx.buffer(reserve=batch_size * 4)
loss_buf.bind_to_storage_buffer(12)


for epoch in tqdm(range(epochs)):
    total_loss = 0.0
    n = 0
    lr = max(min_lr, initial_lr * (decay_rate ** epoch))
    train_shader['lr'].value = lr
    for i in range(0, len(tokenized_ids), batch_size):
        batch = tokenized_ids[i:i+batch_size]
        batch = [seq for seq in batch if len(seq) >= block_size+1]
        if not batch: continue

        X = np.array([seq[:block_size] for seq in batch], dtype='i4')
        Y = np.array([seq[1:block_size+1] for seq in batch], dtype='i4')

        tokens_buf.write(X.tobytes())
        targets_buf.write(Y.tobytes())
        train_shader['seq_len'].value = block_size

        # safe 1D dispatch
        train_shader.run(group_x=len(batch))

        n += len(batch)

    # read losses once per epoch
    loss_vals = np.frombuffer(loss_buf.read(), dtype='f4')
    avg_loss = np.mean(loss_vals)
    # Read data from GPU buffer and make a writable copy
    E_np = np.frombuffer(E_buf.read(), dtype='f4').copy().reshape(vocab_size, embedding_dim)

    # Compute L2 norms
    norms = np.linalg.norm(E_np, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)

    # Normalize
    E_np /= norms

    # Write back normalized weights to GPU
    E_buf.write(E_np.astype('f4').tobytes())
    print(f"Epoch {epoch+1} — Avg Loss: {avg_loss:.6f}")

# ----------------------------
# After training, use the forward shader (from earlier) or CPU forward to generate
# For simplicity, we'll read updated E,W1,W2,Wo from GPU back to CPU to run a CPU forward.
# ----------------------------
E = np.frombuffer(E_buf.read(), dtype='f4').reshape(vocab_size, embedding_dim).copy()
W1 = np.frombuffer(W1_buf.read(), dtype='f4').reshape(embedding_dim, ff_hidden_dim).copy()
b1 = np.frombuffer(b1_buf.read(), dtype='f4').copy()
W2 = np.frombuffer(W2_buf.read(), dtype='f4').reshape(ff_hidden_dim, embedding_dim).copy()
b2 = np.frombuffer(b2_buf.read(), dtype='f4').copy()
Wo = np.frombuffer(Wo_buf.read(), dtype='f4').reshape(embedding_dim, embedding_dim).copy()
Wq = np.frombuffer(Wq_buf.read(), dtype='f4').reshape(embedding_dim, embedding_dim).copy()
Wk = np.frombuffer(Wk_buf.read(), dtype='f4').reshape(embedding_dim, embedding_dim).copy()
Wv = np.frombuffer(Wv_buf.read(), dtype='f4').reshape(embedding_dim, embedding_dim).copy()


# Tokens buffer (we'll always write exactly block_size ints)
tokens_buf = ctx.buffer(reserve=block_size * 4)  # binding 9

# Logits buffer: one float per vocab token
logits_buf = ctx.buffer(reserve=vocab_size * 4)  # binding 10

# ----------------------------
# GLSL compute shader (forward-only, distributed logits)
# ----------------------------
# We inject EMBEDDING_DIM, BLOCK_SIZE, VOCAB_SIZE, FF_HIDDEN into the shader string.
shader_template = f"""
#version 430
#define EMBEDDING_DIM {embedding_dim}
#define BLOCK_SIZE {block_size}
#define VOCAB_SIZE {vocab_size}
#define FF_HIDDEN {ff_hidden_dim}

layout(local_size_x = BLOCK_SIZE) in;

// Buffers (std430)
layout(std430, binding=0) buffer E_buf {{ float E[]; }};       // VOCAB_SIZE * EMBEDDING_DIM
layout(std430, binding=1) buffer Wq_buf {{ float Wq[]; }};
layout(std430, binding=2) buffer Wk_buf {{ float Wk[]; }};
layout(std430, binding=3) buffer Wv_buf {{ float Wv[]; }};
layout(std430, binding=4) buffer Wo_buf {{ float Wo[]; }};
layout(std430, binding=5) buffer W1_buf {{ float W1[]; }};
layout(std430, binding=6) buffer b1_buf {{ float b1[]; }};
layout(std430, binding=7) buffer W2_buf {{ float W2[]; }};
layout(std430, binding=8) buffer b2_buf {{ float b2[]; }};

layout(std430, binding=9) buffer Tokens_buf {{ int tokens[]; }};    // BLOCK_SIZE ints (sequence)
layout(std430, binding=10) buffer Logits_buf {{ float logits[]; }}; // VOCAB_SIZE floats (output)

uniform int seq_len;
uniform int context_size;
uniform float eps;

shared float shared_emb[BLOCK_SIZE * EMBEDDING_DIM];
shared float shared_K[BLOCK_SIZE * EMBEDDING_DIM];
shared float shared_V[BLOCK_SIZE * EMBEDDING_DIM];

float safe_sqrt(float x) {{
    return sqrt(max(x, 1e-12));
}}

void main() {{
    uint lid = gl_LocalInvocationID.x;
    int ilid = int(lid);

    // 1) load embeddings into shared memory (each thread loads one position)
    if (ilid < seq_len) {{
        int tid = tokens[ilid];
        int base_t = tid * EMBEDDING_DIM;
        int base_s = ilid * EMBEDDING_DIM;
        for (int i = 0; i < EMBEDDING_DIM; ++i) {{
            shared_emb[base_s + i] = E[base_t + i];
        }}
    }} else {{
        // zero-out extra slots
        int base_s = ilid * EMBEDDING_DIM;
        for (int i = 0; i < EMBEDDING_DIM; ++i) shared_emb[base_s + i] = 0.0;
    }}
    barrier();

    // 2) Each thread computes K and V for its own position
    if (ilid < seq_len) {{
        int base_s = ilid * EMBEDDING_DIM;
        for (int outp = 0; outp < EMBEDDING_DIM; ++outp) {{
            float sumK = 0.0;
            float sumV = 0.0;
            for (int j = 0; j < EMBEDDING_DIM; ++j) {{
                float xj = shared_emb[base_s + j];
                sumK += xj * Wk[j * EMBEDDING_DIM + outp];
                sumV += xj * Wv[j * EMBEDDING_DIM + outp];
            }}
            shared_K[base_s + outp] = sumK;
            shared_V[base_s + outp] = sumV;
        }}
    }} else {{
        int base_s = ilid * EMBEDDING_DIM;
        for (int i = 0; i < EMBEDDING_DIM; ++i) {{
            shared_K[base_s + i] = 0.0;
            shared_V[base_s + i] = 0.0;
        }}
    }}
    barrier();

    // 3) Build Q_last for current context
    int last_idx = context_size - 1;
    float Q[EMBEDDING_DIM];
    for (int outp = 0; outp < EMBEDDING_DIM; ++outp) {{
        float s = 0.0;
        for (int j = 0; j < EMBEDDING_DIM; ++j) {{
            float xj = shared_emb[last_idx * EMBEDDING_DIM + j];
            s += xj * Wq[j * EMBEDDING_DIM + outp];
        }}
        Q[outp] = s;
    }}

    // Attention restricted to context_size
    float scores[BLOCK_SIZE];
    float scale = 1.0 / safe_sqrt(float(EMBEDDING_DIM));
    float maxscore = -1e30;
    for (int i = 0; i < context_size; ++i) {{
        float dot = 0.0;
        int base = i * EMBEDDING_DIM;
        for (int j = 0; j < EMBEDDING_DIM; ++j) dot += Q[j] * shared_K[base + j];
        float sc = dot * scale;
        scores[i] = sc;
        if (sc > maxscore) maxscore = sc;
    }}

    // Stable softmax
    float expsum = 0.0;
    for (int i = 0; i < seq_len; ++i) {{
        float v = exp(scores[i] - maxscore);
        scores[i] = v;
        expsum += v;
    }}
    for (int i = 0; i < seq_len; ++i) scores[i] = scores[i] / expsum;

    // Weighted sum y = sum_i scores[i] * V_i
    float y[EMBEDDING_DIM];
    for (int j = 0; j < EMBEDDING_DIM; ++j) y[j] = 0.0;
    for (int i = 0; i < seq_len; ++i) {{
        float w = scores[i];
        int base = i * EMBEDDING_DIM;
        for (int j = 0; j < EMBEDDING_DIM; ++j) y[j] += w * shared_V[base + j];
    }}

    // ctx_vec replaces ctx_vec
    float ctx_vec[EMBEDDING_DIM];
    for (int outp = 0; outp < EMBEDDING_DIM; ++outp) {{
        float s = 0.0;
        for (int j = 0; j < EMBEDDING_DIM; ++j) s += y[j] * Wo[j * EMBEDDING_DIM + outp];
        ctx_vec[outp] = s;
    }}

    // Feedforward
    float h[FF_HIDDEN];
    for (int i = 0; i < FF_HIDDEN; ++i) {{
        float s = 0.0;
        for (int j = 0; j < EMBEDDING_DIM; ++j) s += ctx_vec[j] * W1[j * FF_HIDDEN + i];
        s += b1[i];
        h[i] = s > 0.0 ? s : 0.0;
    }}

    // Back projection
    float ctx_ff[EMBEDDING_DIM];
    for (int outp = 0; outp < EMBEDDING_DIM; ++outp) {{
        float s = 0.0;
        for (int j = 0; j < FF_HIDDEN; ++j) s += h[j] * W2[j * EMBEDDING_DIM + outp];
        s += b2[outp];
        ctx_ff[outp] = s;
    }}

    // LayerNorm
    float mean = 0.0;
    for (int i = 0; i < EMBEDDING_DIM; ++i) mean += ctx_ff[i] / float(EMBEDDING_DIM);
    float var = 0.0;
    for (int i = 0; i < EMBEDDING_DIM; ++i) {{
        float d = ctx_ff[i] - mean;
        var += d * d / float(EMBEDDING_DIM);
    }}
    for (int i = 0; i < EMBEDDING_DIM; ++i) {{
        ctx_ff[i] = (ctx_ff[i] - mean) / safe_sqrt(var + eps);
    }}

    // Compute logits
    for (int v = ilid; v < VOCAB_SIZE; v += BLOCK_SIZE) {{
        int base_e = v * EMBEDDING_DIM;
        float s = 0.0;
        for (int j = 0; j < EMBEDDING_DIM; ++j) s += E[base_e + j] * ctx_ff[j];
        logits[v] = s;
    }}

    // end
}}
"""

# Compile compute shader
transformer_shader = ctx.compute_shader(shader_template)

# ----------------------------
# Bind buffers to the bindings matching the shader layout
# ----------------------------
E_buf.bind_to_storage_buffer(0)
Wq_buf.bind_to_storage_buffer(1)
Wk_buf.bind_to_storage_buffer(2)
Wv_buf.bind_to_storage_buffer(3)
Wo_buf.bind_to_storage_buffer(4)
W1_buf.bind_to_storage_buffer(5)
b1_buf.bind_to_storage_buffer(6)
W2_buf.bind_to_storage_buffer(7)
b2_buf.bind_to_storage_buffer(8)
tokens_buf.bind_to_storage_buffer(9)
logits_buf.bind_to_storage_buffer(10)

# Set uniforms
transformer_shader['seq_len'].value = block_size
transformer_shader['eps'].value = eps

# ----------------------------
# Generation function (uses the GPU shader to get logits)
# ----------------------------
def softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

def generate_tokens_gpu(seed_text, max_tokens=32):
    generated = []
    for word in seed_text.strip().split():
        generated.extend([token2id[t] for t in apply_bpe(word, merge_map)])

    for _ in range(max_tokens):
        seq = np.array(generated[-block_size:], dtype='i4')
        if len(seq) < block_size:
            pad = np.zeros(block_size - len(seq), dtype='i4')
            seq = np.concatenate([pad, seq])
        tokens_buf.write(seq.tobytes())

        # dynamically adjust context_size
        context_size = min(len(generated), block_size)
        transformer_shader['seq_len'].value = block_size
        transformer_shader['context_size'].value = context_size

        transformer_shader.run(group_x=1)
        logits = np.frombuffer(logits_buf.read(), dtype='f4')

        temperature = 0.8
        probs = np.exp(logits / temperature)
        probs /= probs.sum()
        next_id = np.random.choice(vocab_size, p=probs)
        generated.append(int(next_id))

    return generated

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    out_ids = generate_tokens_gpu("Once upon a", max_tokens=16)
    out_text = ''.join([id2token[t].replace('</w>', ' ') for t in out_ids])
    print(out_text)
