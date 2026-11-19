// r97/replay.cpp -- the residual store, the boundary recorder and the layer replay.
//
// The store is host memory and needs no radiance; the recorder and the replay need the block
// brackets and the post-issue observer that patches/training/0101 and 0102 add, so a serving build
// compiles them away the way src/tape.cpp does.
#include "r97/replay.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace r97 {

ResidualStore::ResidualStore(int64_t tokens, int64_t hidden)
    : tokens_(tokens), hidden_(hidden) {
    if (tokens <= 0 || hidden <= 0)
        throw std::invalid_argument("residual store needs positive tokens and hidden");
}

int64_t ResidualPlan::extras() const {
    int64_t n = 0;
    for (char c : extra) n += c ? 1 : 0;
    return n;
}

std::string ResidualPlan::refusal(int64_t budget_bytes) const {
    return "the residual store holds " + std::to_string(boundaries) + " boundaries of " +
           std::to_string(tokens) + " tokens x " + std::to_string(hidden) +
           " hidden x 2 bytes" +
           (extras() ? " plus " + std::to_string(extras()) + " second copies of a boundary "
                     : "") +
           ", " + std::to_string(store_bytes()) +
           " bytes, and the budget is " + std::to_string(budget_bytes) + " bytes";
}

namespace {

// The plan's own contradiction, the empty string when it holds together: positive dimensions
// and, when it states extras, the extra map parallel to the boundaries with boundary 0 clear.
const char* plan_problem(const ResidualPlan& p) {
    if (p.boundaries <= 0 || p.tokens <= 0 || p.hidden <= 0)
        return "residual plan needs positive boundaries, tokens and hidden";
    if (!p.extra.empty()) {
        if ((int64_t)p.extra.size() != p.boundaries)
            return "the plan's extra map must name every boundary";
        if (p.extra.front())
            return "boundary 0 has no block before it to copy an extra buffer from";
    }
    return nullptr;
}

}  // namespace

ResidualStore::ResidualStore(const ResidualPlan& plan, RadStream stream)
    : tokens_(plan.tokens), hidden_(plan.hidden), stream_(stream) {
    if (const char* why = plan_problem(plan)) throw std::invalid_argument(why);
    extra_ = plan.extra;
    extra_slot_.assign((size_t)plan.boundaries, -1);
    if (!plan.extra.empty()) {
        int64_t idx = plan.boundaries;
        for (int64_t b = 0; b < plan.boundaries; ++b)
            if (plan.extra[(size_t)b]) extra_slot_[(size_t)b] = idx++;
    }
    extra_sources_.assign((size_t)plan.boundaries, nullptr);
    if (stream) {
        // The device walk's form: one pinned host allocation with the card store's slot
        // arithmetic, so a keep or a restore is a blit packet on the lane. Pageable memory would
        // work, but the engine bounces it through an 8 MiB buffer and drains the stream per chunk
        // (core/device/aql.cpp pageable_copy) -- a host stall at every boundary.
        void* p = rad_dev_alloc(plan.store_bytes(), RAD_MEM_HOST_PINNED);
        if (!p)
            throw std::runtime_error("the host refused the residual store's pinned " +
                                     std::to_string(plan.store_bytes()) +
                                     " bytes: " + rad_dev_last_error());
        host_pinned_ =
            std::shared_ptr<void>(p, [](void* q) { rad_dev_free(q, RAD_MEM_HOST_PINNED); });
        allocated_ = plan.store_bytes();
    }
}

ResidualStore::ResidualStore(RadStream stream, const ResidualPlan& plan, int64_t budget_bytes)
    : tokens_(plan.tokens), hidden_(plan.hidden), stream_(stream) {
    if (!stream) throw std::invalid_argument("vram residual store needs the step's stream");
    if (const char* why = plan_problem(plan)) throw std::invalid_argument(why);
    if (!plan.fits(budget_bytes)) throw std::runtime_error(plan.refusal(budget_bytes));
    extra_ = plan.extra;
    extra_slot_.assign((size_t)plan.boundaries, -1);
    if (!plan.extra.empty()) {
        int64_t idx = plan.boundaries;
        for (int64_t b = 0; b < plan.boundaries; ++b)
            if (plan.extra[(size_t)b]) extra_slot_[(size_t)b] = idx++;
    }
    extra_sources_.assign((size_t)plan.boundaries, nullptr);
    void* p = rad_dev_alloc(plan.store_bytes(), RAD_MEM_DEVICE);
    if (!p)
        throw std::runtime_error(std::string("vram residual store cannot allocate the card: ") +
                                   rad_dev_last_error());
    device_.reset(p, [](void* q) { rad_dev_free(q, RAD_MEM_DEVICE); });
    kept_ = 0;
    extras_kept_ = 0;
    allocated_ = plan.store_bytes();
}

void ResidualStore::keep(int64_t boundary, const void* data) {
    if (!data) throw std::invalid_argument("residual store needs bytes to keep");
    if (boundary < 0 || boundary != boundaries())
        throw std::out_of_range("residual store keeps boundaries in order from zero");
    if (on_device()) {
        // A stream-ordered copy on the step's lane: behind the op the observer just saw.
        if (rad_memcpy_async(slot(boundary), data, boundary_bytes(), stream_) < 0)
            throw std::runtime_error(std::string("the card refused the boundary copy: ") +
                                       rad_dev_last_error());
        ++kept_;
    } else if (on_pinned_host()) {
        // The device walk: the boundary is the card's residual, so this is a stream-ordered copy
        // into the pinned block, behind the op the observer just saw -- the ordering the card
        // store's copy relies on too.
        if (rad_memcpy_async(slot(boundary), data, boundary_bytes(), stream_) < 0)
            throw std::runtime_error(std::string("the host store refused the boundary copy: ") +
                                     rad_dev_last_error());
        ++kept_;
    } else {
        values_.emplace_back((size_t)(tokens_ * hidden_));
        std::memcpy(values_.back().data(), data, (size_t)boundary_bytes());
    }
    // The restore goes back into the buffer the forward wrote, host or card: both paths keep it.
    sources_.push_back(data);
}

void ResidualStore::set_name(int64_t boundary, const std::string& buffer) {
    if (boundary < 0) throw std::out_of_range("residual store names boundaries from zero");
    if (names_.size() <= static_cast<size_t>(boundary))
        names_.resize(static_cast<size_t>(boundary) + 1);
    names_[static_cast<size_t>(boundary)] = buffer;
}

std::string ResidualStore::name(int64_t boundary) const {
    if (boundary < 0 || static_cast<size_t>(boundary) >= names_.size()) return {};
    return names_[static_cast<size_t>(boundary)];
}

void ResidualStore::keep_extra(int64_t boundary, const void* data) {
    if (!data) throw std::invalid_argument("the extra copy needs bytes to keep");
    if (!extra_at(boundary))
        throw std::logic_error("the plan holds no extra copy at that boundary");
    if (!has(boundary))
        throw std::out_of_range("the extra copy follows the boundary's residual copy");
    if (extra_sources_[(size_t)boundary])
        throw std::out_of_range("the extra copy was already kept at that boundary");
    if (on_device()) {
        if (rad_memcpy_async(extra_slot(boundary), data, boundary_bytes(), stream_) < 0)
            throw std::runtime_error(std::string("the card refused the extra copy: ") +
                                       rad_dev_last_error());
    } else if (on_pinned_host()) {
        if (rad_memcpy_async(extra_slot(boundary), data, boundary_bytes(), stream_) < 0)
            throw std::runtime_error(std::string("the host store refused the extra copy: ") +
                                     rad_dev_last_error());
    } else {
        extra_values_.resize((size_t)boundary + 1);
        extra_values_[(size_t)boundary].resize((size_t)(tokens_ * hidden_));
        std::memcpy(extra_values_[(size_t)boundary].data(), data, (size_t)boundary_bytes());
    }
    extra_sources_[(size_t)boundary] = data;
    ++extras_kept_;
}

bool ResidualStore::has(int64_t boundary) const {
    return boundary >= 0 && boundary < boundaries();
}

const uint16_t* ResidualStore::at(int64_t boundary) const {
    if (on_device()) throw std::logic_error("a card-backed store has no host view; use read()");
    if (!has(boundary)) return nullptr;
    return on_pinned_host() ? static_cast<const uint16_t*>(slot(boundary))
                            : values_[(size_t)boundary].data();
}

const void* ResidualStore::source(int64_t boundary) const {
    return has(boundary) ? sources_[(size_t)boundary] : nullptr;
}

bool ResidualStore::extra_at(int64_t boundary) const {
    return boundary >= 0 && boundary < (int64_t)extra_.size() && extra_[(size_t)boundary];
}

const void* ResidualStore::source_extra(int64_t boundary) const {
    return boundary >= 0 && boundary < (int64_t)extra_sources_.size()
               ? extra_sources_[(size_t)boundary]
               : nullptr;
}

int64_t ResidualStore::boundaries() const {
    return on_device() || on_pinned_host() ? kept_ : (int64_t)values_.size();
}

int64_t ResidualStore::bytes() const {
    if (on_device() || on_pinned_host()) return allocated_;
    return (boundaries() + extras_kept_) * boundary_bytes();
}

void* ResidualStore::slot(int64_t boundary) const {
    uint8_t* base = on_device() ? static_cast<uint8_t*>(device_.get())
                                : static_cast<uint8_t*>(host_pinned_.get());
    return base + (size_t)boundary * (size_t)boundary_bytes();
}

void* ResidualStore::extra_slot(int64_t boundary) const {
    uint8_t* base = on_device() ? static_cast<uint8_t*>(device_.get())
                                : static_cast<uint8_t*>(host_pinned_.get());
    return base + (size_t)extra_slot_[(size_t)boundary] * (size_t)boundary_bytes();
}

void ResidualStore::restore(int64_t boundary, void* dest) const {
    if (!dest) throw std::invalid_argument("residual store needs bytes to restore into");
    if (!has(boundary)) throw std::out_of_range("residual store has not kept that boundary");
    if (extra_at(boundary) && !source_extra(boundary))
        throw std::runtime_error("the plan holds an extra copy at that boundary and the store "
                                 "never kept it");
    if (on_device()) {
        // Device to device on the step's lane: it orders ahead of the replayed block's reads.
        if (rad_memcpy_async(dest, slot(boundary), boundary_bytes(), stream_) < 0)
            throw std::runtime_error(std::string("the card refused the restore copy: ") +
                                       rad_dev_last_error());
        if (extra_at(boundary))
            if (rad_memcpy_async(const_cast<void*>(source_extra(boundary)),
                                 extra_slot(boundary), boundary_bytes(), stream_) < 0)
                throw std::runtime_error(std::string("the card refused the extra restore copy: ") +
                                           rad_dev_last_error());
    } else if (on_pinned_host()) {
        // The device walk: the block about to re-run reads these buffers on the card, so each
        // restore is a stream-ordered copy out of the pinned block, ahead of its ops on the lane.
        if (rad_memcpy_async(dest, slot(boundary), boundary_bytes(), stream_) < 0)
            throw std::runtime_error(std::string("the host store refused the restore copy: ") +
                                     rad_dev_last_error());
        if (extra_at(boundary))
            if (rad_memcpy_async(const_cast<void*>(source_extra(boundary)), extra_slot(boundary),
                                 boundary_bytes(), stream_) < 0)
                throw std::runtime_error(
                    std::string("the host store refused the extra restore copy: ") +
                    rad_dev_last_error());
    } else {
        std::memcpy(dest, values_[(size_t)boundary].data(), (size_t)boundary_bytes());
        if (extra_at(boundary))
            std::memcpy(const_cast<void*>(source_extra(boundary)),
                        extra_values_[(size_t)boundary].data(), (size_t)boundary_bytes());
    }
}

void ResidualStore::read(int64_t boundary, void* host) const {
    if (!host) throw std::invalid_argument("residual store needs bytes to read into");
    if (!has(boundary)) throw std::out_of_range("residual store has not kept that boundary");
    if (on_device()) {
        if (rad_memcpy_async(host, slot(boundary), boundary_bytes(), stream_) < 0 ||
            rad_stream_sync(stream_) < 0)
            throw std::runtime_error(std::string("the card refused the read-back copy: ") +
                                       rad_dev_last_error());
    } else if (on_pinned_host()) {
        // The copies landed on the lane; the reader wants the bytes here and now.
        if (rad_stream_sync(stream_) < 0)
            throw std::runtime_error(std::string("the host store could not synchronize its "
                                                 "stream: ") + rad_dev_last_error());
        std::memcpy(host, slot(boundary), (size_t)boundary_bytes());
    } else {
        std::memcpy(host, values_[(size_t)boundary].data(), (size_t)boundary_bytes());
    }
}

void ResidualStore::read_extra(int64_t boundary, void* host) const {
    if (!host) throw std::invalid_argument("residual store needs bytes to read into");
    if (!extra_at(boundary)) throw std::out_of_range("the plan holds no extra copy at that boundary");
    if (!extra_sources_[(size_t)boundary])
        throw std::out_of_range("the extra copy was not kept at that boundary");
    if (on_device()) {
        if (rad_memcpy_async(host, extra_slot(boundary), boundary_bytes(), stream_) < 0 ||
            rad_stream_sync(stream_) < 0)
            throw std::runtime_error(std::string("the card refused the read-back copy: ") +
                                       rad_dev_last_error());
    } else if (on_pinned_host()) {
        if (rad_stream_sync(stream_) < 0)
            throw std::runtime_error(std::string("the host store could not synchronize its "
                                                 "stream: ") + rad_dev_last_error());
        std::memcpy(host, extra_slot(boundary), (size_t)boundary_bytes());
    } else {
        std::memcpy(host, extra_values_[(size_t)boundary].data(), (size_t)boundary_bytes());
    }
}

}  // namespace r97

#if defined(RAD_ISSUE_OBSERVER) && defined(RAD_REPLAY_BLOCK)

#include <utility>

#include "engine.h"

namespace r97 {
namespace {

// The residual is one operand: bf16, two dimensions, `tokens * hidden * 2` bytes, written, and it
// names a declared buffer. The first one seen names that buffer (the embeddings' output); later
// operands must carry that name, so a norm, a tap of the same byte count, or a KV write (whose
// tape identity is synthetic) is not mistaken for the residual.
bool residual_bytes(const Operand& o, const OperandContent& c, const ResidualStore& s,
                    std::string& name) {
    if (!o.written || !c.named || c.bytes != (size_t)s.boundary_bytes()) return false;
    if (!name.empty()) return name == o.buffer;
    if (o.buffer.empty()) return false;
    name = o.buffer;
    return true;
}

}  // namespace

struct BoundaryRecorder::Impl {
    Tape* tape = nullptr;
    ResidualStore store;
    std::string name;
    std::string read_name;
    std::string last_name;
    const void* last = nullptr;
    const void* last_extra = nullptr;
    const void* named_extra = nullptr;
    const void* read_ptr = nullptr;
    bool extra_seen = false;
    bool named_seen = false;
    bool ext_read = false;
    bool attached = false;

    Impl(Tape& t, int64_t tokens, int64_t hidden) : tape(&t), store(tokens, hidden) { ready(); }

    Impl(Tape& t, ResidualStore s) : tape(&t), store(std::move(s)) { ready(); }

    void ready() {
        if (!tape->attached())
            throw std::invalid_argument("boundary recorder needs a tape attached to a started engine");
        attach();
    }

    void attach() {
        if (attached) return;
        tape->set_issue_hook([this](const Tape&, const OpRecord& op,
                                    const std::vector<OperandContent>& content) {
            issued(op, content);
        });
        tape->set_bracket_hook([this](const Tape&, int64_t b, bool begin) { block(b, begin); });
        attached = true;
    }

    void detach() {
        if (!attached) return;
        // The recorder must let go before a replay replaces these hooks.
        tape->set_issue_hook(nullptr);
        tape->set_bracket_hook(nullptr);
        attached = false;
    }

    void issued(const OpRecord& op, const std::vector<OperandContent>& content) {
        for (size_t i = 0; i < op.operands.size() && i < content.size(); ++i) {
            const Operand& o = op.operands[i];
            const OperandContent& c = content[i];
            const bool named_bb = c.named && !o.buffer.empty() && o.buffer != name &&
                                  c.bytes == (size_t)store.boundary_bytes();
            if (residual_bytes(o, c, store, name)) {
                last = c.data;
                last_name = o.buffer;
            } else if (o.written && named_bb) {
                // The block's latest boundary-sized write that is not the residual, and the
                // latest one under the name the block reads as its second input: the value the
                // next block takes there, at the boundary this block ends at.
                extra_seen = true;
                last_extra = c.data;
                if (!read_name.empty() && o.buffer == read_name) {
                    named_extra = c.data;
                    named_seen = true;
                }
            } else if (!o.written && named_bb && !extra_seen) {
                // The block's second input: a boundary-sized buffer read before this block wrote
                // one. Its name is the one the block after this one reads too.
                ext_read = true;
                read_name = o.buffer;
                read_ptr = c.data;
            }
        }
    }

    void block(int64_t b, bool begin) {
        if (begin) {
            extra_seen = false;
            named_seen = false;
            ext_read = false;
            // Boundary 0 is what ran before the first block: the embeddings.
            if (b == 0 && last) {
                store.keep(0, last);
                store.set_name(0, last_name);
            }
            return;
        }
        // The block's last residual write is boundary b + 1.
        if (last) {
            store.keep(b + 1, last);
            store.set_name(b + 1, last_name);
        }
        if (store.extra_at(b + 1)) {
            // The extra copy at boundary b + 1 is the buffer the block after this one reads as
            // its second input. This block reads that same buffer (the arch's blocks share the
            // shape), so the last write here under that name is the boundary's value. The first
            // boundary has no such read behind it: there the block's last boundary-sized write
            // is the only candidate, and the block that reads it checks the copy below.
            const void* src = named_seen ? named_extra : (extra_seen ? last_extra : nullptr);
            if (!src)
                throw std::runtime_error(
                    "the plan holds an extra copy at boundary " + std::to_string(b + 1) +
                    " and the block before it wrote no boundary-sized buffer");
            store.keep_extra(b + 1, src);
        }
        // The block's own second input, when it read one: the plan must have kept it at the
        // boundary the block starts at, or a replay of the block reads a stale buffer.
        if (ext_read && !store.extra_at(b))
            throw std::runtime_error("block " + std::to_string(b) +
                                     " reads a second boundary-sized input and the plan keeps "
                                     "no copy at its boundary");
        // ...and the copy kept there must be the very buffer this block reads: a copy taken from
        // another boundary-sized buffer would restore the wrong bytes, in the wrong place.
        if (ext_read && store.extra_at(b) && read_ptr != store.source_extra(b))
            throw std::runtime_error("the extra copy at boundary " + std::to_string(b) +
                                     " is not the buffer the block reads");
    }
};

BoundaryRecorder::BoundaryRecorder(Tape& tape, int64_t tokens, int64_t hidden)
    : impl_(new Impl(tape, tokens, hidden)) {}

BoundaryRecorder::BoundaryRecorder(Tape& tape, ResidualStore store)
    : impl_(new Impl(tape, std::move(store))) {}

BoundaryRecorder::~BoundaryRecorder() = default;

void BoundaryRecorder::detach() { impl_->detach(); }

const ResidualStore& BoundaryRecorder::store() const { return impl_->store; }

struct LayerReplay::Impl {
    rad::Engine* engine = nullptr;
    Tape* tape = nullptr;
    ResidualStore store;
    std::vector<OpRecord> ops;
    std::vector<Tape::Block> blocks;
    const ReplaySnapshot* after = nullptr;
    int64_t target = -1;
    int64_t op = 0;
    bool inside = false;

    Impl(rad::Engine& e, Tape& t, ResidualStore s)
        : engine(&e), tape(&t), store(std::move(s)) {
        if (!e.issue_ctx()) throw std::runtime_error("layer replay needs a started engine");
        ops = t.ops();
        blocks = t.blocks();
        // The forward's tape has recorded what this replay needs. It lets go of the context so the
        // same tape can attach again for the replayed block, with this replay's hooks.
        t.detach();
    }

    const Tape::Block* find(int64_t layer) const {
        for (const Tape::Block& b : blocks)
            if (b.index == layer) return &b;
        return nullptr;
    }

    void issued(const OpRecord&, const std::vector<OperandContent>& content) {
        if (!inside) return;
        const Tape::Block* b = find(target);
        if (b) {
            const int64_t at = b->begin + op;
            if (at < b->end && at < (int64_t)ops.size() && after) (*after)(ops[(size_t)at], content);
        }
        ++op;
    }

    void block(int64_t b, bool begin) {
        if (b != target) return;
        inside = begin;
        if (begin) op = 0;
    }

    void watch(const ReplaySnapshot* fn) {
        after = fn;
        target = -1;
        op = 0;
        inside = false;
        RadCtx* ctx = engine->issue_ctx();
        if (!ctx) throw std::runtime_error("layer replay needs a started engine");
        tape->set_issue_hook([this](const Tape&, const OpRecord& o,
                                    const std::vector<OperandContent>& content) {
            issued(o, content);
        });
        tape->set_bracket_hook([this](const Tape&, int64_t b, bool begin) { block(b, begin); });
        tape->attach(ctx);
    }

    void stop() {
        tape->detach();
        tape->set_issue_hook(nullptr);
        tape->set_bracket_hook(nullptr);
        after = nullptr;
        inside = false;
    }
};

LayerReplay::LayerReplay(rad::Engine& engine, Tape& tape, ResidualStore store)
    : impl_(new Impl(engine, tape, std::move(store))) {}

LayerReplay::~LayerReplay() { impl_->stop(); }

int64_t LayerReplay::layers() const { return (int64_t)impl_->blocks.size(); }

const ResidualStore& LayerReplay::store() const { return impl_->store; }

const void* LayerReplay::residual(int64_t boundary) const {
    return impl_->store.source(boundary);
}

void LayerReplay::replay(int64_t layer, const ReplaySnapshot& after) {
    Impl& s = *impl_;
    const Tape::Block* b = s.find(layer);
    if (!b) throw std::out_of_range("no recorded block for that layer");

    const void* dst = s.store.source(layer);
    if (!s.store.has(layer) || !dst)
        throw std::runtime_error("the store lacks this layer's input boundary");
    // The residual buffer is the one the forward wrote and the one this block writes again.
    s.store.restore(layer, const_cast<void*>(dst));
    s.watch(&after);
    s.target = layer;
    const int rc = s.engine->replay_block(layer);
    s.stop();
    if (rc != RAD_OK) throw std::runtime_error(std::string("layer replay failed: ") +
                                               rad_strerror(rc));
}

}  // namespace r97

#endif  // RAD_ISSUE_OBSERVER && RAD_REPLAY_BLOCK
