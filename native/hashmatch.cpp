// Dependency-free Hamming top-k kernel for 256-bit perceptual hashes.
//
// The Linux artifact is built as a freestanding shared object: no libstdc++,
// libc, OpenSSL or compiler runtime is required in the evaluator container.

typedef unsigned long long u64;
typedef signed int i32;
typedef unsigned short u16;
typedef signed long long i64;

static inline unsigned int popcount64(u64 value) {
    // Portable SWAR popcount.  It avoids libgcc's __popcountdi2 and therefore
    // keeps the cross-compiled ELF completely dependency-free.
    value -= (value >> 1) & 0x5555555555555555ULL;
    value = (value & 0x3333333333333333ULL)
        + ((value >> 2) & 0x3333333333333333ULL);
    value = (value + (value >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (unsigned int)((value * 0x0101010101010101ULL) >> 56);
}

extern "C" i32 hm_topk256(
    const u64* train_hashes,
    i64 train_count,
    const u64* query_hashes,
    i64 query_count,
    i32 max_distance,
    i32 top_k,
    i32* output_indices,
    u16* output_distances
) {
    if (!train_hashes || !query_hashes || !output_indices || !output_distances
        || train_count < 0 || query_count < 0 || top_k <= 0
        || max_distance < 0 || max_distance > 256) {
        return -1;
    }

    for (i64 query_index = 0; query_index < query_count; ++query_index) {
        i32* indices = output_indices + query_index * top_k;
        u16* distances = output_distances + query_index * top_k;
        for (i32 slot = 0; slot < top_k; ++slot) {
            indices[slot] = -1;
            distances[slot] = 0xFFFFU;
        }

        const u64* query = query_hashes + query_index * 4;
        for (i64 train_index = 0; train_index < train_count; ++train_index) {
            const u64* candidate = train_hashes + train_index * 4;
            unsigned int distance = 0;
            distance += popcount64(query[0] ^ candidate[0]);
            distance += popcount64(query[1] ^ candidate[1]);
            if (distance > (unsigned int)max_distance) {
                continue;
            }
            distance += popcount64(query[2] ^ candidate[2]);
            distance += popcount64(query[3] ^ candidate[3]);
            if (distance > (unsigned int)max_distance) {
                continue;
            }

            const u16 short_distance = (u16)distance;
            i32 insert_at = top_k;
            for (i32 slot = 0; slot < top_k; ++slot) {
                if (short_distance < distances[slot]
                    || (short_distance == distances[slot]
                        && train_index < indices[slot])) {
                    insert_at = slot;
                    break;
                }
            }
            if (insert_at == top_k) {
                continue;
            }
            for (i32 slot = top_k - 1; slot > insert_at; --slot) {
                distances[slot] = distances[slot - 1];
                indices[slot] = indices[slot - 1];
            }
            distances[insert_at] = short_distance;
            indices[insert_at] = (i32)train_index;
        }
    }
    return 0;
}
