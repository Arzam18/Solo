#include "nnue.h"
#include <cstring>
#include "nnue_data.h"
#include <iostream>
#include <array>
#include "board.h"
#include <algorithm>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define SOLO_NNUE_NEON 1
#else
#define SOLO_NNUE_NEON 0
#endif

constexpr int INPUT_SIZE = 768;
constexpr int HIDDEN_SIZE = 512;

// Solo's trained network is embedded in nnue_data.h. The header contains the
// parameter payload followed by a 62-byte metadata sentinel; load_nnue copies
// only the model parameters below.
constexpr size_t NNUE_PARAMETER_BYTES =
    sizeof(int16_t) * (INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * 2 + 1);
static_assert(sizeof(nnue_data) >= NNUE_PARAMETER_BYTES,
              "Embedded Solo NNUE data is shorter than its declared network dimensions");




// NNUE Weights and Biases
alignas(64) int16_t hiddenWeight[INPUT_SIZE * HIDDEN_SIZE]; 
alignas(64) int16_t hiddenBias[HIDDEN_SIZE];

alignas(64) int16_t outputWeight[HIDDEN_SIZE * 2]; 
int16_t outputBias;



// Optimization additions. 
// Instead of running a lot of loops while updating the accumulator, 
// we run a single loop that updates both accumulators at once and do add/sub in one loop.

void applyDirtyState(Accumulator* __restrict__ acc, const DirtyState& dirty) {
    if (dirty.type == 0) {
        applyQuietBoth(acc, dirty.wAdd[0], dirty.wSub[0], dirty.bAdd[0], dirty.bSub[0]);
    } else if (dirty.type == 1) {
        applyCaptureBoth(acc, dirty.wAdd[0], dirty.wSub[0], dirty.wSub[1], dirty.bAdd[0], dirty.bSub[0], dirty.bSub[1]);
    } else if (dirty.type == 2) {
        applyCastlingBoth(acc, dirty.wAdd[0], dirty.wAdd[1], dirty.wSub[0], dirty.wSub[1], dirty.bAdd[0], dirty.bAdd[1], dirty.bSub[0], dirty.bSub[1]);
    }
}

// Quiet move or non-capture promotion 
// 1 add + 1 sub per perspective
void applyQuietBoth(Accumulator* __restrict__ acc,
    int wAdd, int wSub,
    int bAdd, int bSub)
{
    const int addOffW = wAdd * HIDDEN_SIZE;
    const int subOffW = wSub * HIDDEN_SIZE;
    const int addOffB = bAdd * HIDDEN_SIZE;
    const int subOffB = bSub * HIDDEN_SIZE;
#if SOLO_NNUE_NEON
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        const int16x8_t deltaW = vsubq_s16(vld1q_s16(hiddenWeight + addOffW + i),
                                           vld1q_s16(hiddenWeight + subOffW + i));
        const int16x8_t deltaB = vsubq_s16(vld1q_s16(hiddenWeight + addOffB + i),
                                           vld1q_s16(hiddenWeight + subOffB + i));
        vst1q_s16(acc[0].data() + i, vaddq_s16(vld1q_s16(acc[0].data() + i), deltaW));
        vst1q_s16(acc[1].data() + i, vaddq_s16(vld1q_s16(acc[1].data() + i), deltaB));
    }
#else
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        acc[0][i] += hiddenWeight[addOffW + i] - hiddenWeight[subOffW + i];
        acc[1][i] += hiddenWeight[addOffB + i] - hiddenWeight[subOffB + i];
    }
#endif
}

// Capture
// 1 add + 2 subs per perspective (moved piece arrives, moved piece leaves + captured piece removed)
void applyCaptureBoth(Accumulator* __restrict__ acc,
    int wAdd, int wSub1, int wSub2,
    int bAdd, int bSub1, int bSub2)
{
    const int addOffW  = wAdd  * HIDDEN_SIZE;
    const int sub1OffW = wSub1 * HIDDEN_SIZE;
    const int sub2OffW = wSub2 * HIDDEN_SIZE;
    const int addOffB  = bAdd  * HIDDEN_SIZE;
    const int sub1OffB = bSub1 * HIDDEN_SIZE;
    const int sub2OffB = bSub2 * HIDDEN_SIZE;
#if SOLO_NNUE_NEON
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        const int16x8_t deltaW = vsubq_s16(
            vsubq_s16(vld1q_s16(hiddenWeight + addOffW + i), vld1q_s16(hiddenWeight + sub1OffW + i)),
            vld1q_s16(hiddenWeight + sub2OffW + i));
        const int16x8_t deltaB = vsubq_s16(
            vsubq_s16(vld1q_s16(hiddenWeight + addOffB + i), vld1q_s16(hiddenWeight + sub1OffB + i)),
            vld1q_s16(hiddenWeight + sub2OffB + i));
        vst1q_s16(acc[0].data() + i, vaddq_s16(vld1q_s16(acc[0].data() + i), deltaW));
        vst1q_s16(acc[1].data() + i, vaddq_s16(vld1q_s16(acc[1].data() + i), deltaB));
    }
#else
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        acc[0][i] += hiddenWeight[addOffW + i] - hiddenWeight[sub1OffW + i] - hiddenWeight[sub2OffW + i];
        acc[1][i] += hiddenWeight[addOffB + i] - hiddenWeight[sub1OffB + i] - hiddenWeight[sub2OffB + i];
    }
#endif
}

// Castling
// 2 adds + 2 subs per perspective (king + rook each move)
void applyCastlingBoth(Accumulator* __restrict__ acc,
    int wAdd1, int wAdd2, int wSub1, int wSub2,
    int bAdd1, int bAdd2, int bSub1, int bSub2)
{
    const int add1OffW = wAdd1 * HIDDEN_SIZE;
    const int add2OffW = wAdd2 * HIDDEN_SIZE;
    const int sub1OffW = wSub1 * HIDDEN_SIZE;
    const int sub2OffW = wSub2 * HIDDEN_SIZE;
    const int add1OffB = bAdd1 * HIDDEN_SIZE;
    const int add2OffB = bAdd2 * HIDDEN_SIZE;
    const int sub1OffB = bSub1 * HIDDEN_SIZE;
    const int sub2OffB = bSub2 * HIDDEN_SIZE;
#if SOLO_NNUE_NEON
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        const int16x8_t addW = vaddq_s16(vld1q_s16(hiddenWeight + add1OffW + i),
                                         vld1q_s16(hiddenWeight + add2OffW + i));
        const int16x8_t subW = vaddq_s16(vld1q_s16(hiddenWeight + sub1OffW + i),
                                         vld1q_s16(hiddenWeight + sub2OffW + i));
        const int16x8_t addB = vaddq_s16(vld1q_s16(hiddenWeight + add1OffB + i),
                                         vld1q_s16(hiddenWeight + add2OffB + i));
        const int16x8_t subB = vaddq_s16(vld1q_s16(hiddenWeight + sub1OffB + i),
                                         vld1q_s16(hiddenWeight + sub2OffB + i));
        const int16x8_t deltaW = vsubq_s16(addW, subW);
        const int16x8_t deltaB = vsubq_s16(addB, subB);
        vst1q_s16(acc[0].data() + i, vaddq_s16(vld1q_s16(acc[0].data() + i), deltaW));
        vst1q_s16(acc[1].data() + i, vaddq_s16(vld1q_s16(acc[1].data() + i), deltaB));
    }
#else
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        acc[0][i] += hiddenWeight[add1OffW + i] + hiddenWeight[add2OffW + i]
                   - hiddenWeight[sub1OffW + i] - hiddenWeight[sub2OffW + i];
        acc[1][i] += hiddenWeight[add1OffB + i] + hiddenWeight[add2OffB + i]
                   - hiddenWeight[sub1OffB + i] - hiddenWeight[sub2OffB + i];
    }
#endif
}

// Legacy single accumulator update (kept for RefreshAccumulator)
void updateAccumulator(Accumulator& acc, int featureIdx, bool isAdd) { 
    int offset = featureIdx * HIDDEN_SIZE;
    
#if SOLO_NNUE_NEON
    const int16x8_t sign = vdupq_n_s16(isAdd ? 1 : -1);
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        const int16x8_t values = vld1q_s16(acc.data() + i);
        const int16x8_t weights = vld1q_s16(hiddenWeight + offset + i);
        vst1q_s16(acc.data() + i, vaddq_s16(values, vmulq_s16(weights, sign)));
    }
#else
    if (isAdd) {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            acc[i] += hiddenWeight[offset + i];
        }
    } else {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            acc[i] -= hiddenWeight[offset + i];
        }
    }
#endif
}

int makeFeatureIndex(int piece_type, int piece_color, int square, int perspective) {
    // (if perspective is 0 it remains the same, if perspective is 1 it flips to the opposite color and square)
    int is_enemy = piece_color ^ perspective; 
    
    int pieceIdx = (piece_type - 1) + (is_enemy * 6);
    
    // if the perspective is zero (white), the square index remains the same. If the perspective is one (black), we flip the square index to mirror it vertically.
    int sq = square ^ (perspective * 56);

    return (pieceIdx * 64) + sq;
}

void load_nnue() {
    size_t offset = 0;

    std::memcpy(hiddenWeight, nnue_data + offset, sizeof(hiddenWeight));
    offset += sizeof(hiddenWeight);

    std::memcpy(hiddenBias, nnue_data + offset, sizeof(hiddenBias));
    offset += sizeof(hiddenBias);

    std::memcpy(outputWeight, nnue_data + offset, sizeof(outputWeight));
    offset += sizeof(outputWeight);

    std::memcpy(&outputBias, nnue_data + offset, sizeof(outputBias));
}

void RefreshAccumulator(const Board& board, Accumulator* acc_white, Accumulator* acc_black) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        (*acc_white)[i] = hiddenBias[i];
        (*acc_black)[i] = hiddenBias[i];
    }

    for (int sq = 0; sq < 64; sq++) {
        int piece = board.mailbox[sq];
        
        if (piece != 0) {
            int p_color = (piece - 1) / 6; 
            int p_type  = ((piece - 1) % 6) + 1; 

            int white_idx = makeFeatureIndex(p_type, p_color, sq, WHITE); // WHITE = 0
            updateAccumulator(*acc_white, white_idx, true);

            int black_idx = makeFeatureIndex(p_type, p_color, sq, BLACK); // BLACK = 1
            updateAccumulator(*acc_black, black_idx, true);
        }
    }
}

int evaluate_nnue(const Accumulator& acc_white, const Accumulator& acc_black, int side_to_move) {
    const Accumulator& us = (side_to_move == WHITE) ? acc_white : acc_black;
    const Accumulator& them = (side_to_move == WHITE) ? acc_black : acc_white;

    int32_t raw_sum = 0;

#if SOLO_NNUE_NEON
    const int16x8_t zero = vdupq_n_s16(0);
    const int16x8_t maxActivation = vdupq_n_s16(255);
    int32x4_t sum32 = vdupq_n_s32(0);
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        const int16x8_t usRaw = vld1q_s16(us.data() + i);
        const int16x8_t themRaw = vld1q_s16(them.data() + i);
        const int16x8_t vUs = vmaxq_s16(zero, vminq_s16(usRaw, maxActivation));
        const int16x8_t vThem = vmaxq_s16(zero, vminq_s16(themRaw, maxActivation));

        // Keep the original scalar model's int16 intermediate multiplication,
        // including its two's-complement wrap, before widening the final product.
        const int16x8_t vwUs = vmulq_s16(vUs, vld1q_s16(outputWeight + i));
        const int16x8_t vwThem = vmulq_s16(vThem, vld1q_s16(outputWeight + HIDDEN_SIZE + i));

        sum32 = vaddq_s32(sum32, vmull_s16(vget_low_s16(vUs), vget_low_s16(vwUs)));
        sum32 = vaddq_s32(sum32, vmull_s16(vget_high_s16(vUs), vget_high_s16(vwUs)));
        sum32 = vaddq_s32(sum32, vmull_s16(vget_low_s16(vThem), vget_low_s16(vwThem)));
        sum32 = vaddq_s32(sum32, vmull_s16(vget_high_s16(vThem), vget_high_s16(vwThem)));
    }
    raw_sum = vaddvq_s32(sum32);
#else
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        int16_t v_us = std::clamp(us[i], (int16_t)0, (int16_t)255);
        int16_t vw_us = v_us * outputWeight[i];
        raw_sum += v_us * vw_us;

        int16_t v_them = std::clamp(them[i], (int16_t)0, (int16_t)255);
        int16_t vw_them = v_them * outputWeight[HIDDEN_SIZE + i];
        raw_sum += v_them * vw_them;
    }
#endif

    // Turning it into a real chess score (QA=255, QB=64, Scale=400)
    int finalScore = ((raw_sum / 255) + outputBias) * 400 / 16320; // 255 * 64 = 16320
    
    return finalScore;
}