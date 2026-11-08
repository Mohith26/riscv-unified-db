// Copyright (c) Mohith Gajjela
// SPDX-License-Identifier: BSD-3-Clause-Clear

// Differential test of the double-precision helpers in fp.idl (used by the D
// extension) against berkeley-softfloat-3.
//
// Every operation is run in all five rounding modes on a mix of special
// values, boundary values, and random operands, and must match softfloat
// bit-for-bit, including the accrued exception flags. The softfloat library
// linked here is built with the x86 specialization, so NaN results and the
// integer results of invalid conversions are mapped to their RISC-V values
// (canonical NaN, and the saturated integers from the RISC-V F/D chapter)
// before comparing.

#include <fmt/format.h>

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <udb/bits.hpp>
#include <udb/cfgs/rv64/hart.hxx>
#include <udb/csr.hpp>
#include <udb/db_data.hxx>
#include <udb/enum.hxx>
#include <vector>

extern "C" {
#include <softfloat.h>
}

using namespace udb;

namespace {

  struct NullSocModel {
    uint64_t read_hpm_counter(uint64_t) { return 0; }
    uint64_t read_mcycle() { return 0; }
    uint64_t read_mtime() { return 0; }
    uint64_t sw_write_mcycle(uint64_t value) { return value; }
    void cache_block_zero(uint64_t) {}
    void eei_ecall_from_m() {}
    void eei_ecall_from_s() {}
    void eei_ecall_from_u() {}
    void eei_ecall_from_vs() {}
    void eei_ebreak() {}
    void memory_model_acquire() {}
    void memory_model_release() {}
    void notify_mode_change(PrivilegeMode::ValueType, PrivilegeMode::ValueType) {}
    void prefetch_instruction(uint64_t) {}
    void prefetch_read(uint64_t) {}
    void prefetch_write(uint64_t) {}
    void fence(uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t) {}
    void fence_tso() {}
    void ifence() {}
    void order_pgtbl_writes_before_vmafence() {}
    void order_pgtbl_reads_after_vmafence() {}
    uint64_t read_physical_memory_8(uint64_t) { return 0; }
    uint64_t read_physical_memory_16(uint64_t) { return 0; }
    uint64_t read_physical_memory_32(uint64_t) { return 0; }
    uint64_t read_physical_memory_64(uint64_t) { return 0; }
    void write_physical_memory_8(uint64_t, uint64_t) {}
    void write_physical_memory_16(uint64_t, uint64_t) {}
    void write_physical_memory_32(uint64_t, uint64_t) {}
    void write_physical_memory_64(uint64_t, uint64_t) {}
    int memcpy_from_host(uint64_t, const uint8_t*, uint64_t) { return 0; }
    int memcpy_to_host(uint8_t*, uint64_t, uint64_t) { return 0; }
    uint8_t atomic_check_then_write_32(uint64_t, uint32_t, uint32_t) { return 0; }
    uint8_t atomic_check_then_write_64(uint64_t, uint64_t, uint64_t) { return 0; }
    uint8_t atomically_set_pte_a(uint64_t, uint64_t, uint32_t) { return 0; }
    uint8_t atomically_set_pte_a_d(uint64_t, uint64_t, uint32_t) { return 0; }
    uint64_t atomic_read_modify_write_8(uint64_t, uint8_t, AmoOperation::ValueType) { return 0; }
    uint64_t atomic_read_modify_write_16(uint64_t, uint16_t, AmoOperation::ValueType) { return 0; }
    uint64_t atomic_read_modify_write_32(uint64_t, uint32_t, AmoOperation::ValueType) { return 0; }
    uint64_t atomic_read_modify_write_64(uint64_t, uint64_t, AmoOperation::ValueType) { return 0; }
    uint8_t pma_applies_Q_(PmaAttribute::ValueType, uint64_t, uint32_t) { return 1; }
    void delay(uint64_t) {}
    void iss_syscall(uint64_t, uint64_t) {}
    uint32_t read_device_32(uint64_t) { return 0; }
    void write_device_32(uint64_t, uint32_t) {}
    void sync_read_after_write_device(bool, uint32_t) {}
    void sync_write_after_read_device(bool, uint32_t) {}
    uint32_t read_seed() { return 0b10u << 30; }
    void write_seed() {}
  };

  using TestHart = Rv64_Hart<NullSocModel>;

  constexpr uint64_t DP_CANONICAL_NAN = 0x7ff8000000000000ULL;
  constexpr uint32_t SP_CANONICAL_NAN = 0x7fc00000U;

  // Iterations of random operands per operation and rounding mode.
  constexpr int kRandomIters = 4000;

  Config make_test_config() {
    nlohmann::json implemented_exts = nlohmann::json::array(
        {nlohmann::json::array({"I", "2.1.0"}), nlohmann::json::array({"M", "2.0.0"}),
         nlohmann::json::array({"Zicsr", "2.0.0"}), nlohmann::json::array({"Zifencei", "2.0.0"}),
         nlohmann::json::array({"F", "2.2.0"})});
    nlohmann::json param_values = nlohmann::json::object();
    param_values["MXLEN"] = 64;
    return Config(implemented_exts, param_values);
  }

  uint32_t read_fflags(TestHart& hart) {
    CsrBase* csr = hart.csr("fflags");
    REQUIRE(csr != nullptr);
    return static_cast<uint32_t>(csr->sw_read(Bits<8>{hart.mxlen()}).to_defined().get());
  }

  void clear_fflags(TestHart& hart) {
    CsrBase* csr = hart.csr("fflags");
    REQUIRE(csr != nullptr);
    REQUIRE(csr->sw_write(Bits<64>{0}, Bits<8>{hart.mxlen()}));
  }

  struct Mode {
    const char* name;
    RoundingMode hart;
    uint_fast8_t softfloat;
  };

  const std::array<Mode, 5> kModes{{
      {"RNE", RoundingMode{RoundingMode::RNE}, softfloat_round_near_even},
      {"RTZ", RoundingMode{RoundingMode::RTZ}, softfloat_round_minMag},
      {"RDN", RoundingMode{RoundingMode::RDN}, softfloat_round_min},
      {"RUP", RoundingMode{RoundingMode::RUP}, softfloat_round_max},
      {"RMM", RoundingMode{RoundingMode::RMM}, softfloat_round_near_maxMag},
  }};

  bool is_nan64(uint64_t v) {
    return ((v >> 52) & 0x7ff) == 0x7ff && (v & 0x000fffffffffffffULL) != 0;
  }
  bool is_nan32(uint32_t v) { return ((v >> 23) & 0xff) == 0xff && (v & 0x007fffffU) != 0; }

  // Interesting double-precision bit patterns.
  const std::vector<uint64_t> kSpecial64 = {
      0x0000000000000000ULL, 0x8000000000000000ULL,  // +/-0
      0x7ff0000000000000ULL, 0xfff0000000000000ULL,  // +/-inf
      0x7ff8000000000000ULL, 0xfff8000000000000ULL,  // qNaN
      0x7ff0000000000001ULL, 0x7ff4000000000000ULL,  // sNaN
      0x7fffffffffffffffULL, 0xffffffffffffffffULL,  // qNaN with payload
      0x0000000000000001ULL, 0x8000000000000001ULL,  // smallest subnormal
      0x000fffffffffffffULL, 0x800fffffffffffffULL,  // largest subnormal
      0x0010000000000000ULL, 0x8010000000000000ULL,  // smallest normal
      0x7fefffffffffffffULL, 0xffefffffffffffffULL,  // largest finite
      0x3ff0000000000000ULL, 0xbff0000000000000ULL,  // +/-1
      0x3ff0000000000001ULL, 0x3fefffffffffffffULL,  // 1 +/- ulp
      0x4000000000000000ULL, 0x3fe0000000000000ULL,  // 2, 0.5
      0x3ff8000000000000ULL, 0xc008000000000000ULL,  // 1.5, -3
      0x41dfffffffc00000ULL, 0x41e0000000000000ULL,  // 2^31 - 1, 2^31
      0xc1e0000000000000ULL, 0xc1e0000000200000ULL,  // -2^31, -2^31 - 1
      0x41efffffffe00000ULL, 0x41f0000000000000ULL,  // 2^32 - 1, 2^32
      0x43dfffffffffffffULL, 0x43e0000000000000ULL,  // just below 2^63, 2^63
      0xc3e0000000000000ULL, 0x43f0000000000000ULL,  // -2^63, 2^64
      0x3fe0000000000001ULL, 0xbfe0000000000000ULL,  // just above 0.5, -0.5
      0x4004000000000000ULL, 0x400c000000000000ULL,  // 2.5, 3.5 (ties)
      0x0018000000000000ULL, 0x3cb0000000000000ULL,  // 1.5 * min normal, 2^-52
      0x7fe0000000000000ULL, 0x0008000000000000ULL,  // 2^1023, 2^-1023
  };

  class OperandGen {
   public:
    explicit OperandGen(uint64_t seed) : m_rng(seed) {}

    uint64_t bits() { return m_rng(); }

    uint64_t special() { return kSpecial64[m_rng() % kSpecial64.size()]; }

    // A double with the given biased exponent and a random sign/fraction.
    uint64_t with_exp(uint64_t exp) {
      return (m_rng() & 0x800fffffffffffffULL) | ((exp & 0x7ff) << 52);
    }

    // Mostly random operands, with extra weight on specials, subnormals and
    // values near the overflow/underflow boundaries.
    uint64_t operand() {
      switch (m_rng() % 8) {
        case 0:
          return special();
        case 1:
          return with_exp(0);  // subnormal
        case 2:
          return with_exp(1 + (m_rng() % 4));  // tiny normal
        case 3:
          return with_exp(0x7fe - (m_rng() % 4));  // huge normal
        case 4:
          return with_exp(0x3ff + (m_rng() % 128) - 64);
        default:
          return bits();
      }
    }

    // A second operand whose exponent is close to that of +a+, to exercise
    // cancellation and alignment.
    uint64_t near(uint64_t a) {
      uint64_t exp = (a >> 52) & 0x7ff;
      int64_t delta = static_cast<int64_t>(m_rng() % 7) - 3;
      int64_t e = static_cast<int64_t>(exp) + delta;
      if (e < 0) e = 0;
      if (e > 0x7fe) e = 0x7fe;
      uint64_t b = with_exp(static_cast<uint64_t>(e));
      if (m_rng() % 4 == 0) {
        // near-total cancellation: same high fraction bits, opposite sign
        b = (a ^ 0x8000000000000000ULL) ^ (m_rng() & 0xffffULL);
      }
      return b;
    }

    uint64_t u64() { return m_rng(); }

   private:
    std::mt19937_64 m_rng;
  };

  struct Counter {
    uint64_t cases = 0;
  };

  // Report every mismatch as a failed check (instead of stopping at the first
  // one), but only print the first few so a systematic bug stays readable.
  uint64_t g_mismatches = 0;
  void mismatch(const std::string& msg) {
    if (g_mismatches++ < 25) {
      FAIL_CHECK(msg);
    }
  }

  // Run a check with fflags cleared on both sides.
  template <typename HartFn, typename SoftFn>
  void diff_f64_result(TestHart& hart, const Mode& m, HartFn hart_fn, SoftFn soft_fn,
                       Counter& count, const std::string& what) {
    clear_fflags(hart);
    softfloat_roundingMode = m.softfloat;
    softfloat_detectTininess = softfloat_tininess_afterRounding;
    softfloat_exceptionFlags = 0;

    const uint64_t got = hart_fn();
    const uint32_t got_flags = read_fflags(hart);
    uint64_t want = soft_fn();
    const uint32_t want_flags = softfloat_exceptionFlags;

    if (is_nan64(want)) {
      want = DP_CANONICAL_NAN;  // RISC-V never propagates NaN payloads
    }

    ++count.cases;
    if (got != want || got_flags != want_flags) {
      mismatch(fmt::format("{} [{}]: got {:016x} flags {:02x}, want {:016x} flags {:02x}", what,
                           m.name, got, got_flags, want, want_flags));
    }
  }

  // RISC-V results for an invalid floating-point to integer conversion.
  uint64_t riscv_invalid_int(bool is_nan, bool negative, bool is_signed, int width) {
    const bool pos = is_nan || !negative;
    if (width == 32) {
      if (is_signed) return pos ? 0x7fffffffULL : 0x80000000ULL;
      return pos ? 0xffffffffULL : 0ULL;
    }
    if (is_signed) return pos ? 0x7fffffffffffffffULL : 0x8000000000000000ULL;
    return pos ? 0xffffffffffffffffULL : 0ULL;
  }

}  // namespace

TEST_CASE("fp.idl F64 helpers match berkeley-softfloat-3", "[cpp_hart][fp][f64]") {
  NullSocModel soc;
  Config cfg = make_test_config();
  TestHart hart(0, soc, cfg);
  hart.reset(0);

  OperandGen gen(0x5eed0f64ULL);
  Counter count;

  for (const Mode& m : kModes) {
    // Exhaustive pairs of special values for the two-operand operations
    std::vector<std::pair<uint64_t, uint64_t>> pairs;
    for (uint64_t a : kSpecial64) {
      for (uint64_t b : kSpecial64) {
        pairs.emplace_back(a, b);
      }
    }
    for (int i = 0; i < kRandomIters; i++) {
      uint64_t a = gen.operand();
      pairs.emplace_back(a, gen.operand());
      pairs.emplace_back(a, gen.near(a));
      pairs.emplace_back(gen.bits(), gen.bits());
    }

    for (const auto& [a, b] : pairs) {
      const std::string args = fmt::format("({:016x}, {:016x})", a, b);
      diff_f64_result(
          hart, m,
          [&] { return hart.f64_add(Bits<64>{a}, Bits<64>{b}, m.hart).to_defined().get(); },
          [&] { return f64_add(float64_t{a}, float64_t{b}).v; }, count, "f64_add" + args);
      diff_f64_result(
          hart, m,
          [&] { return hart.f64_sub(Bits<64>{a}, Bits<64>{b}, m.hart).to_defined().get(); },
          [&] { return f64_sub(float64_t{a}, float64_t{b}).v; }, count, "f64_sub" + args);
      diff_f64_result(
          hart, m,
          [&] { return hart.f64_mul(Bits<64>{a}, Bits<64>{b}, m.hart).to_defined().get(); },
          [&] { return f64_mul(float64_t{a}, float64_t{b}).v; }, count, "f64_mul" + args);
      diff_f64_result(
          hart, m,
          [&] { return hart.f64_div(Bits<64>{a}, Bits<64>{b}, m.hart).to_defined().get(); },
          [&] { return f64_div(float64_t{a}, float64_t{b}).v; }, count, "f64_div" + args);

      // fmin/fmax are not in softfloat: check against the IEEE 754-2019
      // minimumNumber/maximumNumber rules the RISC-V spec requires.
      for (bool is_max : {false, true}) {
        clear_fflags(hart);
        uint64_t got = hart.f64_min_max(Bits<64>{a}, Bits<64>{b}, is_max).to_defined().get();
        uint32_t got_flags = read_fflags(hart);
        softfloat_exceptionFlags = 0;
        const bool a_nan = is_nan64(a), b_nan = is_nan64(b);
        uint32_t want_flags = 0;
        f64_eq(float64_t{a}, float64_t{b});  // raises invalid only for signaling NaNs
        want_flags = softfloat_exceptionFlags;
        uint64_t want;
        if (a_nan && b_nan) {
          want = DP_CANONICAL_NAN;
        } else if (a_nan) {
          want = b;
        } else if (b_nan) {
          want = a;
        } else {
          softfloat_exceptionFlags = 0;
          const bool a_lt_b =
              f64_lt_quiet(float64_t{a}, float64_t{b}) || (a == 0x8000000000000000ULL && b == 0);
          const bool b_lt_a =
              f64_lt_quiet(float64_t{b}, float64_t{a}) || (b == 0x8000000000000000ULL && a == 0);
          if (is_max) {
            want = a_lt_b ? b : (b_lt_a ? a : a);
          } else {
            want = a_lt_b ? a : (b_lt_a ? b : b);
          }
          // equal values other than signed zeros are bitwise identical
        }
        ++count.cases;
        if (got != want || got_flags != want_flags) {
          mismatch(fmt::format(
              "f64_min_max(is_max={}){}: got {:016x} flags {:02x}, want {:016x} flags {:02x}",
              is_max, args, got, got_flags, want, want_flags));
        }
      }

      if (&m == &kModes[0]) {
        // comparisons do not depend on the rounding mode
        auto check_cmp = [&](const char* name, bool got, uint32_t got_flags, bool want,
                             uint32_t want_flags) {
          ++count.cases;
          if (got != want || got_flags != want_flags) {
            mismatch(fmt::format("{}{}: got {} flags {:02x}, want {} flags {:02x}", name, args, got,
                                 got_flags, want, want_flags));
          }
        };
        clear_fflags(hart);
        softfloat_exceptionFlags = 0;
        bool g = hart.f64_eq(Bits<64>{a}, Bits<64>{b});
        uint32_t gf = read_fflags(hart);
        bool w = f64_eq(float64_t{a}, float64_t{b});
        check_cmp("f64_eq", g, gf, w, softfloat_exceptionFlags);

        clear_fflags(hart);
        softfloat_exceptionFlags = 0;
        g = hart.f64_lt(Bits<64>{a}, Bits<64>{b});
        gf = read_fflags(hart);
        w = f64_lt(float64_t{a}, float64_t{b});
        check_cmp("f64_lt", g, gf, w, softfloat_exceptionFlags);

        clear_fflags(hart);
        softfloat_exceptionFlags = 0;
        g = hart.f64_le(Bits<64>{a}, Bits<64>{b});
        gf = read_fflags(hart);
        w = f64_le(float64_t{a}, float64_t{b});
        check_cmp("f64_le", g, gf, w, softfloat_exceptionFlags);
      }
    }

    // Fused multiply-add: specials crossed with a few values, plus random
    // triples, including addends close to the product to force cancellation.
    std::vector<std::array<uint64_t, 3>> triples;
    for (uint64_t a : kSpecial64) {
      for (uint64_t b : kSpecial64) {
        triples.push_back({a, b, gen.special()});
        triples.push_back({a, b, gen.operand()});
      }
    }
    for (int i = 0; i < kRandomIters; i++) {
      uint64_t a = gen.operand();
      uint64_t b = gen.operand();
      triples.push_back({a, b, gen.operand()});
      triples.push_back({gen.bits(), gen.bits(), gen.bits()});
      // c close to -(a*b) so most significant bits cancel
      softfloat_roundingMode = softfloat_round_near_even;
      uint64_t p = f64_mul(float64_t{a}, float64_t{b}).v;
      triples.push_back({a, b, gen.near(p) ^ 0x8000000000000000ULL});
      triples.push_back({a, b, p ^ 0x8000000000000000ULL});
      // c with a much smaller or larger exponent than the product
      triples.push_back({a, b, gen.with_exp(gen.u64() % 0x7ff)});
    }
    for (const auto& t : triples) {
      const uint64_t a = t[0], b = t[1], c = t[2];
      diff_f64_result(
          hart, m,
          [&] {
            return hart.f64_muladd(Bits<64>{a}, Bits<64>{b}, Bits<64>{c}, m.hart)
                .to_defined()
                .get();
          },
          [&] { return f64_mulAdd(float64_t{a}, float64_t{b}, float64_t{c}).v; }, count,
          fmt::format("f64_muladd({:016x}, {:016x}, {:016x})", a, b, c));
    }

    // Single-operand operations
    std::vector<uint64_t> singles(kSpecial64.begin(), kSpecial64.end());
    for (int i = 0; i < 4 * kRandomIters; i++) {
      singles.push_back(gen.operand());
      singles.push_back(gen.bits());
    }
    for (uint64_t a : singles) {
      const std::string args = fmt::format("({:016x})", a);
      diff_f64_result(
          hart, m, [&] { return hart.f64_sqrt(Bits<64>{a}, m.hart).to_defined().get(); },
          [&] { return f64_sqrt(float64_t{a}).v; }, count, "f64_sqrt" + args);

      // f64 -> f32
      {
        clear_fflags(hart);
        softfloat_roundingMode = m.softfloat;
        softfloat_exceptionFlags = 0;
        uint32_t got =
            static_cast<uint32_t>(hart.f64_to_f32(Bits<64>{a}, m.hart).to_defined().get());
        uint32_t got_flags = read_fflags(hart);
        uint32_t want = f64_to_f32(float64_t{a}).v;
        if (is_nan32(want)) want = SP_CANONICAL_NAN;
        ++count.cases;
        if (got != want || got_flags != softfloat_exceptionFlags) {
          mismatch(
              fmt::format("f64_to_f32{} [{}]: got {:08x} flags {:02x}, want {:08x} flags {:02x}",
                          args, m.name, got, got_flags, want, softfloat_exceptionFlags));
        }
      }

      // f64 -> integer conversions
      auto check_to_int = [&](const char* name, bool is_signed, int width, uint64_t got) {
        uint32_t got_flags = read_fflags(hart);
        uint64_t want;
        softfloat_exceptionFlags = 0;
        if (width == 32 && is_signed) {
          want = static_cast<uint32_t>(f64_to_i32(float64_t{a}, m.softfloat, true));
        } else if (width == 32) {
          want = f64_to_ui32(float64_t{a}, m.softfloat, true);
        } else if (is_signed) {
          want = static_cast<uint64_t>(f64_to_i64(float64_t{a}, m.softfloat, true));
        } else {
          want = f64_to_ui64(float64_t{a}, m.softfloat, true);
        }
        uint32_t want_flags = softfloat_exceptionFlags;
        if (want_flags & softfloat_flag_invalid) {
          want = riscv_invalid_int(is_nan64(a), (a >> 63) != 0, is_signed, width);
        }
        ++count.cases;
        if (got != want || got_flags != want_flags) {
          mismatch(fmt::format("{}{} [{}]: got {:x} flags {:02x}, want {:x} flags {:02x}", name,
                               args, m.name, got, got_flags, want, want_flags));
        }
      };
      clear_fflags(hart);
      check_to_int("f64_to_i32", true, 32, hart.f64_to_i32(Bits<64>{a}, m.hart).to_defined().get());
      clear_fflags(hart);
      check_to_int("f64_to_ui32", false, 32,
                   hart.f64_to_ui32(Bits<64>{a}, m.hart).to_defined().get());
      clear_fflags(hart);
      check_to_int("f64_to_i64", true, 64, hart.f64_to_i64(Bits<64>{a}, m.hart).to_defined().get());
      clear_fflags(hart);
      check_to_int("f64_to_ui64", false, 64,
                   hart.f64_to_ui64(Bits<64>{a}, m.hart).to_defined().get());

      // integer -> f64 (the operand bits are reused as the integer)
      diff_f64_result(
          hart, m, [&] { return hart.i64_to_f64(Bits<64>{a}, m.hart).to_defined().get(); },
          [&] { return i64_to_f64(static_cast<int64_t>(a)).v; }, count, "i64_to_f64" + args);
      diff_f64_result(
          hart, m, [&] { return hart.ui64_to_f64(Bits<64>{a}, m.hart).to_defined().get(); },
          [&] { return ui64_to_f64(a).v; }, count, "ui64_to_f64" + args);
      const uint32_t a32 = static_cast<uint32_t>(a >> (a & 1 ? 32 : 0));
      diff_f64_result(
          hart, m, [&] { return hart.i32_to_f64(Bits<32>{a32}).to_defined().get(); },
          [&] { return i32_to_f64(static_cast<int32_t>(a32)).v; }, count,
          fmt::format("i32_to_f64({:08x})", a32));
      diff_f64_result(
          hart, m, [&] { return hart.ui32_to_f64(Bits<32>{a32}).to_defined().get(); },
          [&] { return ui32_to_f64(a32).v; }, count, fmt::format("ui32_to_f64({:08x})", a32));
      diff_f64_result(
          hart, m, [&] { return hart.f32_to_f64(Bits<32>{a32}).to_defined().get(); },
          [&] { return f32_to_f64(float32_t{a32}).v; }, count,
          fmt::format("f32_to_f64({:08x})", a32));

      // fclass.d against the classification softfloat exposes through its macros
      if (&m == &kModes[0]) {
        const uint64_t exp = (a >> 52) & 0x7ff;
        const uint64_t frac = a & 0x000fffffffffffffULL;
        const bool neg = (a >> 63) != 0;
        uint64_t want;
        if (exp == 0x7ff) {
          want = frac == 0 ? (neg ? 1u << 0 : 1u << 7) : ((frac >> 51) ? 1u << 9 : 1u << 8);
        } else if (exp == 0) {
          want = frac == 0 ? (neg ? 1u << 3 : 1u << 4) : (neg ? 1u << 2 : 1u << 5);
        } else {
          want = neg ? 1u << 1 : 1u << 6;
        }
        uint64_t got = hart.f64_classify(Bits<64>{a}).to_defined().get();
        ++count.cases;
        if (got != want) {
          mismatch(fmt::format("f64_classify{}: got {:x}, want {:x}", args, got, want));
        }
      }
    }
  }

  WARN(fmt::format("compared {} F64 results against softfloat, {} mismatches", count.cases,
                   g_mismatches));
  CHECK(g_mismatches == 0);
  CHECK(count.cases > 100000);
}
