// Model-free unit test for the streaming diagnostic SHA-256 (Sha256), used to
// report chunk-plan-invariant digests of the PCM and the encoder/adapter output
// rings. Locks the implementation against the canonical FIPS 180-2 vectors and
// exercises the two properties the streaming layer depends on:
//   * chunk-invariance  — update() split at arbitrary boundaries == one update();
//   * finalize-on-copy  — hex() may be called repeatedly and mid-stream.
#include "voxtral-stream-diagnostics.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <string>

static int g_failures = 0;

static void check(const std::string & got, const std::string & want, const char * name) {
    if (got != want) {
        std::printf("FAIL %s\n  got  %s\n  want %s\n", name, got.c_str(), want.c_str());
        ++g_failures;
    } else {
        std::printf("ok   %s\n", name);
    }
}

static std::string sha_of(const std::string & s) {
    Sha256 h;
    h.update(s.data(), s.size());
    return h.hex();
}

int main() {
    // Canonical vectors (verified with sha256sum).
    check(sha_of(""),
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty");
    check(sha_of("abc"),
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc");
    check(sha_of("The quick brown fox jumps over the lazy dog"),
          "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592", "fox");
    // 56 bytes: forces the length/padding to spill into a second 64-byte block.
    check(sha_of("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "fips-56");

    // Chunk-invariance: any split of the same bytes yields the same digest.
    {
        const std::string full = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        Sha256 a, b, c;
        a.update(full.data(), full.size());
        for (char ch : full) b.update(&ch, 1);           // one byte at a time
        c.update(full.data(), 13); c.update(full.data() + 13, full.size() - 13);  // arbitrary split
        check(b.hex(), a.hex(), "chunk-invariant-byte");
        check(c.hex(), a.hex(), "chunk-invariant-split");
    }

    // Multi-block streaming (200 KiB) matches a single update of the same bytes.
    {
        std::string big(200 * 1024, '\0');
        for (size_t i = 0; i < big.size(); ++i) big[i] = (char) (i * 31 + 7);
        Sha256 one; one.update(big.data(), big.size());
        Sha256 many;
        for (size_t off = 0; off < big.size(); off += 777)
            many.update(big.data() + off, std::min<size_t>(777, big.size() - off));
        check(many.hex(), one.hex(), "chunk-invariant-200k");
    }

    // hex() finalizes a copy: repeated calls are stable and further updates still work.
    {
        Sha256 h;
        h.update("ab", 2);
        const std::string mid = h.hex();
        check(h.hex(), mid, "hex-repeatable");
        h.update("c", 1);
        check(h.hex(), sha_of("abc"), "hex-continues-after-query");
    }

    std::printf(g_failures ? "\n%d FAILURE(S)\n" : "\nall diagnostics vectors pass\n", g_failures);
    return g_failures ? 1 : 0;
}
