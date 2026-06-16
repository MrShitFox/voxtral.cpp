# Homebrew packaging & homebrew-core plan

Two tiers, modelled on `whisper-cpp` (which is in homebrew-core and ships `whisper-cli`).

## Tier 1 — personal tap (works today)
`Formula/voxtral.rb` is a git-based formula that builds the bundled ggml submodule
into a static, self-contained binary. Publish by pointing a tap at this repo:

```bash
brew tap andrijdavid/voxtral https://github.com/andrijdavid/voxtral.cpp
brew install --HEAD voxtral
```

Good enough for users now; **not** acceptable for homebrew-core (see below).

## Tier 2 — homebrew-core

homebrew-core has hard rules. Status of each for voxtral.cpp:

| Requirement | whisper-cpp does | voxtral status |
| --- | --- | --- |
| **No bundled/vendored deps** ("reject PRs that bundle ggml") | `depends_on "ggml"`, `-DWHISPER_USE_SYSTEM_GGML=ON` | ❌ bundles ggml as a submodule — **needs the backend-registry migration below** |
| Stable versioned release, tarball + `sha256` (no git/HEAD, no submodule fetch — the build sandbox has **no network**) | `archive/refs/tags/vX.tar.gz` | ❌ no tagged release yet |
| Builds shared, sets `-DCMAKE_INSTALL_RPATH=#{rpath}` | yes | n/a until system-ggml build exists |
| Meaningful `test do` | downloads a tiny model + transcribes `jfk.wav` | partial (`--help`); want a real run |
| Recognised SPDX license | MIT | ✅ MIT |
| **Notability** (maintained, stable, widely used; not a personal fork) | upstream ggml-org project | ⚠️ **the real blocker** — a new personal repo will likely be rejected |

### Prerequisite code change: migrate to the ggml backend **registry** API
This is the gating technical item, and it's why voxtral can't yet use system ggml.

- The Homebrew `ggml` formula (0.15.x) exports only the **registry** API
  (`ggml_backend_reg_count`, `ggml_backend_dev_by_type`, `ggml_backend_dev_init`).
  It does **not** export the direct `ggml_backend_cpu_init` / `ggml_backend_metal_init` /
  `ggml_backend_blas_init` / `ggml_backend_cpu_set_n_threads` symbols voxtral calls —
  those live in dynamically-loaded backend modules.
- whisper.cpp/llama.cpp work with system ggml precisely because they select backends
  via the registry, not direct init.
- Good news (verified): voxtral's headers/API usage **compile cleanly against ggml
  0.15.1** — only the backend *init* path needs porting.

Migration (in `src/voxtral.cpp`, `voxtral_init_from_model` + `voxtral_model_load_from_file`):
- GPU: `ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)` → `ggml_backend_dev_init(dev, NULL)`
  (replaces `ggml_backend_metal_init` / `cuda_init` / `vk_init`).
- CPU: `ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)` → `ggml_backend_dev_init`
  (replaces `ggml_backend_cpu_init`).
- Threads: `ggml_backend_set_n_threads(backend, n)` via the generic setter
  (replaces `ggml_backend_cpu_set_n_threads` / `ggml_backend_blas_set_n_threads`).
- Drop the `GGML_USE_*` `#ifdef` gating around init (the registry discovers backends at
  runtime). Keep it working for the bundled build too — this is a net simplification.

Then add the CMake switch (was prototyped, reverted until the migration lands):
```cmake
option(VOXTRAL_USE_SYSTEM_GGML "Link a system ggml instead of the bundled submodule" OFF)
# when ON: find_package(ggml CONFIG REQUIRED); link ggml::ggml ggml::ggml-base
```

### Target formula (whisper-cpp style)
```ruby
class Voxtral < Formula
  desc "C++ implementation of the Voxtral speech-to-text model (ggml)"
  homepage "https://github.com/andrijdavid/voxtral.cpp"
  url "https://github.com/andrijdavid/voxtral.cpp/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "..."
  license "MIT"
  head "https://github.com/andrijdavid/voxtral.cpp.git", branch: "main"

  depends_on "cmake" => :build
  depends_on "ggml"
  # depends_on "sdl2"   # only once mic streaming (examples/stream) is added

  def install
    system "cmake", "-S", ".", "-B", "build",
           "-DVOXTRAL_USE_SYSTEM_GGML=ON",
           "-DBUILD_SHARED_LIBS=ON",
           "-DCMAKE_INSTALL_RPATH=#{rpath}",
           *std_cmake_args
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"
  end

  test do
    assert_match "usage", shell_output("#{bin}/voxtral --help")
  end
end
```

## Recommended path
1. Land the backend-registry migration + `VOXTRAL_USE_SYSTEM_GGML`; verify a system-ggml
   build transcribes a sample on Metal/CPU.
2. Ship Tier-1 tap; let usage/stars accrue (helps the notability case).
3. Tag `v0.1.0`, compute the tarball sha256, and only then open the homebrew-core PR
   with `brew audit --new --strict voxtral` passing.
4. Be ready for the notability objection — a tap is the realistic home until the project
   is demonstrably used; consider an org-owned repo name.
